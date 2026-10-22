#include "megakernel.cuh"
#include "mmvq-impl.cuh"
#include "quantize.cuh"
#include "unary.cuh"

#include <algorithm>
#include <cstdlib>
#include <vector>

// One block shape for every stage: mul_mat_vec_q_impl takes the warp from threadIdx.y, and both MMVQ instantiations
// used here run 8 warps.
static constexpr int MK_WARP_SIZE = 32;
static constexpr int MK_NWARPS    = 8;

static_assert(calc_nwarps(GGML_TYPE_Q8_0, 1, MMVQ_PARAMETERS_RDNA3_0, false) == MK_NWARPS, "hc down/inject block shape");
static_assert(calc_nwarps(GGML_TYPE_Q8_0, 1, MMVQ_PARAMETERS_RDNA3_0, true)  == MK_NWARPS, "hc up block shape");
static_assert(calc_rows_per_block(GGML_TYPE_Q8_0, 1, MMVQ_PARAMETERS_RDNA3_0, false, MK_NWARPS) == 1, "hc down rows per block");

static constexpr int MK_UP_ROWS = calc_rows_per_block(GGML_TYPE_Q8_0, 1, MMVQ_PARAMETERS_RDNA3_0, true, MK_NWARPS);

// limits of the scratch layout below
static constexpr int MK_MAX_NCOLS4  = 1024;  // rms_norm_f32<1024> vec path: one float4 per thread
static constexpr int MK_MAX_K       = 16384; // hc down K = ncols*hc
static constexpr int MK_MAX_LO      = 512;   // hc down rows = hc up K
static constexpr int MK_MAX_GRID    = 1024;

// Stages: 1 rms_norm*w + q8_1 (a task per stream), 2 down (+ inject) matvec (a task per row), 3 up matvec (a task per
// MK_UP_ROWS rows), 4 gated hc_pre (a task per MK_PRE_COLS columns). Blocks claim tasks from a per-stage counter and
// count finished ones in another; a stage waits only for the previous stage's done count. Nothing waits on a block
// that is not running, so the grid need not be co-resident (the XT shares its CUs with a display).
static constexpr int MK_STAGES   = 4;
static constexpr int MK_PRE_COLS = MK_WARP_SIZE*MK_NWARPS;

struct mk_sync {
    uint32_t claim[MK_STAGES];
    uint32_t done[MK_STAGES];
    uint32_t exit; // the last block out resets claim/done/exit for the next launch
    int      err;  // sticky: a wait gave up (GGML_CUDA_MEGAKERNEL_SPIN)
};

// scratch: [sync | xn | lo | gate | q8_1 of xn | per grid block q8_1 of act].
// The segment's intermediates live here, not in their graph tensors: gallocr reuses the memory of an intermediate
// whose last consumer is inside the segment, which is safe node by node but a race when the stages overlap.
static constexpr size_t MK_OFF_SYNC   = 0;
static constexpr size_t MK_OFF_XN     = 256;
static constexpr size_t MK_OFF_LO     = MK_OFF_XN   + MK_MAX_K*sizeof(float);
static constexpr size_t MK_OFF_GATE   = MK_OFF_LO   + MK_MAX_LO*sizeof(float);
static constexpr size_t MK_OFF_Q8_XN  = MK_OFF_GATE + MK_MAX_K*sizeof(float);
static constexpr size_t MK_OFF_Q8_ACT = MK_OFF_Q8_XN + (MK_MAX_K/QK8_1)*sizeof(block_q8_1);
static constexpr int    MK_ACT_BLOCKS = MK_MAX_LO/QK8_1;
static constexpr size_t MK_BUF_SIZE   = MK_OFF_Q8_ACT + (size_t) MK_MAX_GRID*MK_ACT_BLOCKS*sizeof(block_q8_1);
static_assert(sizeof(mk_sync) <= MK_OFF_XN, "sync block");

// stage timestamps per block: start, end of each stage's tasks and of each wait, end
static constexpr int    MK_PROF_POINTS = 8;
static constexpr size_t MK_PROF_SIZE   = (size_t) MK_MAX_GRID*MK_PROF_POINTS*sizeof(unsigned long long);

// GGML_CUDA_MEGAKERNEL_DIAG: wait timeouts go to a device log; blocks that found no task left in a stage are counted
static constexpr int MK_DIAG_LOG = 1024;
struct mk_diag_entry {
    uint32_t kind;  // 1 wait timeout (idx: stage, seen: done count)
    uint32_t block;
    uint32_t idx;
    uint32_t seen;
    uint32_t t;
};
struct mk_diag {
    uint32_t      count;
    uint32_t      launches;
    uint32_t      idle[MK_STAGES];
    mk_diag_entry log[MK_DIAG_LOG];
};
static constexpr size_t MK_DIAG_SIZE = sizeof(mk_diag);

// bounded spin: a scheduling or logic mistake sets the error flag instead of hanging the GPU
static constexpr int MK_SPIN_LIMIT = 1 << 26;

struct mk_hc_mix_params {
    const float * x;
    const float * w_norm;
    int64_t       w_norm_row;
    float         eps;
    float *       xn;
    const void *  w_down;
    float *       lo;
    uint32_t      down_rows;
    uint32_t      down_stride_row;
    ggml_cuda_mm_fusion_args_device fusion_down; // second_x: inject
    uint32_t      inject_rows;
    float         s;
    float         b;
    const void *  w_up;
    float *       gate;
    uint32_t      up_stride_row;
    float         pre_scale;
    float *       mixed;
    int           ncols;
    int           hc;
    uint3         ratio1;
    mk_sync *     sync;
    unsigned long long * prof; // GGML_CUDA_MEGAKERNEL_PROF: MK_PROF_POINTS wall clocks per block, else nullptr
    block_q8_1 *  q8_xn;
    block_q8_1 *  q8_act;
    mk_diag *     diag;       // GGML_CUDA_MEGAKERNEL_DIAG, else nullptr
    int           spin_limit; // GGML_CUDA_MEGAKERNEL_SPIN, default MK_SPIN_LIMIT
    uint32_t      chunk_down; // tasks per claim
    uint32_t      chunk_up;
};

#if defined(GGML_USE_HIP) && defined(RDNA3)
#define MK_DEVICE_CODE
#endif

#ifdef MK_DEVICE_CODE
static __device__ __forceinline__ bool mk_thread0() {
    return threadIdx.x == 0 && threadIdx.y == 0;
}

static __device__ void mk_diag_log(mk_diag * d, uint32_t kind, uint32_t idx, uint32_t seen) {
    const uint32_t k = atomicAdd(&d->count, 1u);
    if (k < MK_DIAG_LOG) {
        d->log[k] = { kind, blockIdx.x, idx, seen, (uint32_t) wall_clock64() };
    }
}

// first task of the next chunk of n tasks, the same value in every thread of the block
static __device__ __forceinline__ uint32_t mk_claim(uint32_t * claim, uint32_t n) {
    __shared__ uint32_t s_task;
    __syncthreads(); // the previous claim's s_task has been read
    if (mk_thread0()) {
        s_task = atomicAdd(claim, n);
    }
    __syncthreads();
    return s_task;
}

// publish n finished tasks: every wave releases its own stores at agent scope (in WGP mode a block's waves can sit on
// both CUs of the WGP), then one release add
static __device__ __forceinline__ void mk_done(uint32_t * done, uint32_t n) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (mk_thread0()) {
        __hip_atomic_fetch_add(done, n, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

// wait until a stage has finished all its tasks, then every wave acquires before reading what other blocks wrote
static __device__ __forceinline__ void mk_wait(const mk_hc_mix_params & p, int stage, uint32_t target) {
    if (mk_thread0()) {
        const uint32_t * done = &p.sync->done[stage];
        int spins = 0;
        uint32_t v;
        while ((v = __hip_atomic_load(done, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)) < target) {
            __builtin_amdgcn_s_sleep(1);
            if (++spins > p.spin_limit) {
                atomicExch(&p.sync->err, 1);
                if (p.diag) {
                    mk_diag_log(p.diag, 1, stage, v);
                }
                break;
            }
        }
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

// runs body(t) for every task this block claims in a stage, chunk tasks per claim
template <typename F>
static __device__ __forceinline__ void mk_tasks(const mk_hc_mix_params & p, int stage, uint32_t ntask, uint32_t chunk, F && body) {
    bool first = true;
    for (;;) {
        const uint32_t t0 = mk_claim(&p.sync->claim[stage], chunk);
        if (t0 >= ntask) {
            if (first && p.diag && mk_thread0()) {
                atomicAdd(&p.diag->idle[stage], 1u);
            }
            return;
        }
        first = false;
        const uint32_t t1 = min(t0 + chunk, ntask);
        for (uint32_t t = t0; t < t1; ++t) {
            body(t);
            __syncthreads();
        }
        mk_done(&p.sync->done[stage], t1 - t0);
    }
}
#endif // MK_DEVICE_CODE

template <bool has_inject>
__launch_bounds__(MK_WARP_SIZE*MK_NWARPS, 1)
static __global__ void mk_hc_mix_kernel(const mk_hc_mix_params p) {
#ifdef MK_DEVICE_CODE
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = MK_WARP_SIZE*warp + lane;

    unsigned long long * prof = p.prof ? p.prof + blockIdx.x*MK_PROF_POINTS : nullptr;
#define MK_PROF(k) if (prof && mk_thread0()) { prof[k] = wall_clock64(); }
    MK_PROF(0);
    if (p.diag && blockIdx.x == 0 && mk_thread0()) {
        atomicAdd(&p.diag->launches, 1u);
    }

    __shared__ float  s_part[MK_WARP_SIZE];
    __shared__ float4 s_row[MK_MAX_NCOLS4];

    const uint32_t K      = p.ncols*p.hc;
    const int      ncols4 = p.ncols/4;
    const int      nb_row = p.ncols/QK8_1;

    // stage 1, a task per stream: rms_norm_f32<1024, do_multiply, vec> owns one float4 per thread. Warp w runs that
    // kernel's warps w, w+8, w+16, w+24, so every partial sum and the final reduce happen in the same order.
    mk_tasks(p, 0, p.hc, 1, [&](uint32_t r) {
        const float4 * x4 = (const float4 *) (p.x + (int64_t) r*p.ncols);
        const float4 * m4 = (const float4 *) (p.w_norm + r*p.w_norm_row);
        float4 xs[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = MK_WARP_SIZE*(warp + MK_NWARPS*j) + lane;
            float tmp = 0.0f;
            if (i < ncols4) {
                xs[j] = x4[i];
                tmp += xs[j].x*xs[j].x + xs[j].y*xs[j].y + xs[j].z*xs[j].z + xs[j].w*xs[j].w;
            }
            tmp = warp_reduce_sum<MK_WARP_SIZE>(tmp);
            if (lane == 0) {
                s_part[warp + MK_NWARPS*j] = tmp;
            }
        }
        __syncthreads();
        const float tmp   = warp_reduce_sum<MK_WARP_SIZE>(s_part[lane]);
        const float scale = rsqrtf(tmp / p.ncols + p.eps);

        float4 * xn4 = (float4 *) (p.xn + (int64_t) r*p.ncols);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = MK_WARP_SIZE*(warp + MK_NWARPS*j) + lane;
            if (i < ncols4) {
                float4 v = make_float4(scale*xs[j].x, scale*xs[j].y, scale*xs[j].z, scale*xs[j].w);
                const float4 m = m4[i];
                v.x *= m.x;
                v.y *= m.y;
                v.z *= m.z;
                v.w *= m.w;
                xn4[i]   = v;
                s_row[i] = v;
            }
        }
        __syncthreads();

        // q8_1 of this stream, as quantize_q8_1 writes it for the [ncols*hc] src1 row
        const float * row = (const float *) s_row;
        for (int ib = warp; ib < nb_row; ib += MK_NWARPS) {
            float d;
            float sum;
            const int8_t q = quantize_q8_1_lane(row[QK8_1*ib + lane], d, sum);
            block_q8_1 & y = p.q8_xn[r*nb_row + ib];
            y.qs[lane] = q;
            if (lane == 0) {
                y.ds = make_half2(d, sum);
            }
        }
    });
    MK_PROF(1);
    mk_wait(p, 0, p.hc);
    MK_PROF(2);

    // stage 2: down matvec, inject in the second grid row like the MMVQ pair launch
    const uint32_t ntask_down = p.down_rows + (has_inject ? p.inject_rows : 0);
    mk_tasks(p, 1, ntask_down, p.chunk_down, [&](uint32_t t) {
        const uint3 bi = t < p.down_rows ? make_uint3(t, 0, 0) : make_uint3(t - p.down_rows, 1, 0);
        mul_mat_vec_q_impl<GGML_TYPE_Q8_0, 1, has_inject, false, false>(
            p.w_down, p.q8_xn, nullptr, p.fusion_down, p.lo, K, make_uint3(0, 0, 0), p.down_stride_row,
            GGML_PAD(K, MATRIX_ROW_PADDING)/QK8_1, p.down_rows, p.ratio1, 0, 0, 0, p.ratio1, 0, 0, 0, 0,
            bi, has_inject ? 2 : 1);
    });
    MK_PROF(3);
    mk_wait(p, 1, ntask_down);
    MK_PROF(4);

    // stage 3: up matvec, MK_UP_ROWS rows of gate per task. A block quantizes silu(s*lo + b) into its own q8_1 copy
    // before its first task.
    {
        block_q8_1 * q8_act = p.q8_act + blockIdx.x*MK_ACT_BLOCKS;
        bool act_ready = false;
        const ggml_cuda_mm_fusion_args_device no_fusion{};
        mk_tasks(p, 2, K/MK_UP_ROWS, p.chunk_up, [&](uint32_t t) {
            if (!act_ready) {
                for (int ib = warp; ib < (int) p.down_rows/QK8_1; ib += MK_NWARPS) {
                    const int i = QK8_1*ib + lane;
                    const float v = ggml_cuda_op_silu_single(p.s*p.lo[i] + p.b);
                    float d;
                    float sum;
                    const int8_t q = quantize_q8_1_lane(v, d, sum);
                    q8_act[ib].qs[lane] = q;
                    if (lane == 0) {
                        q8_act[ib].ds = make_half2(d, sum);
                    }
                }
                __syncthreads();
                act_ready = true;
            }
            mul_mat_vec_q_impl<GGML_TYPE_Q8_0, 1, false, true, false>(
                p.w_up, q8_act, nullptr, no_fusion, p.gate, p.down_rows, make_uint3(0, 0, 0), p.up_stride_row,
                GGML_PAD(p.down_rows, MATRIX_ROW_PADDING)/QK8_1, K, p.ratio1, 0, 0, 0, p.ratio1, 0, 0, 0, 0,
                make_uint3(t, 0, 0), 1);
        });
    }
    MK_PROF(5);
    mk_wait(p, 2, K/MK_UP_ROWS);
    MK_PROF(6);

    // stage 4: dsv4_hc_pre_f32<gated>, a column per thread
    mk_tasks(p, 3, (p.ncols + MK_PRE_COLS - 1)/MK_PRE_COLS, 1, [&](uint32_t t) {
        const int64_t i0 = (int64_t) MK_PRE_COLS*t + tid;
        if (i0 < p.ncols) {
            float sum = 0.0f;
            for (int64_t ih = 0; ih < p.hc; ++ih) {
                const float xv = p.xn[i0 + ih*p.ncols];
                float wv;
                wv = 1.0f / (1.0f + expf(-p.gate[i0 + ih*p.ncols]));
                sum += xv * wv;
            }
            p.mixed[i0] = p.pre_scale * sum;
        }
    });
    MK_PROF(7);
#undef MK_PROF

    // the last block out resets the counters; the kernel boundary publishes the reset to the next launch
    __syncthreads();
    if (mk_thread0() && atomicAdd(&p.sync->exit, 1u) == gridDim.x - 1) {
        for (int k = 0; k < MK_STAGES; ++k) {
            __hip_atomic_store(&p.sync->claim[k], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&p.sync->done[k],  0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __hip_atomic_store(&p.sync->exit, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif // MK_DEVICE_CODE
}

bool ggml_cuda_megakernel_enabled(int cc) {
    // opt-in (GGML_CUDA_MEGAKERNEL=1) until the rare cross-block race in the hc mixer is fixed
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MEGAKERNEL");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled && GGML_CUDA_CC_IS_RDNA3_0(cc);
}

static bool mk_diag_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MEGAKERNEL_DIAG");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled;
}

void ggml_cuda_megakernel_report(ggml_backend_cuda_context & ctx) {
    if (ctx.mk_buf == nullptr || !mk_diag_enabled()) {
        return;
    }
    ggml_cuda_set_device(ctx.device);
    CUDA_CHECK(cudaDeviceSynchronize());
    mk_sync sync;
    CUDA_CHECK(cudaMemcpy(&sync, ctx.mk_buf + MK_OFF_SYNC, sizeof(sync), cudaMemcpyDeviceToHost));
    mk_diag d;
    CUDA_CHECK(cudaMemcpy(&d, ctx.mk_buf + MK_BUF_SIZE + MK_PROF_SIZE, sizeof(d), cudaMemcpyDeviceToHost));
    fprintf(stderr, "megakernel diag device %d: grid %d, %u launches, err %d, %u timeouts, blocks without a task per "
            "stage %u/%u/%u/%u\n", ctx.device, ctx.mk_grid, d.launches, sync.err, d.count, d.idle[0], d.idle[1], d.idle[2],
            d.idle[3]);
    for (uint32_t k = 0; k < std::min<uint32_t>(d.count, 8); ++k) {
        const mk_diag_entry & e = d.log[k];
        fprintf(stderr, "  timeout block %u stage %u done %u t %u\n", e.block, e.idx, e.seen, e.t);
    }
}

static bool mk_prof_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MEGAKERNEL_PROF");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled;
}

// per device: summed stage times over the profiled launches, printed every 1000 launches
struct mk_prof_sums {
    double  us[MK_PROF_POINTS] = {};
    int64_t n = 0;
};

static bool mk_check_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MEGAKERNEL_CHECK");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled;
}

void ggml_cuda_megakernel_prepare(ggml_backend_cuda_context & ctx) {
    if (ctx.mk_buf != nullptr || !ggml_cuda_megakernel_enabled(ggml_cuda_info().devices[ctx.device].cc)) {
        return;
    }
    ggml_cuda_set_device(ctx.device);

    // Blocks per WGP (GGML_CUDA_MEGAKERNEL_BPW, default 4), capped at the occupancy. The grid need not be co-resident
    // for correctness; GGML_CUDA_MEGAKERNEL_OVERSUB=1 drops the cap to test exactly that.
    static const int bpw_env = [] {
        const char * env = getenv("GGML_CUDA_MEGAKERNEL_BPW");
        return env ? atoi(env) : 4;
    }();
    static const bool oversub = [] {
        const char * env = getenv("GGML_CUDA_MEGAKERNEL_OVERSUB");
        return env != nullptr && atoi(env) != 0;
    }();
    int occ_a = 0;
    int occ_b = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_a, mk_hc_mix_kernel<true>,  MK_WARP_SIZE*MK_NWARPS, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_b, mk_hc_mix_kernel<false>, MK_WARP_SIZE*MK_NWARPS, 0));
    const int bpw  = oversub ? bpw_env : std::min({ bpw_env, occ_a, occ_b });
    const int grid = std::min(bpw*ggml_cuda_info().devices[ctx.device].nsm, MK_MAX_GRID);
    if (bpw < 1) {
        GGML_LOG_WARN("%s: megakernel not co-resident on device %d (occupancy %d/%d), disabled\n", __func__, ctx.device, occ_a, occ_b);
        return;
    }

    CUDA_CHECK(cudaMalloc((void **) &ctx.mk_buf, MK_BUF_SIZE + MK_PROF_SIZE + MK_DIAG_SIZE));
    CUDA_CHECK(cudaMemset(ctx.mk_buf, 0, MK_BUF_SIZE + MK_PROF_SIZE + MK_DIAG_SIZE));
    ctx.mk_grid = grid;
    GGML_LOG_INFO("%s: device %d: megakernel grid %d (%d blocks per WGP, occupancy %d/%d%s)\n",
                  __func__, ctx.device, grid, bpw, occ_a, occ_b, oversub ? ", oversubscribed" : "");
}

static bool mk_contiguous_f32(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

bool ggml_cuda_mk_hc_mix_supported(const ggml_cuda_mk_hc_mix & m) {
    const int64_t ncols = m.x->ne[0];
    const int64_t hc    = m.x->ne[1];
    const int64_t K     = ncols*hc;
    const int64_t nlo   = m.w_down->ne[1];

    // the rms_norm vec path that runs 1024 threads (ncols/4 in (512, 1024])
    if (!mk_contiguous_f32(m.x) || ggml_nrows(m.x) != hc || ncols % QK8_1 != 0 || ncols % MK_UP_ROWS != 0 ||
            ncols/4 <= 512 || ncols/4 > MK_MAX_NCOLS4 || K > MK_MAX_K) {
        return false;
    }
    if (!mk_contiguous_f32(m.w_norm) || m.w_norm->ne[0] != ncols || (m.w_norm->ne[1] != 1 && m.w_norm->ne[1] != hc) ||
            ggml_nrows(m.w_norm) != m.w_norm->ne[1]) {
        return false;
    }
    if (((uintptr_t) m.x->data | (uintptr_t) m.w_norm->data) % 16 != 0) {
        return false;
    }
    if (!mk_contiguous_f32(m.xn) || ggml_nelements(m.xn) != K) {
        return false;
    }

    // MMVQ Q8_0: down at K >= 2048 does not use the small-K rows per block, up at K = nlo <= 512 does
    static const bool small_k_env = [] {
        const char * env = getenv("GGML_CUDA_MMVQ_SMALL_K");
        return env == nullptr || atoi(env) != 0;
    }();
    const auto is_q8_0_matrix = [](const ggml_tensor * w, int64_t k) {
        return w->type == GGML_TYPE_Q8_0 && w->ne[0] == k && ggml_is_matrix(w) && w->nb[0] == ggml_type_size(w->type);
    };
    if (!small_k_env || !is_q8_0_matrix(m.w_down, K) || K < 2048 || nlo % QK8_1 != 0 || nlo > MK_MAX_LO) {
        return false;
    }
    if (!mk_contiguous_f32(m.lo) || ggml_nelements(m.lo) != nlo) {
        return false;
    }
    if (m.w_inject) {
        if (!is_q8_0_matrix(m.w_inject, K) || m.w_inject->nb[1] != m.w_down->nb[1] || m.w_inject->ne[1] > nlo ||
                !mk_contiguous_f32(m.inject) || ggml_nelements(m.inject) != m.w_inject->ne[1]) {
            return false;
        }
    }
    if (!mk_contiguous_f32(m.act) || ggml_nelements(m.act) != nlo) {
        return false;
    }
    if (!is_q8_0_matrix(m.w_up, nlo) || m.w_up->ne[1] != K || !mk_contiguous_f32(m.gate) || ggml_nelements(m.gate) != K) {
        return false;
    }
    if (!mk_contiguous_f32(m.mixed) || ggml_nelements(m.mixed) != ncols) {
        return false;
    }
    return true;
}

void ggml_cuda_mk_hc_mix_run(ggml_backend_cuda_context & ctx, const ggml_cuda_mk_hc_mix & m) {
    mk_hc_mix_params p{};
    p.x          = (const float *) m.x->data;
    p.w_norm     = (const float *) m.w_norm->data;
    p.w_norm_row = m.w_norm->ne[1] == 1 ? 0 : m.w_norm->ne[0];
    p.eps        = m.eps;
    p.xn         = (float *) (ctx.mk_buf + MK_OFF_XN);
    p.w_down     = m.w_down->data;
    p.lo         = (float *) (ctx.mk_buf + MK_OFF_LO);
    p.down_rows  = m.w_down->ne[1];
    p.down_stride_row = m.w_down->nb[1] / ggml_type_size(m.w_down->type);
    if (m.w_inject) {
        p.fusion_down.second_x     = m.w_inject->data;
        p.fusion_down.second_dst   = (float *) m.inject->data;
        p.fusion_down.second_nrows = m.w_inject->ne[1];
        p.inject_rows              = m.w_inject->ne[1];
    }
    p.s             = m.s;
    p.b             = m.b;
    p.w_up          = m.w_up->data;
    p.gate          = (float *) (ctx.mk_buf + MK_OFF_GATE);
    p.up_stride_row = m.w_up->nb[1] / ggml_type_size(m.w_up->type);
    p.pre_scale     = m.pre_scale;
    p.mixed         = (float *) m.mixed->data;
    p.ncols         = m.x->ne[0];
    p.hc            = m.x->ne[1];
    p.ratio1        = init_fastdiv_values(1);
    p.sync          = (mk_sync *) (ctx.mk_buf + MK_OFF_SYNC);
    p.prof          = mk_prof_enabled() ? (unsigned long long *) (ctx.mk_buf + MK_BUF_SIZE) : nullptr;
    p.q8_xn         = (block_q8_1 *) (ctx.mk_buf + MK_OFF_Q8_XN);
    p.diag          = mk_diag_enabled() ? (mk_diag *) (ctx.mk_buf + MK_BUF_SIZE + MK_PROF_SIZE) : nullptr;
    static const int spin_limit = [] {
        const char * env = getenv("GGML_CUDA_MEGAKERNEL_SPIN");
        return env ? atoi(env) : MK_SPIN_LIMIT;
    }();
    p.spin_limit    = spin_limit;
    static const int chunk_down = [] {
        const char * env = getenv("GGML_CUDA_MEGAKERNEL_CHUNK_DOWN");
        return env ? atoi(env) : 2;
    }();
    static const int chunk_up = [] {
        const char * env = getenv("GGML_CUDA_MEGAKERNEL_CHUNK_UP");
        return env ? atoi(env) : 4;
    }();
    p.chunk_down    = std::max(chunk_down, 1);
    p.chunk_up      = std::max(chunk_up, 1);
    p.q8_act        = (block_q8_1 *) (ctx.mk_buf + MK_OFF_Q8_ACT);

    cudaStream_t stream = ctx.stream();
    const dim3 block_dims(MK_WARP_SIZE, MK_NWARPS, 1);
    if (m.w_inject) {
        mk_hc_mix_kernel<true><<<ctx.mk_grid, block_dims, 0, stream>>>(p);
    } else {
        mk_hc_mix_kernel<false><<<ctx.mk_grid, block_dims, 0, stream>>>(p);
    }
    CUDA_CHECK(cudaGetLastError());

    if (p.prof) {
        // needs GGML_CUDA_DISABLE_GRAPHS=1: read back after every launch. wall_clock64 runs at 100 MHz on RDNA3.
        static mk_prof_sums sums[GGML_CUDA_MAX_DEVICES];
        std::vector<unsigned long long> t((size_t) ctx.mk_grid*MK_PROF_POINTS);
        CUDA_CHECK(cudaMemcpyAsync(t.data(), p.prof, t.size()*sizeof(t[0]), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        unsigned long long lo[MK_PROF_POINTS];
        unsigned long long hi[MK_PROF_POINTS] = {};
        std::fill(lo, lo + MK_PROF_POINTS, ~0ull);
        for (int blk = 0; blk < ctx.mk_grid; ++blk) {
            for (int k = 0; k < MK_PROF_POINTS; ++k) {
                lo[k] = std::min(lo[k], t[blk*MK_PROF_POINTS + k]);
                hi[k] = std::max(hi[k], t[blk*MK_PROF_POINTS + k]);
            }
        }
        // per stage: last task end to last wait end is the wait; stage k runs from the last wait before it
        const double d[MK_PROF_POINTS] = {
            (double) (hi[1] - lo[0]), (double) (hi[2] - hi[1]), (double) (hi[3] - hi[2]), (double) (hi[4] - hi[3]),
            (double) (hi[5] - hi[4]), (double) (hi[6] - hi[5]), (double) (hi[7] - hi[6]), (double) (hi[7] - lo[0]) };
        mk_prof_sums & ps = sums[ctx.device];
        for (int k = 0; k < MK_PROF_POINTS; ++k) {
            ps.us[k] += d[k]/100.0;
        }
        if (++ps.n % 1000 == 0) {
            GGML_LOG_INFO("megakernel prof device %d, %lld launches, us: norm %.2f wait %.2f down %.2f wait %.2f up %.2f "
                          "wait %.2f pre %.2f total %.2f\n", ctx.device, (long long) ps.n, ps.us[0]/ps.n, ps.us[1]/ps.n,
                          ps.us[2]/ps.n, ps.us[3]/ps.n, ps.us[4]/ps.n, ps.us[5]/ps.n, ps.us[6]/ps.n, ps.us[7]/ps.n);
        }
    }

    if (p.diag) {
        // graphs off: report every 2000 launches on this device, so a stalled or killed run still shows its events
        static int n_launch[GGML_CUDA_MAX_DEVICES] = {};
        cudaStreamCaptureStatus status;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
        if (status == cudaStreamCaptureStatusNone && ++n_launch[ctx.device] % 2000 == 0) {
            ggml_cuda_megakernel_report(ctx);
        }
    }

    if (mk_check_enabled()) {
        cudaStreamCaptureStatus status;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
        if (status == cudaStreamCaptureStatusNone) {
            int err = 0;
            CUDA_CHECK(cudaMemcpyAsync(&err, &p.sync->err, sizeof(err), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (err != 0) {
                GGML_ABORT("megakernel wait timed out on device %d", ctx.device);
            }
        }
    }
}
