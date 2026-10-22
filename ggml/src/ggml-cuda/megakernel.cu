#include "megakernel.cuh"
#include "convert.cuh"
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
// stage 3 (up matvec): rows per warp and per task
static constexpr int MK_UP_WARP_ROWS = 4;
static constexpr int MK_UP_TASK_ROWS = MK_NWARPS*MK_UP_WARP_ROWS;
static_assert(MK_UP_ROWS == MK_NWARPS && MK_NWARPS > 1, "stage 3 sums in the order of the small-K kernel with nwarps rows");

// limits of the scratch layout below
static constexpr int MK_MAX_NCOLS4  = 1024;  // rms_norm_f32<1024> vec path: one float4 per thread
static constexpr int MK_MAX_K       = 16384; // hc down K = ncols*hc
static constexpr int MK_MAX_LO      = 512;   // hc down rows = hc up K
static constexpr int MK_MAX_GRID    = 1024;

// stage 2 (down matvec, a row per task): K loop iterations of the 256-thread MMVQ block, held in registers at once
static constexpr int MK_DOWN_ITERS = MK_MAX_K/QK8_0 / (VDR_Q8_0_Q8_1_MMVQ*MK_NWARPS*MK_WARP_SIZE/QI8_0);

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
    uint32_t      gdn_launches;
    uint32_t      idle[MK_STAGES];
    mk_diag_entry log[MK_DIAG_LOG];
};
static constexpr size_t MK_DIAG_SIZE = sizeof(mk_diag);

// gated delta net segment scratch: attention and gated output [128*H] each, the per-head and all-task counters
static constexpr int    MK_GDN_S       = 128;
static constexpr int    MK_GDN_MAX_H   = 64;
static constexpr int    MK_GDN_MAX_K   = 4096;  // hc_mixed length (alpha/beta matvec K)
static constexpr int    MK_GDN_QUARTERS = 4;    // tasks per v head, 32 columns each (8 warps x 4 columns)
static constexpr size_t MK_GDN_OFF_ATTN = 0;
static constexpr size_t MK_GDN_OFF_OUT  = MK_GDN_OFF_ATTN + (size_t) MK_GDN_MAX_H*MK_GDN_S*sizeof(float);
static constexpr size_t MK_GDN_OFF_CNT  = MK_GDN_OFF_OUT  + (size_t) MK_GDN_MAX_H*MK_GDN_S*sizeof(float);
static constexpr size_t MK_GDN_SIZE     = MK_GDN_OFF_CNT + (2*MK_GDN_MAX_H + 1)*sizeof(uint32_t);
static constexpr size_t MK_GDN_BASE     = MK_BUF_SIZE + MK_PROF_SIZE + MK_DIAG_SIZE;

// router + topk (topk-moe.cu): a counter, then the logits
static constexpr size_t MK_ROUTER_BASE = (MK_GDN_BASE + MK_GDN_SIZE + 255)/256*256;
static constexpr size_t MK_ROUTER_SIZE = 256 + 1024*sizeof(float);

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
    block_q8_1 *  q8_mixed; // q8_1 copy of mixed for the next MMVQ (nullptr: none); ncols % MATRIX_ROW_PADDING == 0
    mk_diag *     diag;       // GGML_CUDA_MEGAKERNEL_DIAG, else nullptr
    int           spin_limit; // GGML_CUDA_MEGAKERNEL_SPIN, default MK_SPIN_LIMIT
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

    const uint32_t K      = p.ncols*p.hc;
    const int      ncols4 = p.ncols/4;
    const int      nb_row = p.ncols/QK8_1;

    constexpr int qi  = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qi;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;

    // stage 2's weights do not depend on stage 1: a block claims its first down task and loads that row's weights
    // before waiting for stage 1 (not before stage 1 itself: the norm's loads would queue behind 3.5 MB of weights). A task is one row, as in the MMVQ pair launch (inject rows after the down rows); thread tid holds
    // the blocks tid/4 + 64*it, quants 2*(tid%4), of every K loop iteration it.
    const uint32_t ntask_down = p.down_rows + (has_inject ? p.inject_rows : 0);
    const int      bpr_down   = K/QK8_0;
    constexpr int  down_iter  = vdr*MK_NWARPS*MK_WARP_SIZE/qi;
    const int      kqs_down   = vdr * (tid % (qi/vdr));
    int  dv[MK_DOWN_ITERS][vdr];
    half dd[MK_DOWN_ITERS];
    auto load_down = [&](uint32_t t) {
        const bool second = t >= p.down_rows;
        const block_q8_0 * w = (const block_q8_0 *) (second ? p.fusion_down.second_x : p.w_down) +
            (int64_t) (second ? t - p.down_rows : t)*p.down_stride_row;
#pragma unroll
        for (int it = 0; it < MK_DOWN_ITERS; ++it) {
            const int kbx = tid/(qi/vdr) + down_iter*it;
            if (kbx < bpr_down) {
#pragma unroll
                for (int k = 0; k < vdr; ++k) {
                    dv[it][k] = get_int_b2(w[kbx].qs, kqs_down + k);
                }
                dd[it] = w[kbx].d;
            }
        }
    };

    // stage 1, a task per (stream, MK_NWARPS q8_1 blocks of it): rms_norm_f32<1024, do_multiply, vec> owns one float4
    // per thread. Every task recomputes its stream's sum of squares (the row is 10 KB from L2) with warp w running that
    // kernel's warps w, w+8, w+16, w+24, so every partial sum and the final reduce happen in the same order; then warp w
    // scales and quantizes one q8_1 block. (scale*x)*w per element is what the float4 path computes.
    const uint32_t tasks_per_row = (nb_row + MK_NWARPS - 1)/MK_NWARPS;
    mk_tasks(p, 0, p.hc*tasks_per_row, 1, [&](uint32_t t) {
        const uint32_t r  = t / tasks_per_row;
        const int      ib = MK_NWARPS*(t % tasks_per_row) + warp;
        const float  * xr = p.x + (int64_t) r*p.ncols;
        const float4 * x4 = (const float4 *) xr;

        // all loads first, so they are in flight together
        float4 xs[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = MK_WARP_SIZE*(warp + MK_NWARPS*j) + lane;
            if (i < ncols4) {
                xs[j] = x4[i];
            }
        }
        const int   ie = QK8_1*ib + lane;
        const float xe = ib < nb_row ? xr[ie] : 0.0f;
        const float me = ib < nb_row ? p.w_norm[r*p.w_norm_row + ie] : 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = MK_WARP_SIZE*(warp + MK_NWARPS*j) + lane;
            float tmp = 0.0f;
            if (i < ncols4) {
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

        if (ib < nb_row) {
            float v = scale*xe;
            v *= me;
            p.xn[(int64_t) r*p.ncols + ie] = v;

            // q8_1 of this stream, as quantize_q8_1 writes it for the [ncols*hc] src1 row
            float d;
            float sum;
            const int8_t q = quantize_q8_1_lane(v, d, sum);
            block_q8_1 & y = p.q8_xn[r*nb_row + ib];
            y.qs[lane] = q;
            if (lane == 0) {
                y.ds = make_half2(d, sum);
            }
        }
    });
    MK_PROF(1);
    uint32_t td = mk_claim(&p.sync->claim[1], 1);
    if (td < ntask_down) {
        load_down(td);
    }
    mk_wait(p, 0, p.hc*tasks_per_row);
    MK_PROF(2);

    // stage 2: down matvec, summed in mul_mat_vec_q's order: per thread over the K loop, warp 0 adds warps 1..7, the
    // warp butterfly, and the fusion path's zero bias with inject
    if (td >= ntask_down) {
        if (p.diag && mk_thread0()) {
            atomicAdd(&p.diag->idle[1], 1u);
        }
    } else {
        __shared__ float s_red[MK_NWARPS-1][MK_WARP_SIZE];
        for (;;) {
            float tmp = 0.0f;
#pragma unroll
            for (int it = 0; it < MK_DOWN_ITERS; ++it) {
                const int kbx = tid/(qi/vdr) + down_iter*it;
                if (kbx < bpr_down) {
                    int u[vdr];
#pragma unroll
                    for (int k = 0; k < vdr; ++k) {
                        u[k] = get_int_b4(p.q8_xn[kbx].qs, kqs_down + k);
                    }
                    tmp += vec_dot_q8_0_q8_1_impl<float, vdr>(dv[it], u, dd[it], __low2half(p.q8_xn[kbx].ds));
                }
            }
            if (warp > 0) {
                s_red[warp-1][lane] = tmp;
            }
            __syncthreads();
            if (warp == 0) {
#pragma unroll
                for (int l = 0; l < MK_NWARPS-1; ++l) {
                    tmp += s_red[l][lane];
                }
                tmp = warp_reduce_sum<MK_WARP_SIZE>(tmp);
                if (lane == 0) {
                    float result = tmp;
                    if constexpr (has_inject) {
                        result += 0.0f;
                    }
                    if (td < p.down_rows) {
                        p.lo[td] = result;
                    } else {
                        p.fusion_down.second_dst[td - p.down_rows] = result;
                    }
                }
            }
            mk_done(&p.sync->done[1], 1);
            td = mk_claim(&p.sync->claim[1], 1);
            if (td >= ntask_down) {
                break;
            }
            load_down(td);
        }
    }
    MK_PROF(3);

    // stage 3: up matvec, MK_NWARPS*MK_UP_WARP_ROWS rows of gate per task. A block quantizes silu(s*lo + b) into its
    // own q8_1 copy before its first task.
    //
    // The standalone small-K kernel gives a block MK_UP_ROWS rows, but with K <= 512 (<= 16 Q8_0 blocks a row) only its
    // first 4*blocks_per_row threads have work. Here a warp owns whole rows and keeps all of them in flight; per row it
    // sums in that kernel's order: thread tid = lane (+32) takes block tid/4, quants 2*(tid%4); warp 0's partial gets
    // warp 1's then the other warps' zeros added; then the warp butterfly. The weights do not depend on stage 2: the
    // first task is claimed and its weights loaded before waiting for stage 2.
    const uint32_t ntask_up = K/MK_UP_TASK_ROWS;
    const int bpr = p.down_rows/QK8_0;
    const int kb0 = lane / (qi/vdr);
    const int kb1 = kb0 + MK_WARP_SIZE / (qi/vdr);
    const int kqs = vdr * (lane % (qi/vdr));

    int  wv0[MK_UP_WARP_ROWS][vdr];
    int  wv1[MK_UP_WARP_ROWS][vdr];
    half wd0[MK_UP_WARP_ROWS];
    half wd1[MK_UP_WARP_ROWS];
    auto load_up = [&](uint32_t t) {
        const int row0 = MK_UP_TASK_ROWS*t + MK_UP_WARP_ROWS*warp;
#pragma unroll
        for (int i = 0; i < MK_UP_WARP_ROWS; ++i) {
            const block_q8_0 * w = (const block_q8_0 *) p.w_up + (row0 + i)*p.up_stride_row;
            if (kb0 < bpr) {
#pragma unroll
                for (int k = 0; k < vdr; ++k) {
                    wv0[i][k] = get_int_b2(w[kb0].qs, kqs + k);
                }
                wd0[i] = w[kb0].d;
            }
            if (kb1 < bpr) {
#pragma unroll
                for (int k = 0; k < vdr; ++k) {
                    wv1[i][k] = get_int_b2(w[kb1].qs, kqs + k);
                }
                wd1[i] = w[kb1].d;
            }
        }
    };

    uint32_t tu = mk_claim(&p.sync->claim[2], 1);
    if (tu < ntask_up) {
        load_up(tu);
    }
    mk_wait(p, 1, ntask_down);
    MK_PROF(4);

    if (tu >= ntask_up) {
        if (p.diag && mk_thread0()) {
            atomicAdd(&p.diag->idle[2], 1u);
        }
    } else {
        block_q8_1 * q8_act = p.q8_act + blockIdx.x*MK_ACT_BLOCKS;
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

        int  u0[vdr];
        int  u1[vdr];
        half ud0 = 0.0f;
        half ud1 = 0.0f;
        if (kb0 < bpr) {
#pragma unroll
            for (int k = 0; k < vdr; ++k) {
                u0[k] = get_int_b4(q8_act[kb0].qs, kqs + k);
            }
            ud0 = __low2half(q8_act[kb0].ds);
        }
        if (kb1 < bpr) {
#pragma unroll
            for (int k = 0; k < vdr; ++k) {
                u1[k] = get_int_b4(q8_act[kb1].qs, kqs + k);
            }
            ud1 = __low2half(q8_act[kb1].ds);
        }

        for (;;) {
            const int row0 = MK_UP_TASK_ROWS*tu + MK_UP_WARP_ROWS*warp;
#pragma unroll
            for (int i = 0; i < MK_UP_WARP_ROWS; ++i) {
                float tmp0 = 0.0f;
                float tmp1 = 0.0f;
                if (kb0 < bpr) {
                    tmp0 += vec_dot_q8_0_q8_1_impl<float, vdr>(wv0[i], u0, wd0[i], ud0);
                }
                if (kb1 < bpr) {
                    tmp1 += vec_dot_q8_0_q8_1_impl<float, vdr>(wv1[i], u1, wd1[i], ud1);
                }
                float sum = tmp0;
                sum += tmp1;
#pragma unroll
                for (int l = 1; l < MK_NWARPS-1; ++l) {
                    sum += 0.0f;
                }
                sum = warp_reduce_sum<MK_WARP_SIZE>(sum);
                if (lane == 0) {
                    p.gate[row0 + i] = sum;
                }
            }
            mk_done(&p.sync->done[2], 1);
            tu = mk_claim(&p.sync->claim[2], 1);
            if (tu >= ntask_up) {
                break;
            }
            load_up(tu);
        }
    }
    MK_PROF(5);
    mk_wait(p, 2, K/MK_UP_TASK_ROWS);
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
            const float v = p.pre_scale * sum;
            p.mixed[i0] = v;
            if (p.q8_mixed) { // whole warps: ncols is a multiple of the warp size here
                float d;
                float s;
                const int8_t q = quantize_q8_1_lane(v, d, s);
                block_q8_1 & y = p.q8_mixed[i0/QK8_1];
                y.qs[i0 % QK8_1] = q;
                if (i0 % QK8_1 == 0) {
                    y.ds = make_half2(d, s);
                }
            }
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
    // on by default; GGML_CUDA_MEGAKERNEL=0 turns every segment off
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MEGAKERNEL");
        return env == nullptr || atoi(env) != 0;
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
    fprintf(stderr, "megakernel diag device %d: %u gdn launches\n", ctx.device, d.gdn_launches);
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

    // Blocks per WGP (GGML_CUDA_MEGAKERNEL_BPW, default 8), capped at the occupancy. The grid need not be co-resident
    // for correctness; GGML_CUDA_MEGAKERNEL_OVERSUB=1 drops the cap to test exactly that.
    static const int bpw_env = [] {
        const char * env = getenv("GGML_CUDA_MEGAKERNEL_BPW");
        return env ? atoi(env) : 8;
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

    CUDA_CHECK(cudaMalloc((void **) &ctx.mk_buf, MK_ROUTER_BASE + MK_ROUTER_SIZE));
    CUDA_CHECK(cudaMemset(ctx.mk_buf, 0, MK_ROUTER_BASE + MK_ROUTER_SIZE));
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
    if (!mk_contiguous_f32(m.x) || ggml_nrows(m.x) != hc || ncols % QK8_1 != 0 || K % MK_UP_TASK_ROWS != 0 ||
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
    p.q8_act        = (block_q8_1 *) (ctx.mk_buf + MK_OFF_Q8_ACT);
    p.q8_mixed      = nullptr;
    if (p.ncols % MATRIX_ROW_PADDING == 0 && ggml_is_contiguous(m.mixed) && ggml_nrows(m.mixed) == 1) {
        p.q8_mixed = (block_q8_1 *) ctx.q8_1_cache_reserve(m.mixed->data, m.mixed->ne, m.mixed->nb, ggml_nbytes(m.mixed),
                                                           p.ncols/QK8_1*sizeof(block_q8_1));
    }

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
            fprintf(stderr, "megakernel prof device %d, %lld launches, us: norm %.2f wait %.2f down %.2f wait %.2f up %.2f "
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

// ---------------------------------------------------------------------------------------------------------------------
// gated delta net segment

struct mk_gdn_params {
    const int32_t * s_copy;
    const float *   conv_src;
    int64_t         conv_src_row;   // floats between cache_r rows
    float *         conv_dst;
    const float *   qkv;
    const float *   conv_w;
    const float *   state_src;
    int64_t         state_src_row;
    float *         state_dst;
    float           eps_q;
    float           scale_q;
    float           eps_k;
    float           scale_k;
    const float *   x;
    int             K;
    const void *    w_alpha;
    const float *   alpha_bias;
    const float *   ssm_a;
    const void *    w_beta;
    uint32_t        w_stride_row;
    float           gdn_scale;
    const float *   z;
    const float *   w_onorm;
    float           eps_o;
    float *         out;
    float *         attn;           // scratch [128*H]
    float *         out_s;          // scratch [128*H]
    block_q8_1 *    q8_out;         // q8_1 copy of out as a [128*H] row (nullptr: none)
    uint32_t *      cnt_v;          // [H]
    uint32_t *      cnt_k;          // [HK]
    uint32_t *      cnt_all;
    mk_diag *       diag;           // GGML_CUDA_MEGAKERNEL_DIAG: launch count, else nullptr
    int             H;
    int             HK;
};

#ifdef MK_DEVICE_CODE
// the per-node kernels' element functions (unary.cu op_softplus, op_sigmoid)
static __device__ __forceinline__ float mk_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}
static __device__ __forceinline__ float mk_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// true in every thread of the block when this block is the n-th to arrive at *cnt (which it then resets). The
// release covers the block's earlier loads too: the last block may overwrite what the others read.
static __device__ __forceinline__ bool mk_last_of(uint32_t * cnt, uint32_t n) {
    __shared__ bool s_last;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (mk_thread0()) {
        s_last = __hip_atomic_fetch_add(cnt, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) == n - 1;
        if (s_last) {
            __hip_atomic_store(cnt, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
    __syncthreads();
    const bool last = s_last;
    if (last) {
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    return last;
}
#endif // MK_DEVICE_CODE

__launch_bounds__(MK_WARP_SIZE*MK_NWARPS, 1)
static __global__ void mk_gdn_kernel(const mk_gdn_params p) {
#ifdef MK_DEVICE_CODE
    constexpr int S             = MK_GDN_S;
    constexpr int cols_per_task = S/MK_GDN_QUARTERS;
    constexpr int cols_per_warp = 4;                        // gated_delta_net_cuda<128>: gdn_cols_per_warp
    constexpr int lanes_per_col = MK_WARP_SIZE/cols_per_warp;
    constexpr int rows_per_lane = S/lanes_per_col;
    static_assert(cols_per_task == MK_NWARPS*cols_per_warp, "a task is one column per 8 lanes");

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = MK_WARP_SIZE*warp + lane;

    if (p.diag && blockIdx.x == 0 && mk_thread0()) {
        atomicAdd(&p.diag->gdn_launches, 1u);
    }

    const int h  = blockIdx.x / MK_GDN_QUARTERS;  // v head
    const int j  = blockIdx.x % MK_GDN_QUARTERS;  // column quarter
    const int kh = h % p.HK;                      // q/k head (gated_delta_net_cuda: h_idx mod neqk1)

    const int ch_q = kh*S;
    const int ch_k = p.HK*S + kh*S;
    const int ch_v = 2*p.HK*S + h*S + j*cols_per_task;

    __shared__ block_q8_1 s_x8[MK_GDN_MAX_K/QK8_1];
    __shared__ float      s_alpha[MK_GDN_MAX_H];
    __shared__ float      s_beta[MK_GDN_MAX_H];
    __shared__ float4     s_c[2*S/4 + cols_per_task/4]; // conv output: q, k, this task's v
    __shared__ float4     s_qk[2*S/4];                  // normalized q, k

    const int64_t row_c = p.s_copy[0];

    // q8_1 of x, as quantize_q8_1 writes it for the alpha/beta matvecs
    for (int ib = warp; ib < p.K/QK8_1; ib += MK_NWARPS) {
        float d;
        float sum;
        const int8_t q = quantize_q8_1_lane(p.x[QK8_1*ib + lane], d, sum);
        s_x8[ib].qs[lane] = q;
        if (lane == 0) {
            s_x8[ib].ds = make_half2(d, sum);
        }
    }

    // conv + silu (ssm_conv_f32<true>): q and k channels of head kh (read by all 12 tasks of kh), this task's v channels
    float * c = (float *) s_c;
    for (int e = tid; e < 2*S + cols_per_task; e += MK_WARP_SIZE*MK_NWARPS) {
        const int ch = e < S ? ch_q + e : e < 2*S ? ch_k + e - S : ch_v + e - 2*S;
        const float * st = p.conv_src + row_c*p.conv_src_row + 3*ch;
        float xc[4];
        xc[0] = st[0];
        xc[1] = st[1];
        xc[2] = st[2];
        xc[3] = p.qkv[ch];
        float w[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            w[k] = p.conv_w[4*ch + k];
        }
        float sumf = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            sumf += xc[k] * w[k];
        }
        sumf += 0.0f; // no conv bias
        c[e] = ggml_cuda_op_silu_single(sumf);
        if (e >= 2*S) {
            // v channels have one owner: shift its conv state now
            float * dt = p.conv_dst + 3*ch;
            dt[0] = xc[1];
            dt[1] = xc[2];
            dt[2] = xc[3];
        }
    }
    __syncthreads();

    // alpha and beta of head h (mul_mat_vec_q, alpha with the fused bias), then softplus*a and sigmoid
    {
        ggml_cuda_mm_fusion_args_device fa{};
        fa.x_bias = p.alpha_bias;
        const ggml_cuda_mm_fusion_args_device fb{};
        const uint3 one = make_uint3(1, 0, 0);
        mul_mat_vec_q_impl<GGML_TYPE_Q8_0, 1, true, false, false>(
            p.w_alpha, s_x8, nullptr, fa, s_alpha, p.K, make_uint3(0, 0, 0), p.w_stride_row,
            GGML_PAD(p.K, MATRIX_ROW_PADDING)/QK8_1, p.H, one, 0, 0, 0, one, 0, 0, 0, 0, make_uint3(h, 0, 0), 1);
        __syncthreads();
        mul_mat_vec_q_impl<GGML_TYPE_Q8_0, 1, false, false, false>(
            p.w_beta, s_x8, nullptr, fb, s_beta, p.K, make_uint3(0, 0, 0), p.w_stride_row,
            GGML_PAD(p.K, MATRIX_ROW_PADDING)/QK8_1, p.H, one, 0, 0, 0, one, 0, 0, 0, 0, make_uint3(h, 0, 0), 1);
        __syncthreads();
    }

    // rms_norm_f32<32, 1, vec, do_scale>: warp 0 q, warp 1 k, a float4 per lane
    if (warp < 2) {
        const float4 xs   = s_c[warp*S/4 + lane];
        const float  eps  = warp == 0 ? p.eps_q : p.eps_k;
        const float  sout = warp == 0 ? p.scale_q : p.scale_k;
        float tmp = 0.0f;
        tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
        tmp = warp_reduce_sum<MK_WARP_SIZE>(tmp);
        const float scale = rsqrtf(tmp / S + eps);
        float4 r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
        r.x *= sout;
        r.y *= sout;
        r.z *= sout;
        r.w *= sout;
        s_qk[warp*S/4 + lane] = r;
    }

    // the 12 tasks of q/k head kh have read its old conv state: the last one shifts it
    if (mk_last_of(&p.cnt_k[kh], (p.H/p.HK)*MK_GDN_QUARTERS)) {
        for (int e = tid; e < 2*S; e += MK_WARP_SIZE*MK_NWARPS) {
            const int ch = e < S ? ch_q + e : ch_k + e - S;
            const float * st = p.conv_src + row_c*p.conv_src_row + 3*ch;
            const float s1 = st[1];
            const float s2 = st[2];
            float * dt = p.conv_dst + 3*ch;
            dt[0] = s1;
            dt[1] = s2;
            dt[2] = p.qkv[ch];
        }
    }

    // gated_delta_net_cuda<128, !KDA, !keep_rs_t>: 8 lanes per column, 16 state rows per lane
    {
        const int col_in_warp = lane / lanes_per_col;
        const int lane_in_col = lane - col_in_warp*lanes_per_col;
        const int col_local   = cols_per_warp*warp + col_in_warp;
        const int col         = j*cols_per_task + col_local;

        const float * src = p.state_src + row_c*p.state_src_row + (int64_t) h*S*S + col*S;
        float s_shard[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = src[r*lanes_per_col + lane_in_col];
        }
        const float * qn = (const float *) s_qk;
        const float * kn = qn + S;
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r*lanes_per_col + lane_in_col;
            k_reg[r] = kn[i];
            q_reg[r] = qn[i];
        }

        const float g_val    = expf(mk_softplus(s_alpha[h]) * p.ssm_a[h]);
        const float beta_val = mk_sigmoid(s_beta[h]);

        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
        const float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

        const float delta_col = (c[2*S + col_local] - g_val * kv_col) * beta_val;

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        const float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);
        if (lane_in_col == 0) {
            p.attn[h*S + col] = attn_col * p.gdn_scale;
        }

        float * dst = p.state_dst + (int64_t) h*S*S + col*S;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            dst[r*lanes_per_col + lane_in_col] = s_shard[r];
        }
    }

    __shared__ float4 s_o[MK_GDN_S/4];
    // the last of the 4 tasks of head h: rms_norm_f32<32, 1, vec, do_multiply>, then sigmoid(z)*x
    if (mk_last_of(&p.cnt_v[h], MK_GDN_QUARTERS) && warp == 0) {
        const float4 xs = ((const float4 *) (p.attn + h*S))[lane];
        float tmp = 0.0f;
        tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
        tmp = warp_reduce_sum<MK_WARP_SIZE>(tmp);
        const float scale = rsqrtf(tmp / S + p.eps_o);
        float4 r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
        const float4 m = ((const float4 *) p.w_onorm)[lane];
        r.x *= m.x;
        r.y *= m.y;
        r.z *= m.z;
        r.w *= m.w;
        const float4 zz = ((const float4 *) (p.z + h*S))[lane];
        float4 o;
        o.x = mk_sigmoid(zz.x) * r.x;
        o.y = mk_sigmoid(zz.y) * r.y;
        o.z = mk_sigmoid(zz.z) * r.z;
        o.w = mk_sigmoid(zz.w) * r.w;
        ((float4 *) (p.out_s + h*S))[lane] = o;
        if (p.q8_out) { // q8_1 copy of this head's 128 outputs for the ssm_out MMVQ, as quantize_q8_1 writes it
            s_o[lane] = o;
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            const float * so = (const float *) s_o;
#pragma unroll
            for (int ib = 0; ib < S/QK8_1; ++ib) {
                float d;
                float sum;
                const int8_t q = quantize_q8_1_lane(so[QK8_1*ib + lane], d, sum);
                block_q8_1 & y = p.q8_out[h*(S/QK8_1) + ib];
                y.qs[lane] = q;
                if (lane == 0) {
                    y.ds = make_half2(d, sum);
                }
            }
        }
    }

    // out may share memory with an input (gallocr reuses qkv for it): the last task, after every read, writes it
    if (mk_last_of(p.cnt_all, gridDim.x)) {
        for (int e = tid; e < p.H*S/4; e += MK_WARP_SIZE*MK_NWARPS) {
            ((float4 *) p.out)[e] = ((const float4 *) p.out_s)[e];
        }
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif // MK_DEVICE_CODE
}

static bool mk_is_q8_0_rows(const ggml_tensor * w, int64_t K, int64_t rows) {
    return w->type == GGML_TYPE_Q8_0 && w->ne[0] == K && w->ne[1] == rows && ggml_nrows(w) == rows &&
        w->nb[0] == ggml_type_size(GGML_TYPE_Q8_0) && w->nb[1] == ggml_row_size(GGML_TYPE_Q8_0, K);
}

bool ggml_cuda_mk_gdn_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MEGAKERNEL_GDN");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

bool ggml_cuda_mk_gdn_supported(const ggml_cuda_mk_gdn & m) {
    const int64_t S    = MK_GDN_S;
    const int64_t H    = m.H;
    const int64_t HK   = m.HK;
    const int64_t n_ch = 2*HK*S + H*S;
    const int64_t K    = m.x->ne[0];
    if (H <= 0 || H > MK_GDN_MAX_H || HK <= 0 || H % HK != 0 || K % QK8_1 != 0 || K > MK_GDN_MAX_K) {
        return false;
    }
    const auto f32c = [](const ggml_tensor * t, int64_t n) {
        return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == n;
    };
    if (m.s_copy->type != GGML_TYPE_I32 || ggml_nelements(m.s_copy) != 1) {
        return false;
    }
    // cache rows: contiguous within a row, rows of the expected length
    if (m.conv_src->type != GGML_TYPE_F32 || m.conv_src->ne[0] != 3*n_ch || m.conv_src->nb[0] != sizeof(float) ||
            m.conv_src->nb[1] % sizeof(float) != 0) {
        return false;
    }
    if (m.state_src->type != GGML_TYPE_F32 || m.state_src->ne[0] != S*S*H || m.state_src->nb[0] != sizeof(float) ||
            m.state_src->nb[1] % sizeof(float) != 0) {
        return false;
    }
    if (!f32c(m.conv_dst, 3*n_ch) || !f32c(m.state_dst, S*S*H) || !f32c(m.qkv, n_ch) || !f32c(m.conv_w, 4*n_ch) ||
            m.conv_w->ne[0] != 4 || !f32c(m.x, K) || !f32c(m.alpha_bias, H) || !f32c(m.ssm_a, H) || !f32c(m.z, S*H) ||
            !f32c(m.w_onorm, S) || !f32c(m.out, S*H)) {
        return false;
    }
    if (!mk_is_q8_0_rows(m.w_alpha, K, H) || !mk_is_q8_0_rows(m.w_beta, K, H)) {
        return false;
    }
    if (((uintptr_t) m.w_onorm->data | (uintptr_t) m.z->data | (uintptr_t) m.out->data) % 16 != 0) {
        return false;
    }
    return true;
}

void ggml_cuda_mk_gdn_run(ggml_backend_cuda_context & ctx, const ggml_cuda_mk_gdn & m) {
    mk_gdn_params p;
    char * scratch = ctx.mk_buf + MK_GDN_BASE;
    p.s_copy        = (const int32_t *) m.s_copy->data;
    p.conv_src      = (const float *) m.conv_src->data;
    p.conv_src_row  = m.conv_src->nb[1] / sizeof(float);
    p.conv_dst      = (float *) m.conv_dst->data;
    p.qkv           = (const float *) m.qkv->data;
    p.conv_w        = (const float *) m.conv_w->data;
    p.state_src     = (const float *) m.state_src->data;
    p.state_src_row = m.state_src->nb[1] / sizeof(float);
    p.state_dst     = (float *) m.state_dst->data;
    p.eps_q         = m.eps_q;
    p.scale_q       = m.scale_q;
    p.eps_k         = m.eps_k;
    p.scale_k       = m.scale_k;
    p.x             = (const float *) m.x->data;
    p.K             = (int) m.x->ne[0];
    p.w_alpha       = m.w_alpha->data;
    p.alpha_bias    = (const float *) m.alpha_bias->data;
    p.ssm_a         = (const float *) m.ssm_a->data;
    p.w_beta        = m.w_beta->data;
    p.w_stride_row  = m.w_alpha->nb[1] / ggml_type_size(GGML_TYPE_Q8_0);
    p.gdn_scale     = 1.0f / sqrtf((float) MK_GDN_S);
    p.z             = (const float *) m.z->data;
    p.w_onorm       = (const float *) m.w_onorm->data;
    p.eps_o         = m.eps_o;
    p.out           = (float *) m.out->data;
    p.attn          = (float *) (scratch + MK_GDN_OFF_ATTN);
    p.out_s         = (float *) (scratch + MK_GDN_OFF_OUT);
    p.q8_out        = nullptr;
    if ((m.H*MK_GDN_S) % MATRIX_ROW_PADDING == 0 && ggml_is_contiguous(m.out)) {
        p.q8_out = (block_q8_1 *) ctx.q8_1_cache_reserve_row(m.out->data, m.H*MK_GDN_S,
                                                             m.H*MK_GDN_S/QK8_1*sizeof(block_q8_1));
    }
    p.cnt_v         = (uint32_t *) (scratch + MK_GDN_OFF_CNT);
    p.cnt_k         = p.cnt_v + MK_GDN_MAX_H;
    p.cnt_all       = p.cnt_k + MK_GDN_MAX_H;
    p.diag          = mk_diag_enabled() ? (mk_diag *) (ctx.mk_buf + MK_BUF_SIZE + MK_PROF_SIZE) : nullptr;
    p.H             = m.H;
    p.HK            = m.HK;

    const dim3 block_dims(MK_WARP_SIZE, MK_NWARPS, 1);
    mk_gdn_kernel<<<m.H*MK_GDN_QUARTERS, block_dims, 0, ctx.stream()>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------------
// FFN tail: routed weighting, shared expert gate, residual combine, hc_post

static constexpr int MK_TAIL_THREADS = 256; // mul_mat_vec_f block size for this row length, and columns per block
static constexpr int MK_TAIL_MAX_USED = 16;
static constexpr int MK_TAIL_MAX_HC   = 8;

struct mk_ffn_tail_params {
    const float * experts;
    const float * weights;
    const float * shexp;
    const float * w_sg;
    const float * x;
    const float * residual;
    const float * inject;
    float         s1, b1, s2, b2;
    float *       dst;
    int64_t       n_embd;
    int           n_used;
    int           hc;
};

__launch_bounds__(MK_TAIL_THREADS, 1)
static __global__ void mk_ffn_tail_kernel(const mk_ffn_tail_params p) {
#ifdef MK_DEVICE_CODE
    constexpr int warp_size = MK_WARP_SIZE;
    const int tid = threadIdx.x;

    // shared expert gate: mul_mat_vec_f<float, float, 1, 256>, recomputed by every block
    __shared__ float buf_iw[warp_size];
    __shared__ float s_gate;
    if (tid < warp_size) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();
    {
        const float2 * x2 = (const float2 *) p.w_sg;
        const float2 * y2 = (const float2 *) p.x;
        float sumf = 0.0f;
        for (int col2 = tid; col2 < p.n_embd/2; col2 += MK_TAIL_THREADS) {
            const float2 tmpx = x2[col2];
            const float2 tmpy = y2[col2];
            ggml_cuda_mad(sumf, tmpx.x, tmpy.x);
            ggml_cuda_mad(sumf, tmpx.y, tmpy.y);
        }
        sumf = warp_reduce_sum<warp_size>(sumf);
        buf_iw[tid/warp_size] = sumf;
        __syncthreads();
        if (tid < warp_size) {
            sumf = buf_iw[tid];
            sumf = warp_reduce_sum<warp_size>(sumf);
            if (tid == 0) {
                s_gate = sumf;
            }
        }
        __syncthreads();
    }

    const int64_t col = (int64_t) blockIdx.x*MK_TAIL_THREADS + tid;
    if (col >= p.n_embd) {
        return;
    }

    // moe_weighted_reduction_f32 (no expert scale)
    const float first_scale = 1.0f;
    float moe = (p.experts[col] * first_scale) * p.weights[0];
    for (int e = 1; e < p.n_used; ++e) {
        const float scale = 1.0f;
        moe += (p.experts[e*p.n_embd + col] * scale) * p.weights[e];
    }

    // sigmoid_mul_add_kernel
    float out;
    {
#pragma clang fp contract(off)
        const float m = p.shexp[col]*mk_sigmoid(s_gate);
        out = moe + m;
    }

    // dsv4_hc_post_f32<false, true>
    for (int h = 0; h < p.hc; ++h) {
        float pp = p.inject[h];
        pp = p.s2 / (1.0f + expf(-(p.s1*pp + p.b1))) + p.b2;
        float sum = out * pp;
        sum += p.residual[h*p.n_embd + col];
        p.dst[h*p.n_embd + col] = sum;
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif // MK_DEVICE_CODE
}

bool ggml_cuda_mk_ffn_tail_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MEGAKERNEL_FFN");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

bool ggml_cuda_mk_ffn_tail_supported(const ggml_cuda_mk_ffn_tail & m) {
    const int64_t n_embd = m.shexp->ne[0];
    const int64_t n_used = m.experts->ne[1];
    const int64_t hc     = m.residual->ne[1];
    const auto f32c = [](const ggml_tensor * t, int64_t n) {
        return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == n;
    };
    if (n_used < 2 || n_used > MK_TAIL_MAX_USED || hc < 1 || hc > MK_TAIL_MAX_HC || n_embd % 2 != 0 ||
            n_embd > 64*MK_TAIL_THREADS) {
        return false;
    }
    if (!f32c(m.experts, n_embd*n_used) || m.experts->ne[0] != n_embd || !f32c(m.weights, n_used) ||
            !f32c(m.shexp, n_embd) || !f32c(m.w_sg, n_embd) || !f32c(m.x, n_embd) || !f32c(m.residual, n_embd*hc) ||
            m.residual->ne[0] != n_embd || !f32c(m.inject, hc) || !f32c(m.dst, n_embd*hc)) {
        return false;
    }
    // the gate row is bit-identical to mul_mat_vec_f only where that picks 256 threads
    int64_t block_size_best = MK_WARP_SIZE;
    int64_t niter_best      = (n_embd + 2*MK_WARP_SIZE - 1) / (2*MK_WARP_SIZE);
    for (int64_t block_size = 2*MK_WARP_SIZE; block_size <= MK_TAIL_THREADS; block_size += MK_WARP_SIZE) {
        const int64_t niter = (n_embd + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }
    return block_size_best == MK_TAIL_THREADS && ((uintptr_t) m.w_sg->data | (uintptr_t) m.x->data) % 8 == 0;
}

void ggml_cuda_mk_ffn_tail_run(ggml_backend_cuda_context & ctx, const ggml_cuda_mk_ffn_tail & m) {
    mk_ffn_tail_params p;
    p.experts  = (const float *) m.experts->data;
    p.weights  = (const float *) m.weights->data;
    p.shexp    = (const float *) m.shexp->data;
    p.w_sg     = (const float *) m.w_sg->data;
    p.x        = (const float *) m.x->data;
    p.residual = (const float *) m.residual->data;
    p.inject   = (const float *) m.inject->data;
    p.s1       = m.s1;
    p.b1       = m.b1;
    p.s2       = m.s2;
    p.b2       = m.b2;
    p.dst      = (float *) m.dst->data;
    p.n_embd   = m.shexp->ne[0];
    p.n_used   = (int) m.experts->ne[1];
    p.hc       = (int) m.residual->ne[1];
    const int nblocks = (int) ((p.n_embd + MK_TAIL_THREADS - 1) / MK_TAIL_THREADS);
    mk_ffn_tail_kernel<<<nblocks, MK_TAIL_THREADS, 0, ctx.stream()>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

void * ggml_cuda_mk_router_scratch(ggml_backend_cuda_context & ctx) {
    return ctx.mk_buf + MK_ROUTER_BASE;
}

bool ggml_cuda_mk_router_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MEGAKERNEL_ROUTER");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}


// ---------------------------------------------------------------------------------------------------------------------
// in-order small-op executor

enum mk_seq_op : uint8_t {
    MK_SEQ_FILL, MK_SEQ_CPY, MK_SEQ_REPEAT, MK_SEQ_GET_ROWS, MK_SEQ_SET_ROWS, MK_SEQ_SCALE, MK_SEQ_CLAMP,
    MK_SEQ_CONCAT, MK_SEQ_CUMSUM, MK_SEQ_ADD, MK_SEQ_SUB, MK_SEQ_MUL,
};

static constexpr int MK_SEQ_THREADS = 1024;
// one block: larger nodes end the run and launch on their own (the QSA mask's n_kv-sized f16 ops at depth)
static constexpr int64_t MK_SEQ_MAX_ELEMS = 16384;

static __device__ __forceinline__ float mk_seq_load(const char * p, uint8_t t, int64_t i) {
    switch (t) {
        case GGML_TYPE_F32: return ((const float *) p)[i];
        case GGML_TYPE_F16: return ggml_cuda_cast<float>(((const half *) p)[i]);
        default:            return ggml_cuda_cast<float>(((const int32_t *) p)[i]);
    }
}

static __device__ __forceinline__ void mk_seq_store(char * p, uint8_t t, int64_t i, float v) {
    switch (t) {
        case GGML_TYPE_F32: ((float *)   p)[i] = v; break;
        case GGML_TYPE_F16: ((half *)    p)[i] = ggml_cuda_cast<half>(v); break;
        default:            ((int32_t *) p)[i] = ggml_cuda_cast<int32_t>(v); break;
    }
}

// a same-type element copy (get_rows, set_rows, concat between equal types)
static __device__ __forceinline__ void mk_seq_copy(char * d, int64_t id, const char * a, int64_t ia, uint8_t t) {
    if (t == GGML_TYPE_F16) {
        ((half *) d)[id] = ((const half *) a)[ia];
    } else {
        ((int32_t *) d)[id] = ((const int32_t *) a)[ia];
    }
}

static __device__ __forceinline__ int64_t mk_seq_index(const char * b, uint8_t t, int64_t i) {
    return t == GGML_TYPE_I64 ? ((const int64_t *) b)[i] : ((const int32_t *) b)[i];
}

static __device__ __forceinline__ float mk_seq_binop(uint8_t op, float x, float y) {
    switch (op) {
        case MK_SEQ_ADD: return x + y;
        case MK_SEQ_SUB: return x - y;
        default:         return x * y;
    }
}

// cumsum_kernel<float> for one row, as launched with block_size threads (min(ne0 rounded to a warp, 256)); every
// thread of the executor block reaches the barriers
static __device__ void mk_seq_cumsum_row(const float * src_row, float * dst_row, const int64_t ne00, const int block_size,
                                         float * s_vals, float * s_warp_sums, float * s_carry, float * s_chunk_total) {
    constexpr int warp_size = MK_WARP_SIZE;
    const int  tid    = threadIdx.x;
    const int  lane   = tid % warp_size;
    const int  warp   = tid / warp_size;
    const int  warps_per_block = block_size / warp_size;
    const bool active = tid < block_size;

    if (tid == 0) {
        *s_carry = 0.0f;
    }
    __syncthreads();

    constexpr int num_unroll = 4;
    float         temp[num_unroll];

    for (int64_t i = 0; i < ne00; i += num_unroll * block_size) {
        const int64_t idx = i + tid * num_unroll;
        float val = 0.0f;
        if (active) {
            temp[0] = (idx < ne00 ? src_row[idx] : 0.0f);
#pragma unroll
            for (int64_t j = 1; j < num_unroll; j++) {
                temp[j] = temp[j - 1];
                if (idx + j < ne00) {
                    temp[j] += src_row[idx + j];
                } else {
                    temp[j] += 0;
                }
            }
            val = (idx < ne00) ? temp[num_unroll - 1] : 0.0f;
            val = warp_prefix_inclusive_sum<float, warp_size>(val);
            s_vals[tid] = val;
            if (lane == warp_size - 1) {
                s_warp_sums[warp] = val;
            }
        }
        __syncthreads();

        if (warp == 0) {
            float w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            float inc = warp_prefix_inclusive_sum<float, warp_size>(w);
            if (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;
            }
            if (tid == warps_per_block - 1) {
                *s_chunk_total = inc;
            }
        }
        __syncthreads();

        if (active) {
            float carry = *s_carry;
            float final_val_offset = s_vals[tid] + s_warp_sums[warp] + carry - temp[num_unroll - 1];
#pragma unroll
            for (int32_t j = 0; j < num_unroll; j++) {
                if (idx + j < ne00) {
                    dst_row[idx + j] = temp[j] + final_val_offset;
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            *s_carry += *s_chunk_total;
        }
        __syncthreads();
    }
}

__launch_bounds__(MK_SEQ_THREADS, 1)
static __global__ void mk_seq_kernel(const mk_seq_params p) {
#ifdef MK_DEVICE_CODE
    __shared__ float s_vals[256];
    __shared__ float s_warp_sums[256/MK_WARP_SIZE];
    __shared__ float s_carry;
    __shared__ float s_chunk_total;
    const int tid = threadIdx.x;

    for (int k = 0; k < p.n; ++k) {
        const mk_seq_node & nd = p.nodes[k];
        const int64_t n = (int64_t) nd.ne0*nd.ne1;
        if (nd.sync) {
            __syncthreads();
        }
        switch (nd.op) {
            case MK_SEQ_FILL: {
                for (int64_t e = tid; e < n; e += MK_SEQ_THREADS) {
                    if (nd.td == GGML_TYPE_F16) {
                        ((half *) nd.d)[(e / nd.ne0)*nd.d_s1 + e % nd.ne0] = ggml_cuda_cast<half>(nd.f0);
                    } else {
                        ((float *) nd.d)[(e / nd.ne0)*nd.d_s1 + e % nd.ne0] = nd.f0;
                    }
                }
            } break;
            case MK_SEQ_CPY: {
                for (int64_t e = tid; e < n; e += MK_SEQ_THREADS) {
                    const int64_t i0 = e % nd.ne0;
                    const int64_t i1 = e / nd.ne0;
                    const int64_t ia = (e / nd.a_ne0)*nd.a_s1 + e % nd.a_ne0;
                    if (nd.ta == nd.td) {
                        mk_seq_copy(nd.d, i1*nd.d_s1 + i0, nd.a, ia, nd.td);
                    } else if (nd.ta == GGML_TYPE_F32 && nd.td == GGML_TYPE_I32) {
                        ((int32_t *) nd.d)[i1*nd.d_s1 + i0] = ggml_cuda_cast<int32_t>(((const float *) nd.a)[ia]);
                    } else {
                        mk_seq_store(nd.d, nd.td, i1*nd.d_s1 + i0, mk_seq_load(nd.a, nd.ta, ia));
                    }
                }
            } break;
            case MK_SEQ_REPEAT: { // bin_bcast<op_repeat>: through float
                for (int64_t e = tid; e < n; e += MK_SEQ_THREADS) {
                    const int64_t i0 = e % nd.ne0;
                    const int64_t i1 = e / nd.ne0;
                    const float v = mk_seq_load(nd.a, nd.ta, (i1 % nd.a_ne1)*nd.a_s1 + i0 % nd.a_ne0);
                    mk_seq_store(nd.d, nd.td, i1*nd.d_s1 + i0, v);
                }
            } break;
            case MK_SEQ_GET_ROWS: {
                for (int64_t e = tid; e < n; e += MK_SEQ_THREADS) {
                    const int64_t i0 = e % nd.ne0;
                    const int64_t i1 = e / nd.ne0;
                    const int64_t r  = mk_seq_index(nd.b, nd.tb, i1);
                    mk_seq_copy(nd.d, i1*nd.d_s1 + i0, nd.a, r*nd.a_s1 + i0, nd.td);
                }
            } break;
            case MK_SEQ_SET_ROWS: {
                const int64_t ns = (int64_t) nd.a_ne0*nd.a_ne1;
                for (int64_t e = tid; e < ns; e += MK_SEQ_THREADS) {
                    const int64_t i0 = e % nd.a_ne0;
                    const int64_t i1 = e / nd.a_ne0;
                    const int64_t r  = mk_seq_index(nd.b, nd.tb, i1);
                    if (nd.ta == nd.td) {
                        mk_seq_copy(nd.d, r*nd.d_s1 + i0, nd.a, i1*nd.a_s1 + i0, nd.td);
                    } else {
                        mk_seq_store(nd.d, nd.td, r*nd.d_s1 + i0, mk_seq_load(nd.a, nd.ta, i1*nd.a_s1 + i0));
                    }
                }
            } break;
            case MK_SEQ_SCALE: {
                const float scale = nd.f0;
                const float bias  = nd.f1;
                for (int64_t e = tid; e < n; e += MK_SEQ_THREADS) {
                    const int64_t i0 = e % nd.ne0;
                    const int64_t i1 = e / nd.ne0;
                    const float * x = (const float *) nd.a;
                    ((float *) nd.d)[i1*nd.d_s1 + i0] = ggml_cuda_cast<float>(scale * ggml_cuda_cast<float>(x[i1*nd.a_s1 + i0]) + bias);
                }
            } break;
            case MK_SEQ_CLAMP: {
                for (int64_t e = tid; e < n; e += MK_SEQ_THREADS) {
                    const int64_t i0 = e % nd.ne0;
                    const int64_t i1 = e / nd.ne0;
                    const float x = ((const float *) nd.a)[i1*nd.a_s1 + i0];
                    ((float *) nd.d)[i1*nd.d_s1 + i0] = fminf(fmaxf(x, nd.f0), nd.f1);
                }
            } break;
            case MK_SEQ_CONCAT: {
                for (int64_t e = tid; e < n; e += MK_SEQ_THREADS) {
                    const int64_t i0 = e % nd.ne0;
                    const int64_t i1 = e / nd.ne0;
                    const bool    in_a = nd.dim == 0 ? i0 < nd.a_ne0 : i1 < nd.a_ne1;
                    const int64_t is = in_a ? i1*nd.a_s1 + i0 :
                        nd.dim == 0 ? i1*nd.b_s1 + (i0 - nd.a_ne0) : (i1 - nd.a_ne1)*nd.b_s1 + i0;
                    mk_seq_copy(nd.d, i1*nd.d_s1 + i0, in_a ? nd.a : nd.b, is, nd.td);
                }
            } break;
            case MK_SEQ_CUMSUM: {
                const int block_size = min((nd.ne0 + MK_WARP_SIZE - 1)/MK_WARP_SIZE*MK_WARP_SIZE, 256);
                for (int64_t i1 = 0; i1 < nd.ne1; ++i1) {
                    mk_seq_cumsum_row((const float *) nd.a + i1*nd.a_s1, (float *) nd.d + i1*nd.d_s1, nd.ne0, block_size,
                                      s_vals, s_warp_sums, &s_carry, &s_chunk_total);
                }
            } break;
            default: { // ADD, SUB, MUL: bin_bcast, src1 broadcast
                for (int64_t e = tid; e < n; e += MK_SEQ_THREADS) {
                    const int64_t i0 = e % nd.ne0;
                    const int64_t i1 = e / nd.ne0;
                    const float x = mk_seq_load(nd.a, nd.ta, i1*nd.a_s1 + i0);
                    const float y = mk_seq_load(nd.b, nd.tb, (i1 % nd.b_ne1)*nd.b_s1 + i0 % nd.b_ne0);
                    mk_seq_store(nd.d, nd.td, i1*nd.d_s1 + i0, mk_seq_binop(nd.op, x, y));
                }
            } break;
        }
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif // MK_DEVICE_CODE
}

bool ggml_cuda_mk_seq_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MEGAKERNEL_SEQ");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

int ggml_cuda_mk_seq_min_nodes() {
    static const int n = [] {
        const char * env = getenv("GGML_CUDA_MEGAKERNEL_SEQ_MIN");
        return env ? std::max(1, atoi(env)) : 8;
    }();
    return n;
}

// at most 2-D, contiguous rows, row strides that fit in 32 bits
static bool mk_seq_tensor_ok(const ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_I32) {
        return false;
    }
    const size_t ts = ggml_type_size(t->type);
    return t->ne[2] == 1 && t->ne[3] == 1 && t->nb[0] == ts && t->nb[1] % ts == 0 && t->nb[1]/ts < INT32_MAX &&
        t->ne[0] < INT32_MAX && t->ne[1] < INT32_MAX && t->ne[0]*t->ne[1] <= MK_SEQ_MAX_ELEMS && t->data != nullptr;
}

bool ggml_cuda_mk_seq_node_supported(const ggml_tensor * t, mk_seq_node & nd) {
    nd = {};
    if (!mk_seq_tensor_ok(t)) {
        return false;
    }
    const ggml_tensor * a = t->src[0];
    const ggml_tensor * b = t->src[1];
    nd.d    = (char *) t->data;
    nd.ne0  = (int32_t) t->ne[0];
    nd.ne1  = (int32_t) t->ne[1];
    nd.d_s1 = (int32_t) (t->nb[1] / ggml_type_size(t->type));
    nd.td   = t->type;
    auto set_a = [&](const ggml_tensor * x) {
        nd.a = (const char *) x->data; nd.a_ne0 = (int32_t) x->ne[0]; nd.a_ne1 = (int32_t) x->ne[1];
        nd.a_s1 = (int32_t) (x->nb[1] / ggml_type_size(x->type)); nd.ta = x->type;
    };
    auto set_b = [&](const ggml_tensor * x) {
        nd.b = (const char *) x->data; nd.b_ne0 = (int32_t) x->ne[0]; nd.b_ne1 = (int32_t) x->ne[1];
        nd.b_s1 = (int32_t) (x->nb[1] / ggml_type_size(x->type)); nd.tb = x->type;
    };
    const bool f32 = t->type == GGML_TYPE_F32;
    switch (t->op) {
        case GGML_OP_FILL:
            if (t->type == GGML_TYPE_I32 || !ggml_is_contiguous(t)) {
                return false;
            }
            nd.op = MK_SEQ_FILL;
            memcpy(&nd.f0, t->op_params, sizeof(float));
            return true;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            // cpy_scalar_contiguous; the conversions it has
            if (!mk_seq_tensor_ok(a) || !ggml_is_contiguous(a) || !ggml_is_contiguous(t) ||
                    ggml_nelements(a) != ggml_nelements(t) ||
                    !(a->type == t->type || (a->type == GGML_TYPE_F32 && t->type != GGML_TYPE_F32) ||
                      (t->type == GGML_TYPE_F32 && a->type != GGML_TYPE_F32))) {
                return false;
            }
            if (t->op == GGML_OP_CPY && t->src[1] != nullptr && t->src[1]->data != t->data) {
                return false;
            }
            nd.op = MK_SEQ_CPY;
            set_a(a);
            return true;
        case GGML_OP_REPEAT:
            if (!mk_seq_tensor_ok(a) || t->type == GGML_TYPE_I32 || a->type != t->type ||
                    t->ne[0] % a->ne[0] != 0 || t->ne[1] % a->ne[1] != 0) {
                return false;
            }
            nd.op = MK_SEQ_REPEAT;
            set_a(a);
            return true;
        case GGML_OP_GET_ROWS:
            if (!mk_seq_tensor_ok(a) || a->type != t->type || t->type == GGML_TYPE_F16 || b->type != GGML_TYPE_I32 ||
                    b->ne[1] != 1 || b->ne[2] != 1 || b->ne[3] != 1 || b->ne[0] != t->ne[1] || a->ne[0] != t->ne[0] ||
                    !ggml_is_contiguous(b)) {
                return false;
            }
            nd.op = MK_SEQ_GET_ROWS;
            set_a(a);
            set_b(b);
            return true;
        case GGML_OP_SET_ROWS:
            if (!mk_seq_tensor_ok(a) || t->type == GGML_TYPE_I32 || (a->type != t->type && a->type != GGML_TYPE_F32) ||
                    (b->type != GGML_TYPE_I32 && b->type != GGML_TYPE_I64) || b->ne[1] != 1 || b->ne[2] != 1 ||
                    b->ne[3] != 1 || b->ne[0] != a->ne[1] || a->ne[0] != t->ne[0] || !ggml_is_contiguous(b)) {
                return false;
            }
            nd.op = MK_SEQ_SET_ROWS;
            set_a(a);
            nd.b = (const char *) b->data; nd.b_ne0 = (int32_t) b->ne[0]; nd.b_ne1 = 1; nd.tb = b->type;
            return true;
        case GGML_OP_SCALE:
            if (!f32 || !mk_seq_tensor_ok(a) || a->type != GGML_TYPE_F32 || !ggml_are_same_shape(a, t)) {
                return false;
            }
            nd.op = MK_SEQ_SCALE;
            nd.f0 = ggml_get_op_params_f32(t, 0);
            nd.f1 = ggml_get_op_params_f32(t, 1);
            set_a(a);
            return true;
        case GGML_OP_CLAMP:
            if (!f32 || !mk_seq_tensor_ok(a) || a->type != GGML_TYPE_F32 || !ggml_are_same_shape(a, t)) {
                return false;
            }
            nd.op = MK_SEQ_CLAMP;
            nd.f0 = ggml_get_op_params_f32(t, 0);
            nd.f1 = ggml_get_op_params_f32(t, 1);
            set_a(a);
            return true;
        case GGML_OP_CONCAT: {
            const int dim = ggml_get_op_params_i32(t, 0);
            if (t->type == GGML_TYPE_F16 || !mk_seq_tensor_ok(a) || !mk_seq_tensor_ok(b) || a->type != t->type ||
                    b->type != t->type || (dim != 0 && dim != 1) || !ggml_is_contiguous(a) || !ggml_is_contiguous(b) ||
                    !ggml_is_contiguous(t)) {
                return false;
            }
            nd.op  = MK_SEQ_CONCAT;
            nd.dim = dim;
            set_a(a);
            set_b(b);
            return true;
        }
        case GGML_OP_CUMSUM:
            if (!f32 || !mk_seq_tensor_ok(a) || a->type != GGML_TYPE_F32 || !ggml_are_same_shape(a, t)) {
                return false;
            }
            nd.op = MK_SEQ_CUMSUM;
            set_a(a);
            return true;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
            if (t->type == GGML_TYPE_I32 || !mk_seq_tensor_ok(a) || !mk_seq_tensor_ok(b) || a->type == GGML_TYPE_I32 ||
                    b->type == GGML_TYPE_I32 || !ggml_are_same_shape(a, t) || t->ne[0] % b->ne[0] != 0 ||
                    t->ne[1] % b->ne[1] != 0) {
                return false;
            }
            nd.op = t->op == GGML_OP_ADD ? MK_SEQ_ADD : t->op == GGML_OP_SUB ? MK_SEQ_SUB : MK_SEQ_MUL;
            set_a(a);
            set_b(b);
            return true;
        default:
            return false;
    }
}

void ggml_cuda_mk_seq_run(ggml_backend_cuda_context & ctx, const mk_seq_params & p) {
    mk_seq_kernel<<<1, MK_SEQ_THREADS, 0, ctx.stream()>>>(p);
    CUDA_CHECK(cudaGetLastError());
}
