#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <cmath>
#include <initializer_list>

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool delayed_softmax;
    int  n_groups;
    int  n_group_used;
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : logf(1.0f + expf(vals[i]))) : -INFINITY;
    }
}

#if !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000
// Set -inf for experts outside the top n_group_used groups. Group score is the sum of its top-2 experts (DeepSeek V3).
template <int n_experts, int experts_per_thread>
__device__ void group_mask_warp_inplace(float (&vals)[experts_per_thread], const int n_groups, const int n_group_used, const int lane) {
    const int n_per_group = n_experts / n_groups;

    float score = -INFINITY; // lane g holds the score of group g
    for (int g = 0; g < n_groups; g++) {
        float top1 = -INFINITY;
        float top2 = -INFINITY;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            if ((i * WARP_SIZE + lane) / n_per_group == g) {
                top2 = fmaxf(top2, fminf(top1, vals[i]));
                top1 = fmaxf(top1, vals[i]);
            }
        }
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            const float o1 = __shfl_xor_sync(0xFFFFFFFF, top1, offset, WARP_SIZE);
            const float o2 = __shfl_xor_sync(0xFFFFFFFF, top2, offset, WARP_SIZE);
            top2 = fmaxf(fminf(top1, o1), fmaxf(top2, o2));
            top1 = fmaxf(top1, o1);
        }
        if (lane == g) {
            score = top1 + top2;
        }
    }

    // rank of each group, lower index wins ties
    int rank = 0;
    for (int g = 0; g < n_groups; g++) {
        const float other = __shfl_sync(0xFFFFFFFF, score, g, WARP_SIZE);
        rank += other > score || (other == score && g < lane);
    }
    const auto keep = __ballot_sync(0xFFFFFFFFULL, lane < n_groups && rank < n_group_used);

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int g = (i * WARP_SIZE + lane) / n_per_group;
        if (g >= n_groups || !((keep >> g) & 1)) {
            vals[i] = -INFINITY;
        }
    }
}
#endif // !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000

#if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4)) && WARP_SIZE == 32
#define TOPK_MOE_DPP_ARGMAX
// [ignite] the argmax rounds as a max over one 64-bit key per lane: orderable value bits above (65535 - expert), so the
// larger value wins and ties go to the lower expert like the shuffle loop below. The cross-lane steps are DPP and
// permlanex16 moves (VALU latency) instead of ds_bpermute round trips through LDS.
static __device__ __forceinline__ uint32_t topk_moe_orderable(const float v) {
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
static __device__ __forceinline__ float topk_moe_unorderable(const uint32_t k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}
template <int ctrl>
static __device__ __forceinline__ uint64_t topk_moe_max_dpp(const uint64_t x) {
    const uint32_t lo = __builtin_amdgcn_update_dpp(0u, (uint32_t) x,         ctrl, 0xf, 0xf, false);
    const uint32_t hi = __builtin_amdgcn_update_dpp(0u, (uint32_t) (x >> 32), ctrl, 0xf, 0xf, false);
    const uint64_t y = ((uint64_t) hi << 32) | lo;
    return y > x ? y : x;
}
static __device__ __forceinline__ uint64_t topk_moe_wave_max(uint64_t x) {
    x = topk_moe_max_dpp<0xB1>(x);  // quad_perm [1,0,3,2]: lane ^ 1
    x = topk_moe_max_dpp<0x4E>(x);  // quad_perm [2,3,0,1]: lane ^ 2
    x = topk_moe_max_dpp<0x164>(x); // row_xmask:4
    x = topk_moe_max_dpp<0x168>(x); // row_xmask:8
    const uint32_t lo = __builtin_amdgcn_permlanex16((uint32_t) x,         (uint32_t) x,         0x76543210u, 0xfedcba98u, false, false);
    const uint32_t hi = __builtin_amdgcn_permlanex16((uint32_t) (x >> 32), (uint32_t) (x >> 32), 0x76543210u, 0xfedcba98u, false, false);
    const uint64_t y = ((uint64_t) hi << 32) | lo;  // lane ^ 16
    return y > x ? y : x;
}
#endif

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
// one row, one warp (lane threadIdx.x). block_sync: the topk_moe_cuda kernel, whose rows share a block and whose
// weights and ids can alias the logits
template <int n_experts, bool has_bias, bool block_sync>
static __device__ __forceinline__ void topk_moe_row(const float *         logits,
                                                    float *               weights,
                                                    int32_t *             ids,
                                                    float *               bias,
                                                    const int             n_expert_used,
                                                    const float           clamp_val,
                                                    const float           scale_val,
                                                    const topk_moe_config config) {
    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

    if constexpr (block_sync) {
        ggml_cuda_pdl_sync();
    }
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    // Weights and IDs can alias logits, so wait until every row in the block reads its logits.
    if constexpr (block_sync) {
        __syncthreads();
    }

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + threadIdx.x;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
        }
    }

#if !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000
    // without bias, masking wt is safe: the matcher ensures the kept groups hold all used experts
    if (config.n_groups > 0) {
        if constexpr (has_bias) {
            group_mask_warp_inplace<n_experts>(selection_wt, config.n_groups, config.n_group_used, threadIdx.x);
        } else {
            group_mask_warp_inplace<n_experts>(wt, config.n_groups, config.n_group_used, threadIdx.x);
        }
    }
#endif // !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    if constexpr (block_sync) {
        ggml_cuda_pdl_lc();
    }
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const float val_s  = __shfl_xor_sync(0xFFFFFFFF, max_val_s, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                selection_wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        } else {
#ifdef TOPK_MOE_DPP_ARGMAX
            static_assert(n_experts <= 65536, "expert index in 16 bits");
            uint64_t key = 0;
#pragma unroll
            for (int i = 0; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if (n_experts % WARP_SIZE == 0 || expert < n_experts) {
                    const uint64_t k = ((uint64_t) topk_moe_orderable(wt[i]) << 32) | (uint32_t) (65535 - expert);
                    key = k > key ? k : key;
                }
            }
            key        = topk_moe_wave_max(key);
            max_val    = topk_moe_unorderable((uint32_t) (key >> 32));
            max_expert = 65535 - (int) (uint32_t) key;
#else
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }
#endif // TOPK_MOE_DPP_ARGMAX

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <int n_experts, bool has_bias>
__launch_bounds__(TOPK_MOE_ROWS_PER_BLOCK * WARP_SIZE, 1)
__global__ void topk_moe_cuda(const float *         logits,
                              float *               weights,
                              int32_t *             ids,
                              float *               bias,
                              const int             n_rows,
                              const int             n_expert_used,
                              const float           clamp_val,
                              const float           scale_val,
                              const topk_moe_config config) {
#if defined(GGML_USE_MUSA)
    // MUSA: every warp of a partially filled block must reach the barrier below.
    const int row = MIN(blockIdx.x * blockDim.y + threadIdx.y, n_rows - 1);
#else
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
#endif // defined(GGML_USE_MUSA)
    if (row >= n_rows) {
        return;
    }

    topk_moe_row<n_experts, has_bias, true>(logits + n_experts * row, weights + n_expert_used * row,
                                            ids + n_experts * row, bias, n_expert_used, clamp_val, scale_val, config);
}

// [ignite] one token's router matvec (f32 weight, as mul_mat_vec_f with 256 threads computes each row) and topk_moe in
// one launch: a block per expert row, the last block to finish runs the selection on the logits in scratch
static constexpr int ROUTER_TOPK_THREADS = 256;

template <int n_experts>
__launch_bounds__(ROUTER_TOPK_THREADS, 1)
__global__ void router_topk_moe_cuda(const float *         w,
                                     const float *         x,
                                     const int             ncols,
                                     const int64_t         stride_row,
                                     float *               logits,
                                     unsigned int *        counter,
                                     float *               weights,
                                     int32_t *             ids,
                                     const int             n_expert_used,
                                     const float           clamp_val,
                                     const float           scale_val,
                                     const topk_moe_config config) {
    constexpr int warp_size = WARP_SIZE;
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    __shared__ float buf_iw[warp_size];
    __shared__ bool  s_last;
    if (tid < warp_size) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();

    const float2 * x2 = (const float2 *) (w + row*stride_row);
    const float2 * y2 = (const float2 *) x;
    float sumf = 0.0f;
    for (int col2 = tid; col2 < ncols/2; col2 += ROUTER_TOPK_THREADS) {
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
            logits[row] = sumf;
        }
    }

    // the last block out (threadfence reduction); it resets the counter for the next launch
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        s_last = atomicAdd(counter, 1u) == (unsigned int) gridDim.x - 1;
        if (s_last) {
            atomicExch(counter, 0u);
        }
    }
    __syncthreads();
    if (!s_last || tid >= warp_size) {
        return;
    }
    __threadfence();
    topk_moe_row<n_experts, false, false>(logits, weights, ids, nullptr, n_expert_used, clamp_val, scale_val, config);
}

template<bool has_bias>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 float *                     bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val,
                                 const float                 scale_val,
                                 const topk_moe_config       config) {
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");
    const int    rows_per_block = TOPK_MOE_ROWS_PER_BLOCK;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

#if defined(GGML_USE_HIP) && HIP_VERSION < 60300000
    GGML_ASSERT(config.n_groups == 0 && "expert groups fusion is not supported on this HIP version");
#endif

    switch (n_expert) {
        case 1:
            ggml_cuda_kernel_launch(topk_moe_cuda<1, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 2:
            ggml_cuda_kernel_launch(topk_moe_cuda<2, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 4:
            ggml_cuda_kernel_launch(topk_moe_cuda<4, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 8:
            ggml_cuda_kernel_launch(topk_moe_cuda<8, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 16:
            ggml_cuda_kernel_launch(topk_moe_cuda<16, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 32:
            ggml_cuda_kernel_launch(topk_moe_cuda<32, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 64:
            ggml_cuda_kernel_launch(topk_moe_cuda<64, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 128:
            ggml_cuda_kernel_launch(topk_moe_cuda<128, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 256:
            ggml_cuda_kernel_launch(topk_moe_cuda<256, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 288: // StepFun 3.7
            ggml_cuda_kernel_launch(topk_moe_cuda<288, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 512:
            ggml_cuda_kernel_launch(topk_moe_cuda<512, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 576:
            ggml_cuda_kernel_launch(topk_moe_cuda<576, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
    }
}

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    float *       bias_d    = bias ? (float *) bias->data : nullptr;

    float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    const bool with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.delayed_softmax   = args.delayed_softmax;
    config.n_groups          = args.n_groups;
    config.n_group_used      = args.n_group_used;

    if (bias) {
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    } else {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    }
}

bool ggml_cuda_router_topk_moe(ggml_backend_cuda_context &     ctx,
                               const ggml_tensor *             router,
                               void *                          scratch,
                               const ggml_tensor *             logits,
                               ggml_tensor *                   weights,
                               ggml_tensor *                   ids,
                               const ggml_tensor *             clamp,
                               const ggml_tensor *             scale,
                               const ggml_tensor *             bias,
                               const ggml_cuda_topk_moe_args & args) {
    const ggml_tensor * w = router->src[0];
    const ggml_tensor * x = router->src[1];
    const int n_experts = logits->ne[0];
    const int ncols     = w->ne[0];
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    // mul_mat_vec_f's block size choice: these rows are bit-identical only where it picks 256 threads
    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    for (int64_t block_size = 2*warp_size; block_size <= ROUTER_TOPK_THREADS; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }
    if (block_size_best != ROUTER_TOPK_THREADS || warp_size != WARP_SIZE || bias != nullptr || args.delayed_softmax ||
            args.n_groups > 0 || n_experts != 288 || logits->ne[1] != 1 || w->type != GGML_TYPE_F32 ||
            x->type != GGML_TYPE_F32 || ncols % 2 != 0 || !ggml_is_contiguous(x) || ggml_nrows(x) != 1 ||
            w->nb[0] != sizeof(float) || w->nb[1] % (2*sizeof(float)) != 0 || ggml_nrows(w) != n_experts ||
            ((uintptr_t) w->data | (uintptr_t) x->data) % 8 != 0 || router->ne[0] != n_experts || router->ne[1] != 1) {
        return false;
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = clamp != nullptr;
    config.delayed_softmax   = false;
    config.n_groups          = 0;
    config.n_group_used      = 0;
    const float clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    const float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    unsigned int * counter = (unsigned int *) scratch;
    float *        lg      = (float *) ((char *) scratch + 256);
    router_topk_moe_cuda<288><<<n_experts, ROUTER_TOPK_THREADS, 0, ctx.stream()>>>(
        (const float *) w->data, (const float *) x->data, ncols, w->nb[1]/sizeof(float), lg, counter,
        (float *) weights->data, (int32_t *) ids->data, weights->ne[1], clamp_val, scale_val, config);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids) {
    // must match an instantiation of launch_topk_moe_cuda: a power of 2 up to 512,
    // or one of the non-power-of-2 expert counts of supported models
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512) && n_expert != 288 && n_expert != 576) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax  = gating_op;
        float               scale    = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
    }

    return true;
}
