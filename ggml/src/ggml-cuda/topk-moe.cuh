#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

// Rows that one CUDA block handles.
#define TOPK_MOE_ROWS_PER_BLOCK 8

struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool sqrt_softplus{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
    int  n_groups{};     // > 0: pick experts only from the top n_group_used expert groups
    int  n_group_used{};
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);

// [ignite] the router matvec (a batch-1 f32 MUL_MAT producing logits) and topk_moe as one launch. scratch: 256 bytes
// holding a zeroed counter, then the logits. false: not supported, nothing launched
bool ggml_cuda_router_topk_moe(ggml_backend_cuda_context &     ctx,
                               const ggml_tensor *             router,
                               void *                          scratch,
                               const ggml_tensor *             logits,
                               ggml_tensor *                   weights,
                               ggml_tensor *                   ids,
                               const ggml_tensor *             clamp,
                               const ggml_tensor *             scale,
                               const ggml_tensor *             bias,
                               const ggml_cuda_topk_moe_args & args);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);
