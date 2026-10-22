#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// [ignite] fuses SCALE -> SIGMOID -> SCALE -> DSV4_HC_POST at node i (true = 4 nodes done)
bool ggml_cuda_dsv4_hc_post_try_fuse_gate(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
