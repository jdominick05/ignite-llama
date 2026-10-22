#pragma once

#include "common.cuh"

// [ignite] decode megakernels: a run of batch-1 graph nodes in one persistent launch, the stages separated by a grid
// barrier instead of kernel boundaries. The node bodies are the existing kernels' device code (mul_mat_vec_q_impl,
// quantize_q8_1_lane), so results are bit-identical to the per-node path. RDNA3 only. Opt-in: GGML_CUDA_MEGAKERNEL=1.

bool ggml_cuda_megakernel_enabled(int cc);

// device scratch: barrier counter, error flag, q8_1 buffers. Allocated before graph capture.
void ggml_cuda_megakernel_prepare(ggml_backend_cuda_context & ctx);

// GGML_CUDA_MEGAKERNEL_DIAG: print the barrier/visibility event log (called when the context is destroyed)
void ggml_cuda_megakernel_report(ggml_backend_cuda_context & ctx);

// qwen4exp hc mixer:
//   xn     = rms_norm(x)*w_norm                  x [ncols, hc], w_norm [ncols, hc] or [ncols]
//   lo     = w_down*xn, inject = w_inject*xn     (inject optional)
//   act    = silu(s*lo + b)
//   gate   = w_up*act                            [ncols*hc]
//   mixed  = pre_scale*sum_h xn[:, h]*sigmoid(gate[:, h])
// Only inject and mixed are written to their tensors; xn, lo, act and gate stay in megakernel scratch, so nothing
// after the segment may read them.
struct ggml_cuda_mk_hc_mix {
    const ggml_tensor * x;
    const ggml_tensor * w_norm;
    float               eps;
    ggml_tensor *       xn;
    const ggml_tensor * w_down;
    ggml_tensor *       lo;
    const ggml_tensor * w_inject; // nullptr: no inject matmul
    ggml_tensor *       inject;
    float               s;
    float               b;
    ggml_tensor *       act;
    const ggml_tensor * w_up;
    ggml_tensor *       gate;
    float               pre_scale;
    ggml_tensor *       mixed;
};

// shape and type checks that do not depend on the graph around the nodes
bool ggml_cuda_mk_hc_mix_supported(const ggml_cuda_mk_hc_mix & m);

void ggml_cuda_mk_hc_mix_run(ggml_backend_cuda_context & ctx, const ggml_cuda_mk_hc_mix & m);
