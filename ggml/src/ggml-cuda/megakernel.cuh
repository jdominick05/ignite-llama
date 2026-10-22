#pragma once

#include "common.cuh"

// [ignite] decode megakernels: a run of batch-1 graph nodes in one launch. Blocks hand work over through task counters
// (never by waiting on a block that may not be resident), and every node computes in its per-node kernel's order, so
// results are bit-identical to the per-node path. RDNA3 only. Opt-in: GGML_CUDA_MEGAKERNEL=1.

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

// qwen4exp gated delta net (decode, one token, one sequence), the nodes between the qkv/z and ssm_out matmuls:
//   x4    = concat(conv_state[s_copy], qkv)            conv_state (3 per channel) -> conv_dst = x4[1..3]
//   c     = silu(conv(x4, conv_w))                      [q | k | v] channels, HK q/k heads and H v heads of 128
//   q, k  = rms_norm(c_q)*scale_q, rms_norm(c_k)*scale_k
//   g     = softplus(w_alpha*x + alpha_bias)*ssm_a, beta = sigmoid(w_beta*x)
//   attn  = gated_delta_net(q, k, v, g, beta, state[s_copy]) -> state_dst
//   out   = rms_norm(attn)*w_onorm * sigmoid(z)
// One task per (v head, quarter of its columns); no grid-wide waits. Only out and the two caches are written.
struct ggml_cuda_mk_gdn {
    const ggml_tensor * s_copy;     // i32 [1]
    const ggml_tensor * conv_src;   // cache_r rows [3*n_ch, n_cells]
    ggml_tensor *       conv_dst;   // cache_r view [3*n_ch]
    const ggml_tensor * qkv;        // [n_ch]
    const ggml_tensor * conv_w;     // [4, n_ch]
    const ggml_tensor * state_src;  // cache_s rows [128*128*H, n_cells]
    ggml_tensor *       state_dst;  // cache_s view [128*128*H]
    float               eps_q;
    float               scale_q;
    float               eps_k;
    float               scale_k;
    const ggml_tensor * x;          // hc_mixed [K]
    const ggml_tensor * w_alpha;    // q8_0 [K, H]
    const ggml_tensor * alpha_bias; // [H]
    const ggml_tensor * ssm_a;      // [H]
    const ggml_tensor * w_beta;     // q8_0 [K, H]
    const ggml_tensor * z;          // [128*H]
    const ggml_tensor * w_onorm;    // [128]
    float               eps_o;
    ggml_tensor *       out;        // [128, H]
    int                 H;
    int                 HK;
};

// GGML_CUDA_MEGAKERNEL_GDN=0 turns this segment off
bool ggml_cuda_mk_gdn_enabled();

bool ggml_cuda_mk_gdn_supported(const ggml_cuda_mk_gdn & m);

void ggml_cuda_mk_gdn_run(ggml_backend_cuda_context & ctx, const ggml_cuda_mk_gdn & m);
