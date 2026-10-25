#include "models.h"
#include "llama-memory-recurrent.h"

// ignite-drafter: a small hybrid LM trained as a speculative-decoding draft model for Qwen-tokenizer targets.
//   block:  x + mixer(norm(x)) + ffn(norm(x))   (parallel, one RMSNorm per block)
//   mixer:  Gated DeltaNet (flash-linear-attention's GatedDeltaNet: short conv on q|k|v, L2-normed q/k, gated
//           output RMSNorm) on most layers, plain causal attention (GQA, RoPE on adjacent pairs) every
//           full_attention_interval-th layer
//   head:   scores a subset of the vocabulary; d2t (I64) maps head rows to token ids and the logits are scattered
//           to the full vocabulary (-inf elsewhere), so the model plugs into the standalone draft-model path.

// flash-linear-attention's defaults, which the trained weights assume
static constexpr float IDRAFT_GDN_NORM_EPS = 1e-5f;   // FusedRMSNormGated on the GDN output
static constexpr float IDRAFT_L2_NORM_EPS  = 1e-6f;   // L2 norm of q and k

void llama_model_ignite_drafter::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    uint32_t full_attn_interval = 4;
    ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        hparams.is_recr_impl[i] = (i + 1) % full_attn_interval != 0;
    }
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_ignite_drafter::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), { n_embd, n_vocab }, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);

    const struct ggml_tensor * d2t_meta = ml.get_tensor_meta("d2t");
    if (!d2t_meta) {
        throw std::runtime_error("ignite-drafter: missing d2t (head row -> token id map)");
    }
    const int64_t n_head_vocab = d2t_meta->ne[0];
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_head_vocab }, 0);
    d2t    = create_tensor(tn(LLM_TENSOR_D2T), { n_head_vocab }, 0);

    const int64_t head_k_dim = hparams.ssm_d_state;
    const int64_t n_k_heads  = hparams.ssm_n_group;
    const int64_t n_v_heads  = hparams.ssm_dt_rank;
    const int64_t key_dim    = head_k_dim * n_k_heads;
    const int64_t value_dim  = hparams.ssm_d_inner;
    const int64_t head_v_dim = value_dim / n_v_heads;
    const int64_t conv_dim   = key_dim * 2 + value_dim;

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), { n_embd }, 0);

        if (!hparams.is_recr(il)) {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", il), { n_embd, n_embd_head_k * n_head }, 0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", il), { n_embd, n_embd_k_gqa }, 0);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", il), { n_embd, n_embd_v_gqa }, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, 0);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, conv_dim }, 0);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, 0);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, 0);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { n_v_heads }, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,          il), { n_v_heads }, 0);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, 0);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, 0);
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), { n_embd, n_ff }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), { n_ff, n_embd }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_ignite_drafter::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_ignite_drafter::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();
    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * h = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(h, "attn_norm", il);

        ggml_tensor * mix = hparams.is_recr(il) ? build_layer_gdn(inp->get_recr(), h, il)
                                                : build_layer_attn(inp->get_attn(), h, inp_pos, il);
        ggml_tensor * ffn = build_ffn(h,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn, "ffn_out", il);

        if (il == n_layer - 1 && inp_out_ids) {   // only the output rows from here on
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            mix  = ggml_get_rows(ctx0, mix,  inp_out_ids);
            ffn  = ggml_get_rows(ctx0, ffn,  inp_out_ids);
        }
        inpL = ggml_add(ctx0, ggml_add(ctx0, inpL, mix), ffn);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);   // [n_head_vocab, n_outputs]

    // scatter to the full vocabulary: rows outside the head can never be drafted
    const int64_t n_head_vocab = cur->ne[0];
    const int64_t n_outputs    = cur->ne[1];
    const int64_t n_vocab      = (int64_t) model.vocab.n_tokens();
    GGML_ASSERT(model.d2t->type == GGML_TYPE_I64 && model.d2t->ne[0] == n_head_vocab);
    ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
    cur = ggml_set_rows(ctx0, logits,
            ggml_reshape_3d(ctx0, cur,       1,            n_head_vocab, n_outputs),
            ggml_reshape_3d(ctx0, model.d2t, n_head_vocab, 1,            1));
    cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_outputs);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_ignite_drafter::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_k();

    ggml_tensor * Qcur = ggml_reshape_3d(ctx0, build_lora_mm(model.layers[il].wq, cur), n_embd_head, n_head,    n_tokens);
    ggml_tensor * Kcur = ggml_reshape_3d(ctx0, build_lora_mm(model.layers[il].wk, cur), n_embd_head, n_head_kv, n_tokens);
    ggml_tensor * Vcur = ggml_reshape_3d(ctx0, build_lora_mm(model.layers[il].wv, cur), n_embd_head, n_head_kv, n_tokens);

    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                         ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                         ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    cur = build_attn(inp, nullptr, nullptr, nullptr, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                     1.0f / sqrtf(float(n_embd_head)), il);
    cur = build_lora_mm(model.layers[il].wo, cur);
    cb(cur, "attn_out", il);
    return cur;
}

ggml_tensor * llama_model_ignite_drafter::graph::build_layer_gdn(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;
    const auto & layer    = model.layers[il];

    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_inner / num_v_heads;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    ggml_tensor * qkv_mixed = ggml_reshape_3d(ctx0, build_lora_mm(layer.wqkv, cur), layer.wqkv->ne[1], n_seq_tokens, n_seqs);
    ggml_tensor * z         = build_lora_mm(layer.wqkv_gate, cur);

    ggml_tensor * beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, build_lora_mm(layer.ssm_beta, cur), 1, num_v_heads, n_seq_tokens, n_seqs));
    ggml_tensor * alpha = ggml_reshape_3d(ctx0, build_lora_mm(layer.ssm_alpha, cur), num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * gate  = ggml_mul(ctx0, ggml_softplus(ctx0, ggml_add(ctx0, alpha, layer.ssm_dt)), layer.ssm_a);   // ssm_a = -exp(A_log)
    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);
    const int64_t conv_channels   = hparams.ssm_d_inner + 2 * num_k_heads * head_k_dim;
    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, layer.ssm_conv1d->ne[0], conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);

    ggml_tensor * conv = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, layer.ssm_conv1d));
    cb(conv, "conv_output_silu", il);

    const int64_t nb1 = ggml_row_size(conv->type, conv_channels);
    ggml_tensor * q = ggml_view_4d(ctx0, conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv->type, head_k_dim), nb1, nb1 * n_seq_tokens, 0);
    ggml_tensor * k = ggml_view_4d(ctx0, conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv->type, head_k_dim), nb1, nb1 * n_seq_tokens, ggml_row_size(conv->type, head_k_dim * num_k_heads));
    ggml_tensor * v = ggml_view_4d(ctx0, conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv->type, head_v_dim), nb1, nb1 * n_seq_tokens, ggml_row_size(conv->type, 2 * head_k_dim * num_k_heads));

    q = build_gdn_l2_norm(ctx0, q, IDRAFT_L2_NORM_EPS);
    k = build_gdn_l2_norm(ctx0, k, IDRAFT_L2_NORM_EPS);
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q = ggml_repeat_4d(ctx0, q, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k = ggml_repeat_4d(ctx0, k, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    ggml_tensor * o = build_recurrent_attn(inp, ssm_states_all, q, k, v, gate, beta, state, il);

    // gated output norm: rms_norm(o) * w * silu(z), per head
    ggml_tensor * z4 = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    o = ggml_mul(ctx0, ggml_mul(ctx0, ggml_rms_norm(ctx0, o, IDRAFT_GDN_NORM_EPS), layer.ssm_norm), ggml_silu(ctx0, z4));
    o = ggml_reshape_3d(ctx0, o, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);

    cur = build_lora_mm(layer.ssm_out, o);
    cb(cur, "linear_attn_out", il);
    return ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);
}
