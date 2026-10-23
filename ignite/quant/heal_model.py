#!/usr/bin/env python3
# A Flash-Next GGUF (qwen4exp, any expert count) as a transformers Qwen4ExpForCausalLM, for router healing: the
# model is built on the meta device and each decoder layer is materialized from the GGUF (dequantized to fp32) just
# before it runs and dropped after, so it runs on a machine with less RAM than the model. The token embedding and
# the n-gram (PLE) table stay quantized and are gathered row by row; the output head is dequantized once.
#
# GGUF -> HF transforms are the inverse of the converter's (verified tensor by tensor in mlx_q41.py): V-head reorder
# of the linear-attention tensors, ssm_a = -exp(A_log), indexer q/k concatenated, Qwen4ExpTextRMSNorm weights stored
# as 1 + w.
#
#   heal_model.py check MODEL.gguf KLD_BASE.bin [--config config.json]   fidelity vs llama.cpp's logits of MODEL
import argparse, json, os, re, sys, time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from transformers.models.qwen4_exp import modeling_qwen4_exp as qm  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402

HF_CONFIG = 'https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/config.json'


def v_head_perm(num_k, v_per_k):
    """grouped-by-K -> tiled V head order (as mlx_q41.v_head_perm): gguf head i = HF head perm[i]"""
    return np.arange(num_k * v_per_k).reshape(num_k, v_per_k).T.reshape(-1)


# llama.cpp quantizes the activation side of a quantized matmul: q8_K (256-blocks) for K-quants and the 2/3-bit IQ
# types, q8_0 (32-blocks, fp16 scale) for Q8_0 / IQ4_NL / Q4_x. HEAL_ACTQ=1 emulates it, so the student computes what
# llama.cpp computes.
Q8K = {'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'IQ2_XXS', 'IQ2_XS', 'IQ2_S', 'IQ3_XXS', 'IQ3_S', 'IQ1_S', 'IQ1_M', 'IQ4_XS'}
Q80 = {'Q8_0', 'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'IQ4_NL'}

LINEAR_SRC = {   # HF Linear (under a decoder layer) -> GGUF tensor whose type sets the activation rounding
    'linear_attn.in_proj_qkv': 'attn_qkv', 'linear_attn.in_proj_z': 'attn_gate', 'linear_attn.in_proj_a': 'ssm_alpha',
    'linear_attn.in_proj_b': 'ssm_beta', 'linear_attn.out_proj': 'ssm_out',
    'self_attn.q_proj': 'attn_q', 'self_attn.k_proj': 'attn_k', 'self_attn.v_proj': 'attn_v',
    'self_attn.o_proj': 'attn_output', 'self_attn.indexer.index_qk_proj': 'indexer.q_proj',
    'mlp.shared_expert.gate_proj': 'ffn_gate_shexp', 'mlp.shared_expert.up_proj': 'ffn_up_shexp',
    'mlp.shared_expert.down_proj': 'ffn_down_shexp', 'mlp.shared_expert_gate': 'ffn_gate_inp_shexp',
    'attn_hyper_connection.input_mix_weight_down': 'hc_attn_down', 'attn_hyper_connection.input_mix_weight_up': 'hc_attn_up',
    'attn_hyper_connection.block_inject_weight': 'hc_attn_inject', 'mlp_hyper_connection.input_mix_weight_down': 'hc_ffn_down',
    'mlp_hyper_connection.input_mix_weight_up': 'hc_ffn_up', 'mlp_hyper_connection.block_inject_weight': 'hc_ffn_inject',
    'ple.key_proj': 'ple_key', 'ple.value_proj': 'ple_value',
}


def act_block(qtype):
    n = qtype.name
    return 256 if n in Q8K else 32 if n in Q80 else 0


def act_round(x, bs):
    """x rounded as llama.cpp's q8_K (bs 256) / q8_0 (bs 32) activation copy"""
    if not bs or x.shape[-1] % bs:
        return x
    xb = x.float().reshape(*x.shape[:-1], -1, bs)
    d = xb.abs().amax(-1, keepdim=True) / 127
    if bs == 32:
        d = d.half().float()
    q = torch.where(d > 0, (xb / d.clamp_min(1e-30)).round().clamp(-127, 127) * d, torch.zeros_like(xb))
    return q.reshape(x.shape).to(x.dtype)


def experts_forward_actq(self, hidden_states, top_k_index, top_k_weights):
    """Qwen4ExpTextExperts.forward with llama.cpp activation rounding on the gate_up and down inputs"""
    final = torch.zeros_like(hidden_states)
    x_in = act_round(hidden_states, self.actq[0])
    for e in top_k_index.unique().tolist():
        if e >= self.num_experts:
            continue
        token_idx, top_k_pos = torch.where(top_k_index == e)
        gate, up = nn.functional.linear(x_in[token_idx], self.gate_up_proj[e]).chunk(2, dim=-1)
        h = act_round(self.act_fn(gate) * up, self.actq[1])
        h = nn.functional.linear(h, self.down_proj[e]) * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, h.to(final.dtype))
    return final


class QuantRows(nn.Module):
    """an embedding whose quantized GGUF rows are gathered and dequantized per lookup (frozen)"""

    def __init__(self, t):
        super().__init__()
        self.data, self.qtype = t.data, t.tensor_type   # memmap [rows, row bytes]
        self.register_buffer('weight', torch.empty(0), persistent=False)   # HF reads .weight.device

    def forward(self, ids):
        flat = ids.reshape(-1).cpu().numpy()
        uniq, inv = np.unique(flat, return_inverse=True)
        rows = torch.from_numpy(dequantize(np.ascontiguousarray(self.data[uniq]), self.qtype).astype(np.float32))
        return rows[torch.from_numpy(inv)].reshape(*ids.shape, -1).to(ids.device)


class GGUFModel:
    def __init__(self, path, config_json, dtype=torch.float32):
        self.r = gguf.GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}
        self.dtype = dtype
        c = json.load(open(config_json))
        c = c.get('text_config', c)
        c['num_experts'] = int(self.t['blk.0.ffn_gate_inp.weight'].shape[1])
        self.cfg = Qwen4ExpTextConfig(**c)
        self.cfg._attn_implementation = 'sdpa'
        hp = self.cfg
        self.nk, self.nv, self.hk = hp.linear_num_key_heads, hp.linear_num_value_heads, hp.linear_key_head_dim
        self.vinv = np.argsort(v_head_perm(self.nk, self.nv // self.nk))
        with torch.device('meta'):
            self.m = qm.Qwen4ExpForCausalLM(self.cfg)
        self.m.eval()
        self._check_ple()
        mdl = self.m.model
        mdl.embed_tokens = QuantRows(self.t['token_embd.weight'])
        mdl.rotary_emb = qm.Qwen4ExpTextRotaryEmbedding(config=self.cfg)
        self.load_module(mdl.hyper_connection_mixer, self.top_tensors('model.hyper_connection_mixer.'),
                         'model.hyper_connection_mixer.')
        self.m.lm_head.to_empty(device='cpu')
        with torch.no_grad():
            self.m.lm_head.weight.copy_(self.f32('output.weight'))
        if os.environ.get('HEAL_ACTQ') == '1':
            self.emulate_act_rounding()
        for il, layer in enumerate(mdl.layers):
            layer.register_forward_pre_hook(lambda mod, args, il=il: self.materialize(il))
            layer.register_forward_hook(lambda mod, args, out, il=il: self.release(il))

    def emulate_act_rounding(self):
        def pre(bs):
            return lambda mod, args: (act_round(args[0], bs),) + tuple(args[1:])
        mdl = self.m.model
        self.m.lm_head.register_forward_pre_hook(pre(act_block(self.t['output.weight'].tensor_type)))
        mdl.hyper_connection_mixer.input_mix_weight_down.register_forward_pre_hook(pre(act_block(self.t['output_hc_down.weight'].tensor_type)))
        mdl.hyper_connection_mixer.input_mix_weight_up.register_forward_pre_hook(pre(act_block(self.t['output_hc_up.weight'].tensor_type)))
        for il, layer in enumerate(mdl.layers):
            for path, src in LINEAR_SRC.items():
                name = f'blk.{il}.{src}.weight'
                if name not in self.t:
                    continue
                layer.get_submodule(path).register_forward_pre_hook(pre(act_block(self.t[name].tensor_type)))
            ex = layer.mlp.experts
            ex.actq = (act_block(self.t[f'blk.{il}.ffn_gate_exps.weight'].tensor_type),
                       act_block(self.t[f'blk.{il}.ffn_down_exps.weight'].tensor_type))
            ex.forward = experts_forward_actq.__get__(ex)

    # --- tensors

    def f32(self, name):
        t = self.t[name]
        return torch.from_numpy(np.ascontiguousarray(dequantize(t.data, t.tensor_type), dtype=np.float32))

    def norm(self, name):
        """Qwen4ExpTextRMSNorm: x * (1 + w) in HF, x * w in llama.cpp"""
        return self.f32(name) - 1

    def top_tensors(self, prefix):
        return {prefix + 'hc_norm.weight': self.norm('output_hc_norm.weight'),
                prefix + 'input_mix_weight_down.weight': self.f32('output_hc_down.weight'),
                prefix + 'input_mix_weight_up.weight': self.f32('output_hc_up.weight')}

    def vheads(self, x, dim=0):
        """GGUF V-head order -> HF along dim (head-major blocks)"""
        x = x.movedim(dim, 0)
        sh = x.shape
        x = x.reshape(self.nv, -1, *sh[1:])[torch.from_numpy(self.vinv)].reshape(sh)
        return x.movedim(0, dim)

    def layer_tensors(self, il):
        g = lambda s: self.f32(f'blk.{il}.{s}')
        n = lambda s: self.norm(f'blk.{il}.{s}')
        has = lambda s: f'blk.{il}.{s}' in self.t
        out = {}
        if has('attn_qkv.weight'):
            qkv = g('attn_qkv.weight')
            o = 2 * self.nk * self.hk
            conv = g('ssm_conv1d.weight')          # [channels, kernel]
            out.update({
                'linear_attn.in_proj_qkv.weight': torch.cat([qkv[:o], self.vheads(qkv[o:])]),
                'linear_attn.in_proj_z.weight': self.vheads(g('attn_gate.weight')),
                'linear_attn.in_proj_a.weight': self.vheads(g('ssm_alpha.weight')),
                'linear_attn.in_proj_b.weight': self.vheads(g('ssm_beta.weight')),
                'linear_attn.out_proj.weight': self.vheads(g('ssm_out.weight'), dim=1),
                'linear_attn.A_log': self.vheads(torch.log(-g('ssm_a'))),
                'linear_attn.dt_bias': self.vheads(g('ssm_dt.bias')),
                'linear_attn.conv1d.weight': torch.cat([conv[:o], self.vheads(conv[o:])]).unsqueeze(1),
                'linear_attn.norm.weight': g('ssm_norm.weight'),    # RMSNormGated: plain w
            })
        else:
            out.update({
                'self_attn.q_proj.weight': g('attn_q.weight'), 'self_attn.k_proj.weight': g('attn_k.weight'),
                'self_attn.v_proj.weight': g('attn_v.weight'), 'self_attn.o_proj.weight': g('attn_output.weight'),
                'self_attn.q_norm.weight': n('attn_q_norm.weight'), 'self_attn.k_norm.weight': n('attn_k_norm.weight'),
                'self_attn.indexer.index_qk_proj.weight': torch.cat([g('indexer.q_proj.weight'), g('indexer.k_proj.weight')]),
                'self_attn.indexer.q_layernorm.weight': g('indexer.q_norm.weight'),
                'self_attn.indexer.k_layernorm.weight': g('indexer.k_norm.weight'),
            })
        out.update({
            'mlp.gate.weight': g('ffn_gate_inp.weight'),
            'mlp.experts.gate_up_proj': torch.cat([g('ffn_gate_exps.weight'), g('ffn_up_exps.weight')], dim=1),
            'mlp.experts.down_proj': g('ffn_down_exps.weight'),
            'mlp.shared_expert.gate_proj.weight': g('ffn_gate_shexp.weight'),
            'mlp.shared_expert.up_proj.weight': g('ffn_up_shexp.weight'),
            'mlp.shared_expert.down_proj.weight': g('ffn_down_shexp.weight'),
            'mlp.shared_expert_gate.weight': g('ffn_gate_inp_shexp.weight').reshape(1, -1),
        })
        for hc in ('attn', 'ffn'):
            p = f'{"attn" if hc == "attn" else "mlp"}_hyper_connection.'
            out.update({p + 'hc_norm.weight': n(f'hc_{hc}_norm.weight'),
                        p + 'input_mix_weight_down.weight': g(f'hc_{hc}_down.weight'),
                        p + 'input_mix_weight_up.weight': g(f'hc_{hc}_up.weight'),
                        p + 'block_inject_weight.weight': g(f'hc_{hc}_inject.weight')})
        if has('ple_key.weight'):
            out.update({'ple.key_proj.weight': g('ple_key.weight'), 'ple.value_proj.weight': g('ple_value.weight'),
                        'ple.norm_key.weight': n('ple_norm_key.weight'), 'ple.norm_query.weight': n('ple_norm_query.weight'),
                        'ple.norm_conv.weight': n('ple_norm_conv.weight'),
                        'ple.conv1d.weight': g('ple_conv1d.weight').unsqueeze(1)})
        return out

    # --- modules

    def _check_ple(self):
        """the GGUF's n-gram hashing constants must be the ones HF computes"""
        kv = lambda k: list(self.r.fields[f'qwen4exp.ple.{k}'].contents())
        for il, layer in enumerate(self.m.model.layers):
            if layer.ple is None:
                continue
            e = layer.ple.ple_embedding
            mult = qm._build_layer_multipliers(self.cfg.vocab_size, self.cfg.ngram_size, e.ple_layer_index, self.cfg.seed)
            if [int(x) for x in mult.tolist()] != [int(x) for x in kv('layer_multipliers')] \
                    or e.head_offsets != [int(x) for x in kv('head_offsets')] \
                    or e.head_vocab_sizes != [int(x) for x in kv('head_vocab_sizes')]:
                sys.exit(f'layer {il}: GGUF PLE hashing constants differ from transformers')
            self.ple = (il, mult, e.head_vocab_sizes, e.head_offsets)

    def load_module(self, mod, tensors, prefix):
        mod.to_empty(device='cpu')
        params = dict(mod.named_parameters())
        missing = set(params) - {k[len(prefix):] for k in tensors}
        if missing:
            sys.exit(f'{prefix}: no GGUF source for {sorted(missing)}')
        with torch.no_grad():
            for k, v in tensors.items():
                p = params[k[len(prefix):]]
                if tuple(p.shape) != tuple(v.shape):
                    sys.exit(f'{k}: shape {tuple(v.shape)}, want {tuple(p.shape)}')
                p.copy_(v)
        mod.to(self.dtype)

    def materialize(self, il):
        layer = self.m.model.layers[il]
        p = f'model.layers.{il}.'
        ple = None
        if layer.ple is not None:
            ple = layer.ple.ple_embedding
            layer.ple.ple_embedding = nn.Identity()       # keep the 205 GB table out of to_empty
        self.load_module(layer, {p + k: v for k, v in self.layer_tensors(il).items()}, p)
        if ple is not None:
            _, mult, sizes, offs = self.ple
            ple.ngram_embedding = QuantRows(self.t['per_layer_token_embd.weight'])
            ple.layer_multipliers = mult
            ple.ngram_heads_vocab_sizes = torch.tensor(sizes, dtype=torch.long)
            ple.ngram_heads_offsets = torch.tensor(offs, dtype=torch.long)
            layer.ple.ple_embedding = ple

    def release(self, il):
        layer = self.m.model.layers[il]
        ple = layer.ple.ple_embedding if layer.ple is not None else None
        if ple is not None:
            layer.ple.ple_embedding = nn.Identity()
        layer.to('meta')
        if ple is not None:
            layer.ple.ple_embedding = ple

    @torch.no_grad()
    def logits(self, ids):
        return self.m(input_ids=ids, use_cache=False).logits


def read_kld_base(path):
    """llama-perplexity --kl-divergence-base: tokens and, per chunk, the log-probs of positions n_ctx/2 .. n_ctx-2"""
    f = open(path, 'rb')
    if f.read(8) != b'_logits_':
        sys.exit(f'{path}: not a KLD base file')
    n_ctx, n_vocab, n_chunk = np.frombuffer(f.read(12), dtype=np.int32)
    tokens = np.frombuffer(f.read(4 * n_chunk * n_ctx), dtype=np.int32).reshape(n_chunk, n_ctx)
    nv = 2 * ((n_vocab + 1) // 2) + 4
    n_row = n_ctx - 1 - n_ctx // 2
    chunks = []
    for _ in range(n_chunk):
        raw = np.frombuffer(f.read(2 * n_row * nv), dtype=np.uint16).reshape(n_row, nv)
        head = raw[:, :4].copy().view(np.float32)                  # scale, min_log_prob
        chunks.append((head[:, 0:1], head[:, 1:2], raw[:, 4:4 + n_vocab]))
    return int(n_ctx), tokens, chunks


def cmd_check(a):
    n_ctx, tokens, chunks = read_kld_base(a.base)
    gm = GGUFModel(a.model, a.config)
    first = n_ctx // 2
    for c in range(min(a.chunks, len(chunks))):
        t0 = time.time()
        lg = gm.logits(torch.from_numpy(tokens[c].astype(np.int64))[None])[0, first:n_ctx - 1].double()
        lp = torch.log_softmax(lg, -1).numpy()
        if a.save:
            np.save(f'{a.save}.chunk{c}.npy', lp.astype(np.float32))
        scale, minlp, q = chunks[c]
        ref = np.where(q > 0, minlp + scale * q, -np.inf).astype(np.float64)
        ref -= np.logaddexp.reduce(ref, axis=-1, keepdims=True)    # renormalize the clipped tail
        p = np.exp(ref)
        kld = np.sum(np.where(p > 0, p * (ref - lp), 0), axis=-1)
        top = (ref.argmax(-1) == lp.argmax(-1)).mean()
        print(f'chunk {c}: KLD mean {kld.mean():.5f} max {kld.max():.4f}, same top-1 {100 * top:.1f}%, '
              f'{len(kld)} positions, {time.time() - t0:.0f} s', flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check')
    c.add_argument('model')
    c.add_argument('base')
    c.add_argument('--config', required=True, help=f'the HF config.json ({HF_CONFIG})')
    c.add_argument('--chunks', type=int, default=1)
    c.add_argument('--save', help='also write the student log-probs to SAVE.chunkN.npy')
    a = p.parse_args()
    torch.set_num_threads(int(os.environ.get('HEAL_THREADS', '8')))
    cmd_check(a)


if __name__ == '__main__':
    main()
