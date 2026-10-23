#!/usr/bin/env python3
# Replicates an MLX affine 4-bit qwen4exp checkpoint (mlx-vlm layout, e.g. sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit)
# as a GGUF, without requantizing: an MLX group of g values (scale, bias, 4-bit codes) is g/32 Q4_1 blocks
# (d, m, codes) with the same numbers, so every quantized weight is carried over exactly (up to the bf16 -> fp16
# cast of the scales and biases, which is reported). A GGUF of the same model with the same experts is the template:
# its metadata, tensor order and names are kept, its unquantized tensors (norms, routers, ssm constants, convs) are
# copied after checking them against the MLX bf16 values, and each quantized tensor is replaced by the MLX one in
# Q4_1, with the converter's transforms (V-head reorder, indexer q/k split, expert order) applied as whole-row or
# whole-block moves only.
#
#   mlx_q41.py MLX_DIR TEMPLATE.gguf OUT.gguf [--check] [--layers 0,1,3]
#
# --check maps and verifies every tensor (dequantized cosine against the template) and writes nothing.
import argparse, json, os, re, struct, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

Q41 = gguf.GGMLQuantizationType.Q4_1
ST_DTYPES = {'U32': np.uint32, 'BF16': np.uint16, 'F32': np.float32, 'F16': np.float16, 'I64': np.int64}
FP16_MIN_NORMAL = 6.104e-5


class SafeTensors:
    """the MLX shards, memory-mapped; bf16 comes back as raw uint16"""

    def __init__(self, d):
        self.d = d
        self.wmap = json.load(open(os.path.join(d, 'model.safetensors.index.json')))['weight_map']
        self.files = {}

    def _file(self, f):
        if f not in self.files:
            path = os.path.join(self.d, f)
            with open(path, 'rb') as fh:
                n = struct.unpack('<Q', fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            self.files[f] = (np.memmap(path, dtype=np.uint8, mode='r'), 8 + n, hdr)
        return self.files[f]

    def __contains__(self, name):
        return name in self.wmap

    def get(self, name):
        mm, base, hdr = self._file(self.wmap[name])
        h = hdr[name]
        a, b = h['data_offsets']
        return mm[base + a:base + b].view(ST_DTYPES[h['dtype']]).reshape(h['shape'])


def bf16(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


class Stats:
    def __init__(self):
        self.vals = 0
        self.sub = 0      # nonzero scale/bias magnitudes below fp16's smallest normal
        self.inf = 0
        self.maxrel = 0.0

    def add(self, f32):
        f16 = f32.astype(np.float16)
        self.vals += f32.size
        a = np.abs(f32)
        self.sub += int(np.count_nonzero((a > 0) & (a < FP16_MIN_NORMAL)))
        self.inf += int(np.count_nonzero(~np.isfinite(f16)))
        nz = a >= FP16_MIN_NORMAL
        if nz.any():
            self.maxrel = max(self.maxrel, float(np.max(np.abs(f16[nz].astype(np.float32) - f32[nz]) / a[nz])))
        return f16


def q41(w, s, b, stats):
    """MLX (codes u32 [..., n/8], scales/biases bf16 [..., n/g]) -> Q4_1 bytes [..., n/32 * 20]"""
    lead = w.shape[:-1]
    n = w.shape[-1] * 8
    g = n // s.shape[-1]
    assert g % 32 == 0 and n % 32 == 0, (w.shape, s.shape)
    by = np.ascontiguousarray(w).view(np.uint8).reshape(*lead, n // 2)
    q = np.empty((*lead, n), np.uint8)
    q[..., 0::2] = by & 15
    q[..., 1::2] = by >> 4
    q = q.reshape(*lead, n // 32, 32)
    qs = q[..., :16] | (q[..., 16:] << 4)
    rep = g // 32
    d = np.repeat(stats.add(bf16(s)), rep, axis=-1)
    m = np.repeat(stats.add(bf16(b)), rep, axis=-1)
    out = np.empty((*lead, n // 32, 20), np.uint8)
    out[..., 0:2] = d.view(np.uint8).reshape(*lead, n // 32, 2)
    out[..., 2:4] = m.view(np.uint8).reshape(*lead, n // 32, 2)
    out[..., 4:] = qs
    return out.reshape(*lead, n // 32 * 20)


def v_head_perm(num_k, v_per_k):
    """grouped-by-K -> tiled V head order (conversion/qwen.py _reorder_v_heads)"""
    return np.arange(num_k * v_per_k).reshape(num_k, v_per_k).T.reshape(-1)


class Replica:
    def __init__(self, mlx_dir, template):
        self.st = SafeTensors(mlx_dir)
        cfg = json.load(open(os.path.join(mlx_dir, 'config.json')))
        self.hp = cfg.get('text_config', cfg)
        self.r = gguf.GGUFReader(template)
        self.stats = Stats()
        hp = self.hp
        self.nk, self.nv = hp['linear_num_key_heads'], hp['linear_num_value_heads']
        self.hk, self.hv = hp['linear_key_head_dim'], hp['linear_value_head_dim']
        self.vperm = v_head_perm(self.nk, self.nv // self.nk)
        self.n_iq = hp['indexer_n_heads'] * hp['indexer_head_dim']
        self.expert_perm = {}

    P = 'language_model.model.'

    def quant(self, base):
        """the MLX quantized tensor `base` (without .weight) as Q4_1 rows"""
        st = self.st
        return q41(st.get(base + '.weight'), st.get(base + '.scales'), st.get(base + '.biases'), self.stats)

    def experts(self, il, tmpl):
        """per layer: template expert e is MLX expert perm[e], matched by the router rows (bf16 -> f32 is exact)"""
        if il not in self.expert_perm:
            mine = bf16(self.st.get(f'{self.P}layers.{il}.mlp.gate.weight'))
            theirs = np.asarray(tmpl.data)
            idx = {r.tobytes(): i for i, r in enumerate(mine)}
            perm = np.array([idx.get(r.tobytes(), -1) for r in theirs])
            if (perm < 0).any() or len(set(perm.tolist())) != len(perm):
                sys.exit(f'layer {il}: router rows do not match one to one ({int((perm < 0).sum())} unmatched)')
            self.expert_perm[il] = perm
        return self.expert_perm[il]

    def source(self, t):
        """(kind, producer) for a template tensor: ('q41', fn -> bytes) / ('copy', mlx float, transform)"""
        name = t.name
        P = self.P
        m = re.match(r'^blk\.(\d+)\.(.+)$', name)
        if not m:
            simple = {'token_embd.weight': 'embed_tokens', 'output.weight': '<lm_head>',
                      'output_hc_down.weight': 'hyper_connection_mixer.input_mix_weight_down',
                      'output_hc_up.weight': 'hyper_connection_mixer.input_mix_weight_up'}
            if name in simple:
                b = 'language_model.lm_head' if simple[name] == '<lm_head>' else P + simple[name]
                return 'q41', lambda: self.quant(b)
            if name == 'output_hc_norm.weight':
                return 'float', P + 'hyper_connection_mixer.hc_norm.weight'
            if name == 'per_layer_token_embd.weight':
                return 'table', None
            sys.exit(f'{name}: no mapping')
        il, rest = int(m.group(1)), m.group(2)
        L = f'{P}layers.{il}.'
        q = {
            'attn_q.weight': 'self_attn.q_proj', 'attn_k.weight': 'self_attn.k_proj',
            'attn_v.weight': 'self_attn.v_proj', 'attn_output.weight': 'self_attn.o_proj',
            'ffn_gate_shexp.weight': 'mlp.shared_expert.gate_proj', 'ffn_up_shexp.weight': 'mlp.shared_expert.up_proj',
            'ffn_down_shexp.weight': 'mlp.shared_expert.down_proj',
            'hc_attn_down.weight': 'attn_hyper_connection.input_mix_weight_down',
            'hc_attn_up.weight': 'attn_hyper_connection.input_mix_weight_up',
            'hc_attn_inject.weight': 'attn_hyper_connection.block_inject_weight',
            'hc_ffn_down.weight': 'mlp_hyper_connection.input_mix_weight_down',
            'hc_ffn_up.weight': 'mlp_hyper_connection.input_mix_weight_up',
            'hc_ffn_inject.weight': 'mlp_hyper_connection.block_inject_weight',
            'ple_key.weight': 'ple.key_proj', 'ple_value.weight': 'ple.value_proj',
        }
        if rest in q:
            return 'q41', lambda: self.quant(L + q[rest])
        hd = self.hv
        if rest == 'attn_qkv.weight':
            def f():
                x = self.quant(L + 'linear_attn.in_proj_qkv')
                o = 2 * self.nk * self.hk
                v = x[o:].reshape(self.nv, hd, -1)[self.vperm].reshape(self.nv * hd, -1)
                return np.concatenate([x[:o], v])
            return 'q41', f
        if rest == 'attn_gate.weight':
            return 'q41', lambda: self.quant(L + 'linear_attn.in_proj_z').reshape(self.nv, hd, -1)[self.vperm].reshape(self.nv * hd, -1)
        if rest in ('ssm_alpha.weight', 'ssm_beta.weight'):
            src = 'in_proj_a' if rest == 'ssm_alpha.weight' else 'in_proj_b'
            return 'q41', lambda: self.quant(L + 'linear_attn.' + src)[self.vperm]
        if rest == 'ssm_out.weight':
            def f():
                x = self.quant(L + 'linear_attn.out_proj')   # [2560, 6144/32*20]: input columns are V heads
                rows = x.shape[0]
                return np.ascontiguousarray(x.reshape(rows, self.nv, -1)[:, self.vperm].reshape(rows, -1))
            return 'q41', f
        if rest in ('indexer.q_proj.weight', 'indexer.k_proj.weight'):
            sl = slice(0, self.n_iq) if rest.startswith('indexer.q') else slice(self.n_iq, None)
            return 'q41', lambda: self.quant(L + 'self_attn.indexer.index_qk_proj')[sl]
        e = re.match(r'^ffn_(gate|up|down)_exps\.weight$', rest)
        if e:
            def f(proj=e.group(1)):
                gate = next(x for x in self.r.tensors if x.name == f'blk.{il}.ffn_gate_inp.weight')
                perm = self.experts(il, gate)
                b = f'{L}mlp.switch_mlp.{proj}_proj'
                st = self.st
                return q41(st.get(b + '.weight')[perm], st.get(b + '.scales')[perm], st.get(b + '.biases')[perm], self.stats)
            return 'q41', f
        floats = {
            'attn_q_norm.weight': 'self_attn.q_norm.weight', 'attn_k_norm.weight': 'self_attn.k_norm.weight',
            'indexer.q_norm.weight': 'self_attn.indexer.q_layernorm.weight',
            'indexer.k_norm.weight': 'self_attn.indexer.k_layernorm.weight',
            'hc_attn_norm.weight': 'attn_hyper_connection.hc_norm.weight',
            'hc_ffn_norm.weight': 'mlp_hyper_connection.hc_norm.weight',
            'ffn_gate_inp.weight': 'mlp.gate.weight', 'ffn_gate_inp_shexp.weight': 'mlp.shared_expert_gate.weight',
            'ssm_a': 'linear_attn.A_log', 'ssm_dt.bias': 'linear_attn.dt_bias',
            'ssm_conv1d.weight': 'linear_attn.conv1d.weight', 'ssm_norm.weight': 'linear_attn.norm.weight',
            'ple_conv1d.weight': 'ple.conv1d.weight', 'ple_norm_conv.weight': 'ple.norm_conv.weight',
            'ple_norm_key.weight': 'ple.norm_key.weight', 'ple_norm_query.weight': 'ple.norm_query.weight',
        }
        if rest in floats:
            return 'float', L + floats[rest]
        sys.exit(f'{name}: no mapping')

    def check_float(self, t, src):
        """the template's float tensor must equal the MLX bf16 one under a known converter transform"""
        mine = bf16(self.st.get(src)).astype(np.float64)
        ref = np.asarray(t.data, dtype=np.float64)
        cands = {'same': mine, '+1': mine + 1, '-exp': -np.exp(mine)}
        out = {}
        for k, v in cands.items():
            v = v.reshape(-1)
            n = v.size
            # V-head reorders: whole vector (per-head scalars) or the V part of the conv channels
            if n == self.nv:
                out[k + ',vperm'] = v[self.vperm]
            if src.endswith('linear_attn.conv1d.weight'):
                c = v.reshape(-1, 4)
                o = 2 * self.nk * self.hk
                vv = c[o:].reshape(self.nv, self.hv, 4)[self.vperm].reshape(-1, 4)
                out[k + ',vperm'] = np.concatenate([c[:o], vv]).reshape(-1)
            if src.endswith('ple.conv1d.weight'):
                out[k + ',T'] = v.reshape(-1, 4).T.reshape(-1)
            out[k] = v
        r = ref.reshape(-1)
        best = None
        for k, v in out.items():
            if v.size != r.size:
                continue
            err = float(np.max(np.abs(v - r) / np.maximum(np.abs(r), 1e-3)))
            if best is None or err < best[1]:
                best = (k, err, v)
        return best

    def check_q(self, t, x):
        """cosine of a row sample: our Q4_1 vs the template's quantization of the same weights"""
        tq = t.tensor_type
        td = np.asarray(t.data)
        xs, ts = x.reshape(-1, x.shape[-1]), td.reshape(-1, td.shape[-1])
        if xs.shape[0] != ts.shape[0]:
            return None
        rows = np.linspace(0, xs.shape[0] - 1, min(64, xs.shape[0])).astype(np.int64)
        a = dequantize(xs[rows], Q41).astype(np.float64).ravel()
        b = dequantize(np.ascontiguousarray(ts[rows]), tq).astype(np.float64).ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def table_chunks(rep):
    """the n-gram table, shard by shard in index order (conversion/qwen4exp.py _place_ple_shard)"""
    P = rep.P
    names = sorted({k.rsplit('.', 1)[0] for k in rep.st.wmap if '.ngram_embedding.shards.' in k},
                   key=lambda k: int(k.rsplit('.', 1)[1]))
    if len(names) != rep.hp['split_ngram_parts']:
        sys.exit(f'{len(names)} table shards, expected {rep.hp["split_ngram_parts"]}')
    for b in names:
        n = rep.st.get(b + '.weight').shape[0]
        for i in range(0, n, 1 << 20):
            sl = slice(i, i + (1 << 20))
            yield q41(rep.st.get(b + '.weight')[sl], rep.st.get(b + '.scales')[sl], rep.st.get(b + '.biases')[sl], rep.stats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mlx')
    p.add_argument('template')
    p.add_argument('out')
    p.add_argument('--check', action='store_true')
    p.add_argument('--layers', help='with --check: only these layers (comma list)')
    a = p.parse_args()

    rep = Replica(a.mlx, a.template)
    r = rep.r
    arch = r.fields['general.architecture'].contents()
    only = {int(x) for x in a.layers.split(',')} if a.layers else None

    plan = []
    worst = []
    for t in r.tensors:
        m = re.match(r'^blk\.(\d+)\.', t.name)
        if a.check and only is not None and m and int(m.group(1)) not in only:
            continue
        kind, src = rep.source(t)
        if kind == 'float':
            best = rep.check_float(t, src)
            if best is None or best[1] > 1e-2:
                sys.exit(f'{t.name}: template float tensor does not match {src} under any known transform ({best and best[:2]})')
            if a.check:
                print(f'{t.name:34s} float  {t.tensor_type.name:7s} = mlx {best[0]:10s} max rel err {best[1]:.1e}')
            # the MLX values (bf16, e.g. norms re-centred in bf16), in the template's layout and type
            arr = best[2].astype(np.asarray(t.data).dtype).reshape(np.asarray(t.data).shape)
            plan.append((t, 'copy', arr))
        elif kind == 'q41':
            if a.check:
                x = src()
                c = rep.check_q(t, x)
                if c is None or c < 0.9:
                    sys.exit(f'{t.name}: shape/order mismatch against the template (cos {c}, {x.shape} vs {t.data.shape})')
                worst.append((c, t.name, t.tensor_type.name))
                print(f'{t.name:34s} q4_1 vs {t.tensor_type.name:7s} cos {c:.4f}')
            plan.append((t, 'q41', src))
        else:
            plan.append((t, 'table', None))

    s = rep.stats
    if a.check:
        worst.sort()
        print('lowest cosines:', ', '.join(f'{n} {c:.4f} ({ty})' for c, n, ty in worst[:5]))
        print(f'bf16 -> fp16 scales/biases: {s.vals} values, {s.sub} below fp16 normal, {s.inf} overflow, '
              f'max rel err {s.maxrel:.1e} (normal range)')
        return

    def nbytes(t, kind):
        if kind == 'copy':
            return int(t.n_bytes)
        ne = [int(x) for x in t.shape]
        return int(np.prod(ne[1:])) * ne[0] // 32 * 20

    w = gguf.GGUFWriter(a.out, arch=arch, endianess=r.endianess)
    for field in r.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.') or field.name.startswith('split.'):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    w.add_key_value('general.requantized_from', f'MLX affine 4-bit ({os.path.basename(os.path.normpath(a.mlx))}) as Q4_1',
                    gguf.GGUFValueType.STRING)
    total = 0
    for t, kind, _ in plan:
        nb = nbytes(t, kind)
        total += nb
        if kind == 'copy':
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, nb, t.tensor_type)
        else:
            ne = [int(x) for x in t.shape]
            w.add_tensor_info(t.name, (*reversed(ne[1:]), ne[0] // 32 * 20), np.uint8, nb, Q41)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()

    fout = w.fout[0]
    done = 0
    for i, (t, kind, src) in enumerate(plan):
        ti = w.tensors[0].pop(t.name)
        w.write_padding(fout, fout.tell())
        chunks = [src] if kind == 'copy' else [src()] if kind == 'q41' else table_chunks(rep)
        n = 0
        for c in chunks:
            c = np.ascontiguousarray(c)
            c.tofile(fout)
            n += c.nbytes
            done += c.nbytes
            print(f'\r{done / 1e9:.1f} / {total / 1e9:.1f} GB  {t.name:40s}', end='', flush=True)
        if n != ti.nbytes:
            sys.exit(f'\n{t.name}: wrote {n} bytes, expected {ti.nbytes}')
        w.write_padding(fout, n)
    w.state = gguf.gguf_writer.WriterState.WEIGHTS
    w.close()
    print(f'\rwrote {a.out}: {done / 1e9:.1f} GB')
    print(f'bf16 -> fp16 scales/biases: {s.vals} values, {s.sub} below fp16 normal, {s.inf} overflow, max rel err {s.maxrel:.1e}')


if __name__ == '__main__':
    main()
