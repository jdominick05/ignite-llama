#!/usr/bin/env python3
# Expert-aware quant simulation for qwen4exp (Flash-Next) MoE gate/up experts.
#
# Builds a copy of a base GGUF (FN-R1) in which the least-routed experts of every layer are degraded to a lower
# type and stored back in the base type, so the file keeps the base's size and runs on any build:
#   cold expert: Q8_0 source -> dequant -> quantize(cold type, imatrix) -> dequant -> quantize(base type, imatrix)
#   hot expert:  base bytes unchanged
# The extra base-type round trip makes the cold experts slightly worse than a real tiered file would be, so a
# passing KLD here is a conservative result.
#
# Usage:
#   expert_tier_sim.py validate BASE.gguf Q8.gguf IMATRIX.gguf --ggml libggml-base.so --layer 10
#       quantizes the Q8 experts of one layer straight to the base type and compares with the base bytes
#       (checks that this harness quantizes like llama-quantize did)
#   expert_tier_sim.py build BASE.gguf Q8.gguf IMATRIX.gguf OUT.gguf --ggml libggml-base.so --cold-frac 0.5 --cold-type IQ2_S
#       OUT.gguf must not exist; it is created as a copy of BASE.gguf and patched in place
#   expert_tier_sim.py plan IMATRIX.gguf --cold-frac 0.5
#       prints the share of routings the cold experts carry per layer
import argparse, ctypes, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))
import gguf
from gguf.quants import dequantize

MATS = ('ffn_gate_exps', 'ffn_up_exps')


class Ggml:
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                                 ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        self.lib.ggml_row_size.restype = ctypes.c_size_t
        self.lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self.lib.ggml_quantize_init.argtypes = [ctypes.c_int]

    def quantize(self, qtype, x, imat):
        # x: float32 [nrows, n_per_row]; imat: float32 [n_per_row] -> uint8 [nrows, row_bytes]
        x = np.ascontiguousarray(x, dtype=np.float32)
        imat = np.ascontiguousarray(imat, dtype=np.float32)
        nrows, n = x.shape
        rb = self.lib.ggml_row_size(int(qtype), n)
        out = np.empty((nrows, rb), dtype=np.uint8)
        got = self.lib.ggml_quantize_chunk(int(qtype), x.ctypes.data, out.ctypes.data, 0, nrows, n, imat.ctypes.data)
        assert got == out.nbytes, (got, out.nbytes)
        return out


def tensors(reader):
    return {t.name: t for t in reader.tensors}


def imatrix_tables(imatrix_path):
    r = gguf.GGUFReader(imatrix_path)
    T = tensors(r)
    out = {}
    for name, t in T.items():
        if not name.endswith('.in_sum2'):
            continue
        base = name[:-len('.in_sum2')]
        counts = np.array(T[base + '.counts'].data, dtype=np.float64).ravel()
        sums = np.array(t.data, dtype=np.float64).reshape(len(counts), -1)
        imat = np.where(counts[:, None] > 0, sums / np.maximum(counts[:, None], 1), 1.0).astype(np.float32)
        out[base] = (imat, counts)
    return out


def cold_set(counts, frac):
    n_cold = int(round(len(counts) * frac))
    return set(np.argsort(counts, kind='stable')[:n_cold].tolist())


def layer_ids(base_T):
    return sorted({int(n.split('.')[1]) for n in base_T if n.startswith('blk.') and n.endswith('ffn_gate_exps.weight')})


def cmd_plan(a):
    im = imatrix_tables(a.imatrix)
    shares = []
    for key in sorted((k for k in im if k.endswith('ffn_gate_exps.weight')), key=lambda k: int(k.split('.')[1])):
        imat, counts = im[key]
        c = cold_set(counts, a.cold_frac)
        shares.append(counts[list(c)].sum() / counts.sum())
    s = np.array(shares)
    print(f'cold fraction {a.cold_frac}: cold experts carry {s.mean()*100:.1f}% of routings '
          f'(layers {s.min()*100:.1f}-{s.max()*100:.1f}%)')


def expert_slabs(t):
    # quantized 3-D tensor [K, N, E] -> uint8 view [E, N, row_bytes]
    d = np.asarray(t.data)
    return d.reshape(int(t.shape[2]), int(t.shape[1]), -1)


def cmd_validate(a):
    g = Ggml(a.ggml)
    base, q8 = tensors(gguf.GGUFReader(a.base)), tensors(gguf.GGUFReader(a.q8))
    im = imatrix_tables(a.imatrix)
    for mat in MATS:
        name = f'blk.{a.layer}.{mat}.weight'
        bt, qt = base[name], q8[name]
        btype = int(bt.tensor_type)
        g.lib.ggml_quantize_init(btype)
        imat, _ = im[name]
        bs, qs = expert_slabs(bt), expert_slabs(qt)
        same = 0
        experts = range(min(a.experts, bs.shape[0]))
        for e in experts:
            x = dequantize(qs[e], gguf.GGMLQuantizationType.Q8_0)
            same += int(np.array_equal(g.quantize(btype, x, imat[e]), bs[e]))
        print(f'{name}: {same}/{len(experts)} experts byte-identical to the base')


def cmd_build(a):
    if os.path.exists(a.out):
        sys.exit(f'{a.out} exists')
    g = Ggml(a.ggml)
    cold_type = gguf.GGMLQuantizationType[a.cold_type]
    br, qr = gguf.GGUFReader(a.base), gguf.GGUFReader(a.q8)
    base, q8 = tensors(br), tensors(qr)
    im = imatrix_tables(a.imatrix)
    g.lib.ggml_quantize_init(int(cold_type))
    t0 = time.time()
    shutil.copyfile(a.base, a.out + '.part')
    print(f'copied base in {time.time() - t0:.0f} s', flush=True)
    with open(a.out + '.part', 'r+b') as f, ThreadPoolExecutor(a.threads) as pool:
        for il in layer_ids(base):
            for mat in MATS:
                name = f'blk.{il}.{mat}.weight'
                bt, qt = base[name], q8[name]
                btype = gguf.GGMLQuantizationType(int(bt.tensor_type))
                g.lib.ggml_quantize_init(int(btype))
                imat, counts = im[name]
                cold = sorted(cold_set(im[f'blk.{il}.ffn_gate_exps.weight'][1], a.cold_frac))  # same set for gate and up
                qs = expert_slabs(qt)
                slab_bytes = expert_slabs(bt)[0].nbytes

                def degrade(e):
                    x = dequantize(qs[e], gguf.GGMLQuantizationType.Q8_0)
                    y = dequantize(g.quantize(cold_type, x, imat[e]), cold_type)
                    return e, g.quantize(btype, y, imat[e])

                for e, data in pool.map(degrade, cold):
                    f.seek(bt.data_offset + e * slab_bytes)
                    f.write(data.tobytes())
            print(f'layer {il}: {len(cold)} cold experts -> {a.cold_type} ({time.time() - t0:.0f} s)', flush=True)
    os.rename(a.out + '.part', a.out)
    print('done', a.out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('validate'); b = sub.add_parser('build'); pl = sub.add_parser('plan')
    for s in (v, b):
        s.add_argument('base'); s.add_argument('q8'); s.add_argument('imatrix')
        s.add_argument('--ggml', required=True)
    v.add_argument('--layer', type=int, default=10); v.add_argument('--experts', type=int, default=16)
    b.add_argument('out'); b.add_argument('--cold-type', default='IQ2_S'); b.add_argument('--threads', type=int, default=16)
    for s in (b, pl):
        s.add_argument('--cold-frac', type=float, default=0.5)
    pl.add_argument('imatrix')
    a = p.parse_args()
    {'validate': cmd_validate, 'build': cmd_build, 'plan': cmd_plan}[a.cmd](a)


if __name__ == '__main__':
    main()
