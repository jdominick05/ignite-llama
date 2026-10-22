#!/usr/bin/env python3
# Bytes of VRAM read per decode token (d=0) and the bandwidth ceiling for the 27/21 XTX/XT layer split.
# usage: bw_theory.py <model.gguf> [gguf-py dir]
import sys, os, re, collections
sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import gguf
r = gguf.GGUFReader(sys.argv[1])
N_EXP, N_USED = 288, 10
lay = collections.defaultdict(lambda: collections.Counter())
glob = collections.Counter()
for t in r.tensors:
    b = t.n_bytes
    m = re.match(r"blk\.(\d+)\.(.+)", t.name)
    if not m:
        glob[t.name] += b; continue
    L, name = int(m.group(1)), m.group(2)
    if "_exps" in name:
        lay[L]["experts"] += b * N_USED / N_EXP     # only the routed experts are read
    elif name.startswith(("attn_", "indexer")):
        lay[L]["attn"] += b
    elif name.startswith("ssm_"):
        lay[L]["gdn"] += b
    elif name.startswith("hc_"):
        lay[L]["hc"] += b
    elif "shexp" in name or name == "ffn_gate_inp.weight":
        lay[L]["shared+router"] += b
    else:
        lay[L]["other"] += b
nL = len(lay)
tot = collections.Counter()
for L in lay: tot.update(lay[L])
print(f"{nL} layers; per-token VRAM reads (decode, d=0), MB:")
for k, v in tot.most_common(): print(f"  {k:15s} {v/1e6:8.1f}")
gpu_glob = glob["output.weight"] + sum(v for k, v in glob.items() if k.startswith("output_"))
print(f"  output head     {gpu_glob/1e6:8.1f}   (token_embd {glob['token_embd.weight']/1e6:.0f} MB and PLE table {glob['per_layer_token_embd.weight']/1e9:.1f} GB: gathered rows only, ~0)")
layer_bytes = [sum(lay[L].values()) for L in sorted(lay)]
total = sum(layer_bytes) + gpu_glob
print(f"total {total/1e9:.3f} GB/token")
# layer split 27 XTX / 21 XT (from the trace), output head on the last device (XT)
xtx = sum(layer_bytes[:27]); xt = sum(layer_bytes[27:]) + gpu_glob
for name, bw in (("peak", (960, 800)), ("~85% achievable", (816, 680))):
    t_xtx, t_xt = xtx / bw[0] / 1e6, xt / bw[1] / 1e6
    print(f"{name:16s}: XTX {xtx/1e9:.3f} GB -> {t_xtx:.2f} ms, XT {xt/1e9:.3f} GB -> {t_xt:.2f} ms; serial {t_xtx+t_xt:.2f} ms = {1000/(t_xtx+t_xt):.0f} tok/s")
