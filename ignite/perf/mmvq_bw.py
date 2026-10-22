#!/usr/bin/env python3
# Bandwidth per mul_mat_vec_q variant from a rocprofv3 kernel_trace.csv, with weight shapes taken from the GGUF.
# usage: mmvq_bw.py <kernel_trace.csv> <model.gguf> [n_decode_tokens=33]
# Rows per block on RDNA3: nwarps (workgroup y) when the small_k template flag is set, else 1.
import csv, sys, os, collections, statistics as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import gguf

trace, model = sys.argv[1], sys.argv[2]
ntok = int(sys.argv[3]) if len(sys.argv) > 3 else 33
PEAK = {"Agent 1": 960, "Agent 2": 800}  # GB/s: XTX, XT (first and second ROCm device)

# (ggml type id, rows) -> tensor names, K, bytes per row
shapes = collections.defaultdict(list)
for t in gguf.GGUFReader(model).tensors:
    if len(t.shape) < 2 or t.tensor_type.name in ("F32",):
        continue
    k, rows = int(t.shape[0]), int(t.shape[1])
    bs, ts = gguf.GGML_QUANT_SIZES[t.tensor_type]
    name = t.name.split(".", 2)[-1] if t.name.startswith("blk.") else t.name
    shapes[(int(t.tensor_type.value), rows)].append((name, k, k // bs * ts))

agg = collections.defaultdict(list)
for r in csv.DictReader(open(trace)):
    n = r["Kernel_Name"]
    if "mul_mat_vec_q<" not in n:
        continue
    targs = n.split("mul_mat_vec_q<", 1)[1].split(">", 1)[0].split(",")
    ttype, fusion, small_k = int(targs[0].split(")")[1]), targs[2].strip() == "true", targs[3].strip() == "true"
    wgx, wgy = int(r["Workgroup_Size_X"]), int(r["Workgroup_Size_Y"])
    nblocks, channels = int(r["Grid_Size_X"]) // wgx, int(r["Grid_Size_Y"]) // wgy
    rows = nblocks * (wgy if small_k else 1)
    agg[(r["Agent_Id"], ttype, rows, channels, fusion)].append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3)

out = []
for (ag, ttype, rows, ch, fusion), d in agg.items():
    cand = shapes.get((ttype, rows), [])
    names = sorted({c[0].replace(".weight", "") for c in cand})
    row_bytes = st.median([c[2] for c in cand]) if cand else None
    nbytes = row_bytes * rows * ch * (2 if fusion else 1) if row_bytes else None
    us = st.median(d)
    gbs = nbytes / us / 1e3 if nbytes else None
    out.append((us * len(d) / ntok / 1e3, ag, len(d) / ntok, us, gbs, ttype, rows, ch, fusion, ",".join(names)[:60]))

print(f"{'ms/tok':>7} {'card':4} {'calls':>5} {'us/call':>8} {'GB/s':>6} {'%pk':>4}  type rows x ch  tensors")
for ms, ag, calls, us, gbs, ttype, rows, ch, fusion, names in sorted(out, reverse=True):
    pk = f"{100*gbs/PEAK.get(ag, 960):3.0f}%" if gbs else "   -"
    gb = f"{gbs:6.0f}" if gbs else "     -"
    card = {"Agent 1": "XTX", "Agent 2": "XT"}.get(ag, ag)
    print(f"{ms:7.3f} {card:4} {calls:5.0f} {us:8.2f} {gb} {pk}  {gguf.GGMLQuantizationType(ttype).name:7} {rows}x{ch}{' +gate' if fusion else ''}  {names}")
