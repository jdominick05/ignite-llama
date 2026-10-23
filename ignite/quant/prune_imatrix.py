#!/usr/bin/env python3
# Slices a GGUF imatrix to the experts kept by an expert prune (the companion of prune_experts.py), so a pruned model
# can be requantized with the imatrix of the model it was cut from. Per-expert statistics of the *_exps tensors
# (in_sum2 / in_sum [n_expert, n_cols], counts [n_expert, 1]) keep the kept experts in ascending order, the order
# prune_experts.py writes; everything else is copied.
#
#   prune_imatrix.py IN.imatrix.gguf OUT.imatrix.gguf --prune PRUNE.json
import argparse, json, os, re, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))
import gguf  # noqa: E402

EXPS_RE = re.compile(r'^blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.weight\.(in_sum2|in_sum|counts)$')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inp')
    p.add_argument('out')
    p.add_argument('--prune', required=True)
    a = p.parse_args()

    r = gguf.GGUFReader(a.inp)
    prune = {int(k): set(v) for k, v in json.load(open(a.prune)).items()}
    n_expert = None
    for t in r.tensors:
        if EXPS_RE.match(t.name):
            n_expert = int(t.data.shape[0])
            break
    if n_expert is None:
        sys.exit('no per-expert statistics in the imatrix')
    keep = {l: np.array(sorted(set(range(n_expert)) - v)) for l, v in prune.items()}

    w = gguf.GGUFWriter(a.out, arch=None, endianess=r.endianess)
    for field in r.fields.values():
        if field.name.startswith('GGUF.'):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    n = 0
    for t in r.tensors:
        d = np.asarray(t.data)
        m = EXPS_RE.match(t.name)
        if m:
            il = int(m.group(1))
            if il not in keep or d.shape[0] != n_expert:
                sys.exit(f'{t.name}: layer missing from {a.prune} or shape {d.shape}')
            d = np.ascontiguousarray(d[keep[il]])
            n += 1
        w.add_tensor(t.name, d, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f'wrote {a.out}: {n} per-expert tensors sliced to {len(next(iter(keep.values())))} experts')


if __name__ == '__main__':
    main()
