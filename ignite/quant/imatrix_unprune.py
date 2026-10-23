#!/usr/bin/env python3
# Maps an imatrix computed on an expert-pruned model back into the unpruned model's expert space (the inverse of
# prune_imatrix.py), so statistics collected cheaply on a pruned build can be merged with imatrices of the full model
# (imatrix_merge.py). Kept experts are in ascending order, the order prune_experts.py writes; pruned experts get zero
# sums and counts, which a merge treats as "no data". Non-expert tensors are copied. The derived .stats tensors are
# dropped (the loader ignores them; llama-imatrix --in-file OUT -o OUT2 recomputes them).
#
#   imatrix_unprune.py PRUNED.imatrix.gguf OUT.imatrix.gguf --prune PRUNE.json [--n-expert 512]
import argparse, json, os, re, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))
import gguf  # noqa: E402

EXPS_RE = re.compile(r'^blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.weight\.(in_sum2|in_sum|counts)$')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inp')
    p.add_argument('out')
    p.add_argument('--prune', required=True, help='the prune set the model was cut with (layer -> removed experts)')
    p.add_argument('--n-expert', type=int, default=512)
    a = p.parse_args()

    r = gguf.GGUFReader(a.inp)
    prune = {int(k): set(v) for k, v in json.load(open(a.prune)).items()}
    keep = {l: np.array(sorted(set(range(a.n_expert)) - v)) for l, v in prune.items()}

    w = gguf.GGUFWriter(a.out, arch=None, endianess=r.endianess)
    for field in r.fields.values():
        if field.name.startswith('GGUF.') or field.name == 'imatrix.stats_schema':
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    n = 0
    for t in r.tensors:
        if t.name.endswith('.stats'):
            continue
        d = np.asarray(t.data, dtype=np.float32)
        m = EXPS_RE.match(t.name)
        if m:
            il = int(m.group(1))
            if il not in keep or d.shape[0] != len(keep[il]):
                sys.exit(f'{t.name}: {d.shape[0]} experts, but {a.prune} keeps {len(keep.get(il, []))} in layer {il}')
            full = np.zeros((a.n_expert,) + d.shape[1:], dtype=np.float32)
            full[keep[il]] = d
            d = full
            n += 1
        w.add_tensor(t.name, np.ascontiguousarray(d))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f'wrote {a.out}: {n} per-expert tensors mapped to {a.n_expert} experts')


if __name__ == '__main__':
    main()
