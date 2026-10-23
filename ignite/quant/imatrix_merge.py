#!/usr/bin/env python3
# Weighted merge of GGUF imatrices of the same model. llama-imatrix --in-file a --in-file b also merges, but weights
# each file by the tokens it saw, so a large general imatrix drowns a small domain one. Here every input gets a
# token share: its sums and counts are scaled so that it contributes SHARE of the merged tokens (tokens per input =
# chunk_count * chunk_size), then all are added. Per-expert rows with no data in one input (count 0, e.g. experts an
# unpruned map-back lacks) are filled by the others.
# The derived .stats tensors are dropped (the loader ignores them; llama-imatrix --in-file OUT -o OUT2 recomputes them).
#
#   imatrix_merge.py OUT.imatrix.gguf A.imatrix.gguf:0.5 B.imatrix.gguf:0.3 C.imatrix.gguf:0.2
import argparse, os, sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))
import gguf  # noqa: E402

PARTS = ('.in_sum2', '.in_sum', '.counts')


def field(r, k, default=None):
    f = r.fields.get(k)
    return f.contents() if f is not None else default


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('inputs', nargs='+', help='FILE:SHARE (shares are normalized to sum to 1)')
    a = p.parse_args()

    ins = []
    for spec in a.inputs:
        path, _, share = spec.rpartition(':')
        if not path:
            sys.exit(f'{spec}: expected FILE:SHARE')
        r = gguf.GGUFReader(path)
        tokens = int(field(r, 'imatrix.chunk_count', 0)) * int(field(r, 'imatrix.chunk_size', 0))
        if tokens <= 0:
            sys.exit(f'{path}: no chunk_count / chunk_size metadata, cannot weight it')
        ins.append((path, float(share), tokens, r))
    total_share = sum(s for _, s, _, _ in ins)
    total_tokens = sum(t for _, _, t, _ in ins)
    chunk_size = int(field(ins[0][3], 'imatrix.chunk_size'))

    merged = OrderedDict()
    datasets = []
    for path, share, tokens, r in ins:
        scale = (share / total_share) * total_tokens / tokens
        print(f'{os.path.basename(path)}: {tokens} tokens, share {share / total_share:.0%}, scale {scale:.3f}')
        datasets += [str(d) for d in field(r, 'imatrix.datasets', [])]
        for t in r.tensors:
            if not t.name.endswith(PARTS):
                continue
            d = np.asarray(t.data, dtype=np.float64) * scale
            if t.name in merged:
                if merged[t.name].shape != d.shape:
                    sys.exit(f'{t.name}: shape {d.shape} in {path}, {merged[t.name].shape} before (different models?)')
                merged[t.name] += d
            else:
                merged[t.name] = d

    w = gguf.GGUFWriter(a.out, arch=None)
    w.add_type(gguf.GGUFType.IMATRIX)
    w.add_key_value('imatrix.datasets', datasets, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)
    w.add_key_value('imatrix.chunk_count', max(1, round(total_tokens / chunk_size)), gguf.GGUFValueType.UINT32)
    w.add_key_value('imatrix.chunk_size', chunk_size, gguf.GGUFValueType.UINT32)
    for name, d in merged.items():
        w.add_tensor(name, np.ascontiguousarray(d, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f'wrote {a.out}: {len(merged)} tensors, {total_tokens} tokens of weight')


if __name__ == '__main__':
    main()
