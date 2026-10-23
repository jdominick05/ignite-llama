#!/usr/bin/env python3
# Expert pruning by tensor surgery on a GGUF MoE model: keeps a chosen set of experts per layer and copies
# everything else unchanged (no requantization). Per layer, the expert tensors (*_exps, [.., .., n_expert]) and
# the router (ffn_gate_inp, [n_embd, n_expert]) are sliced to the kept experts, in ascending order; the
# <arch>.expert_count key is set to the kept count. Split input (-00001-of-0000N) is read as one model.
#
#   prune_experts.py IN.gguf OUT.gguf --prune PRUNE.json [--dry-run]
#
# PRUNE.json: {"<layer>": [expert ids to remove], ...} as written by hope_obs.py solve (awslabs HOPE format);
# every MoE layer must keep the same number of experts.
import argparse, glob, json, os, re, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))
import gguf  # noqa: E402

EXPERT_RE = re.compile(r'^blk\.(\d+)\.(ffn_(gate|up|down|gate_up)_exps\.weight|ffn_gate_inp\.weight|exp_probs_b\.bias)$')


def shards(path):
    m = re.match(r'(.*)-00001-of-(\d{5})\.gguf$', path)
    if not m:
        return [path]
    ps = sorted(glob.glob(f'{m.group(1)}-*-of-{m.group(2)}.gguf'))
    if len(ps) != int(m.group(2)):
        sys.exit(f'expected {int(m.group(2))} shards, found {len(ps)}')
    return ps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inp')
    p.add_argument('out')
    p.add_argument('--prune', required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    readers = [gguf.GGUFReader(s) for s in shards(a.inp)]
    r0 = readers[0]
    arch = r0.fields['general.architecture'].contents()
    n_expert = int(r0.fields[f'{arch}.expert_count'].contents())
    n_used = int(r0.fields[f'{arch}.expert_used_count'].contents())

    prune = {int(k): set(v) for k, v in json.load(open(a.prune)).items()}
    keep = {l: np.array(sorted(set(range(n_expert)) - v), dtype=np.int64) for l, v in prune.items()}
    sizes = {len(k) for k in keep.values()}
    if len(sizes) != 1:
        sys.exit(f'layers keep different expert counts: {sorted(sizes)}')
    n_keep = sizes.pop()
    if n_keep < n_used:
        sys.exit(f'keeping {n_keep} experts but {n_used} are used per token')

    # plan: every tensor, sliced or not
    plan = []
    for r in readers:
        for t in r.tensors:
            m = EXPERT_RE.match(t.name)
            if m:
                il = int(m.group(1))
                if il not in keep:
                    sys.exit(f'{t.name}: layer {il} missing from {a.prune}')
                # numpy view: experts on the first axis ([n_expert, ...] for both quantized bytes and F32)
                if t.data.shape[0] != n_expert:
                    sys.exit(f'{t.name}: unexpected shape {t.data.shape}')
                plan.append((t, keep[il]))
            else:
                # a tensor with an expert-sized axis that the pattern misses would be copied unsliced and break loading
                if n_expert in t.data.shape[:1] and re.match(r'^blk\.\d+\.(ffn|exp)', t.name):
                    sys.exit(f'{t.name}: first axis is {n_expert} (experts?) but the tensor is not in the slice list')
                plan.append((t, None))
    sliced = [t.name for t, k in plan if k is not None]
    layers = {int(EXPERT_RE.match(n).group(1)) for n in sliced}
    if layers != set(keep):
        sys.exit(f'layers in {a.prune} without expert tensors: {sorted(set(keep) - layers)}')
    total = sum(int(t.data[k].nbytes) if k is not None else int(t.n_bytes) for t, k in plan)
    print(f'{arch}: {n_expert} -> {n_keep} experts in {len(layers)} layers, {len(sliced)} tensors sliced, '
          f'{len(plan)} tensors, output {total / 1e9:.1f} GB')
    if a.dry_run:
        return

    w = gguf.GGUFWriter(a.out, arch=arch, endianess=r0.endianess)
    for field in r0.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.') or field.name.startswith('split.'):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        val = n_keep if field.name == f'{arch}.expert_count' else field.contents()
        w.add_key_value(field.name, val, vt, sub_type=sub)
    w.add_key_value('general.pruned_experts', f'{n_expert}->{n_keep} by {os.path.basename(a.prune)}', gguf.GGUFValueType.STRING)

    for t, k in plan:
        d = t.data if k is None else t.data[k]
        w.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    done = 0
    for i, (t, k) in enumerate(plan):
        d = t.data if k is None else np.ascontiguousarray(t.data[k])
        w.write_tensor_data(d, tensor_endianess=r0.endianess)
        done += d.nbytes
        if i % 50 == 0:
            print(f'\r{done / 1e9:.1f} / {total / 1e9:.1f} GB', end='', flush=True)
    w.close()
    print(f'\rwrote {a.out}: {done / 1e9:.1f} GB')


if __name__ == '__main__':
    main()
