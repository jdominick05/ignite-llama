#!/usr/bin/env python3
# Tools around llama-hope-collect output (HOPE expert pruning, arXiv 2609.18916).
#
#   hope_obs.py convert OBS.bin OUT [--sets 0,1,2,3] [--per-set]
#       checks the collector's invariants and writes OUT.h5 (the chosen sets summed; default all) in the HDF5 layout
#       of github.com/awslabs/HOPE (`hope calibrate`); --per-set also writes OUT-set<k>.h5. Prints how sparse the
#       pair statistics are (share of co-selected pairs seen fewer than 50 times).
#   hope_obs.py solve OBS.h5 --keep N [--method hope|reap|ean|man|freq] OUT.json --hope-src ~/src/HOPE
#       runs awslabs' solver (hope/solve.py) or one of its baselines (hope/baselines.py) unchanged; OUT.json lists
#       the experts to prune per layer
#   hope_obs.py compare A.json B.json [--map MAP.json]
#       per-layer overlap of the kept sets; --map translates A's expert ids (e.g. a REAP-288 obs) into B's space
#       with the per-layer index map written by expmap.py
#
# The solver modules are loaded by file path: the hope package's __init__ imports torch (calibration), which the
# solver doesn't need.
import argparse, importlib.util, json, os, sys

import numpy as np

MAGIC = b'HOPEOBS2'
VEC_KEYS = ('gate_sums', 'norm_sums', 'norm_square_sums', 'gate_norm_sums', 'gate_norm_square_sums')


def read_obs(path):
    with open(path, 'rb') as f:
        if f.read(8) != MAGIC:
            sys.exit(f'{path}: not a hope-collect file')
        n_layer, n_expert, n_used, n_sets = np.frombuffer(f.read(16), dtype=np.int32)
        n_docs, n_tokens = (int(x) for x in np.frombuffer(f.read(16), dtype=np.int64))
        E = int(n_expert)
        sets = []
        for _ in range(n_sets):
            layers = []
            for _ in range(n_layer):
                d = {'total_tokens': int(np.frombuffer(f.read(8), dtype=np.int64)[0])}
                d['norm_prod_sums'] = np.frombuffer(f.read(8 * E * E), dtype=np.float64).reshape(E, E)
                d['coselect_counts'] = np.frombuffer(f.read(8 * E * E), dtype=np.int64).reshape(E, E)
                for k in VEC_KEYS:
                    d[k] = np.frombuffer(f.read(8 * E), dtype=np.float64)
                layers.append(d)
            sets.append(layers)
        if f.read(1):
            sys.exit(f'{path}: trailing bytes')
    side = {}
    if os.path.exists(path + '.json'):
        side = json.load(open(path + '.json'))
    return dict(n_layer=int(n_layer), n_expert=E, n_used=int(n_used), n_docs=n_docs, n_tokens=n_tokens, sets=sets, side=side)


def check(layers, n_used, label):
    bad = 0
    toks = {d['total_tokens'] for d in layers}
    if len(toks) > 1:
        bad += 1
        print(f'{label}: layers saw different token counts {sorted(toks)[:4]}... (the last layer must run on all rows)')
    for il, d in enumerate(layers):
        n, C, F = d['total_tokens'], d['coselect_counts'], d['norm_prod_sums']
        errs = []
        if np.trace(C) != n * n_used:
            errs.append(f'diag counts {np.trace(C)} != {n * n_used}')
        if C.sum() != n * n_used * n_used:
            errs.append(f'all counts {C.sum()} != {n * n_used * n_used}')
        if not np.array_equal(C, C.T):
            errs.append('counts not symmetric')
        if not np.all(np.isfinite(F)) or F.min() < 0:
            errs.append('F not finite / negative')
        elif not np.allclose(F, F.T, rtol=1e-9, atol=0):
            errs.append('F not symmetric')
        if not np.allclose(np.diag(F), d['gate_norm_square_sums'], rtol=1e-9, atol=0):
            errs.append('diag F != gate_norm_square_sums')
        if n and not np.isclose(d['gate_sums'].sum(), n, rtol=1e-4):
            errs.append(f'gate sums {d["gate_sums"].sum():.1f} != {n} tokens')
        if errs:
            bad += 1
            print(f'{label} layer {il}: ' + '; '.join(errs))
    return bad


def write_h5(path, layers):
    import h5py
    with h5py.File(path, 'w') as f:
        for il, d in enumerate(layers):
            g = f.create_group('layer_%d' % il)
            for k, v in d.items():
                g.create_dataset(k, data=v)


def cmd_convert(a):
    o = read_obs(a.obs)
    pick = [int(x) for x in a.sets.split(',')] if a.sets else list(range(len(o['sets'])))
    chosen = [o['sets'][k] for k in pick]
    total = [{k: sum(s[il][k] for s in chosen) for k in chosen[0][il]} for il in range(o['n_layer'])]
    bad = sum(check(o['sets'][k], o['n_used'], f'set {k}') for k in pick) + check(total, o['n_used'], 'sum')
    set_toks = [s[0]['total_tokens'] for s in o['sets']]
    live = [int((np.diag(d['coselect_counts']) > 0).sum()) for d in total]
    off = [d['coselect_counts'][~np.eye(o['n_expert'], dtype=bool)] for d in total]
    seen = np.concatenate([x[x > 0] for x in off])
    print(f'{o["n_layer"]} layers, {o["n_expert"]} experts, top-{o["n_used"]}, {o["n_docs"]} docs, {o["n_tokens"]} tokens; '
          f'tokens per set {set_toks}, summing sets {pick}')
    print(f'experts never selected per layer: min {o["n_expert"] - max(live)} max {o["n_expert"] - min(live)}; '
          f'expert pairs never co-selected: {np.mean([(x == 0).mean() for x in off])*100:.1f}%; '
          f'of co-selected pairs, seen < 50 times: {(seen < 50).mean()*100:.1f}%, median {int(np.median(seen))}')
    if o['side'].get('src_tokens'):
        st = o['side']['src_tokens']; n = sum(st.values())
        print('tokens by source: ' + ', '.join(f'{k} {v/n*100:.1f}%' for k, v in sorted(st.items(), key=lambda x: -x[1])))
    if bad:
        sys.exit(f'{bad} layer checks failed, nothing written')
    write_h5(a.out + '.h5', total)
    if a.per_set:
        for k in pick:
            write_h5(f'{a.out}-set{k}.h5', o['sets'][k])
    print(f'wrote {a.out}.h5' + (' and per-set files' if a.per_set else ''))


def load_module(src, name):
    spec = importlib.util.spec_from_file_location('hope_' + name, os.path.join(src, 'hope', name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cmd_solve(a):
    import h5py
    with h5py.File(a.obs, 'r') as f:
        E = f['layer_0']['coselect_counts'].shape[0]
    if not 0 < a.keep < E:
        sys.exit(f'--keep must be in 1..{E - 1}')
    src = os.path.expanduser(a.hope_src)
    if a.method == 'hope':
        load_module(src, 'solve').solve(a.obs, a.out, prune_num=E - a.keep)
    else:
        load_module(src, 'baselines').solve_baselines(a.obs, a.out, a.method, prune_num=E - a.keep)


def kept(pruneset, n_expert):
    return {int(l): sorted(set(range(n_expert)) - set(v)) for l, v in pruneset.items()}


def cmd_compare(a):
    A, B = json.load(open(a.a)), json.load(open(a.b))
    ka, kb = kept(A, a.experts_a), kept(B, a.experts_b)
    if a.map:
        m = {int(l): v for l, v in json.load(open(a.map)).items()}
        ka = {l: sorted(m[l][e] for e in v) for l, v in ka.items()}
    ov = []
    for l in sorted(ka):
        sa, sb = set(ka[l]), set(kb[l])
        ov.append(len(sa & sb) / min(len(sa), len(sb)))
    ov = np.array(ov)
    print(f'kept-set overlap (|A & B| / min size) over {len(ov)} layers: mean {ov.mean()*100:.1f}%, '
          f'min {ov.min()*100:.1f}% (layer {int(np.argmin(ov))}), max {ov.max()*100:.1f}%')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('convert'); c.add_argument('obs'); c.add_argument('out')
    c.add_argument('--sets', help='comma-separated set ids to sum (default all)'); c.add_argument('--per-set', action='store_true')
    s = sub.add_parser('solve'); s.add_argument('obs'); s.add_argument('out')
    s.add_argument('--keep', type=int, required=True)
    s.add_argument('--method', default='hope', choices=('hope', 'reap', 'ean', 'man', 'freq'))
    s.add_argument('--hope-src', default='~/src/HOPE')
    m = sub.add_parser('compare'); m.add_argument('a'); m.add_argument('b')
    m.add_argument('--experts-a', type=int, required=True); m.add_argument('--experts-b', type=int, required=True)
    m.add_argument('--map')
    a = p.parse_args()
    {'convert': cmd_convert, 'solve': cmd_solve, 'compare': cmd_compare}[a.cmd](a)


if __name__ == '__main__':
    main()
