#!/usr/bin/env python3
# Decision-point KLD: where does a pruned model depart from the reference along the reference's own replies?
#
#   kld_pos.py pack DUMP.jsonl.gz IN.kldp                 rare_tokens.py --dump of the REFERENCE -> llama-kld-pos input
#   kld_pos.py report DUMP.jsonl.gz OUT.kldo [OUT2.kldo]  per-position results of one or two models, tagged
#
# Tags (positions are the reference's generated tokens, teacher-forced):
#   think_end   the reference emits </think> here (thinking stops)      -> also reports the model's p(</think>)
#   answer_k    the first 8 tokens after </think> (refuse vs comply)    -> model log-prob of the reference tokens
#   thinking    other positions before </think>;   answer   later answer positions
# Per template too (T3 = "add to blocklist and confirm", where REAP-308 refuses more). Aggregates only.
import gzip, json, struct, sys
import numpy as np

THINK_END = 248069   # '</think>' in the Qwen3.5/3.8 vocab (checked by llama-kld-pos, which reports lp_end for it)


def load_dump(path):
    return [json.loads(l) for l in gzip.open(path, 'rt')]


def pack(dump, out):
    d = load_dump(dump); K = len(d[0]['top_ids'][0])
    with open(out, 'wb') as f:
        f.write(b'KLDP' + struct.pack('<II', len(d), K))
        for r in d:
            g = r['gen_ids']; ti = np.array(r['top_ids'], np.int32); tl = np.array(r['top_lp'], np.float32)
            if ti.shape != (len(g), K): sys.exit(f'sample with ragged top-k: {ti.shape}')
            f.write(struct.pack('<II', len(r['prompt_ids']), len(g)))
            f.write(np.array(r['prompt_ids'] + g, np.int32).tobytes()); f.write(ti.tobytes()); f.write(tl.tobytes())
    print(f'{len(d)} samples, K={K}, {sum(len(r["gen_ids"]) for r in d)} positions -> {out}')


def load_out(path):
    b = open(path, 'rb').read(); assert b[:4] == b'KLDO'
    n = struct.unpack_from('<I', b, 4)[0]; o = 8; res = []
    for _ in range(n):
        ng = struct.unpack_from('<I', b, o)[0]; o += 4
        a = np.frombuffer(b, np.float32, ng * 5, o).reshape(ng, 5).copy(); o += ng * 20
        res.append(a)
    return res


def tags(gen):
    t = np.array(['thinking'] * len(gen), dtype=object)
    if THINK_END in gen:
        e = gen.index(THINK_END); t[e] = 'think_end'; t[e + 1:] = 'answer'; t[e + 1:e + 9] = 'answer_k'
    else:
        t[:] = 'thinking'   # cut off while thinking: no answer positions
    return t


def report(dump, outs):
    d = load_dump(dump); models = [(p.rsplit('/', 1)[-1], load_out(p)) for p in outs]
    for name, res in models: assert len(res) == len(d), f'{name}: {len(res)} samples vs {len(d)} in the dump'
    rows = {}
    for i, r in enumerate(d):
        t = tags(r['gen_ids'])
        for name, res in models:
            a = res[i]
            for tag in ('think_end', 'answer_k', 'thinking', 'answer'):
                m = t == tag
                if not m.any(): continue
                for key in (tag, f'{tag} T{r["tpl"]}'):
                    s = rows.setdefault((name, key), {'n': 0, 'kld': 0.0, 'lp': 0.0, 'pend': 0.0})
                    s['n'] += int(m.sum()); s['kld'] += float(a[m, 0].sum()); s['lp'] += float(a[m, 1].sum())
                    s['pend'] += float(np.exp(a[m, 2]).sum())
    keys = sorted({k for _, k in rows}, key=lambda k: (k.split()[0], k))
    print(f'{len(d)} reference replies; per tag: positions, mean KLD(ref||model), mean model log-prob of the ref token'
          + ('' if len(models) < 2 else '; columns per model: ' + ' | '.join(n for n, _ in models)))
    for k in keys:
        cells = []
        for name, _ in models:
            s = rows.get((name, k))
            if not s: cells.append(' ' * 30); continue
            extra = f' p(</think>) {s["pend"] / s["n"]:.3f}' if k.startswith('think_end') else ''
            cells.append(f'n {s["n"]:6d} KLD {s["kld"] / s["n"]:.4f} lp {s["lp"] / s["n"]:7.3f}{extra}')
        print(f'  {k:16s} ' + ' | '.join(cells))


if __name__ == '__main__':
    if sys.argv[1] == 'pack': pack(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'report': report(sys.argv[2], sys.argv[3:])
    else: sys.exit(__doc__)
