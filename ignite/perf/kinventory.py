#!/usr/bin/env python3
# Kernel inventory per decode token from a rocprofv3 kernel_trace.csv (layer split over 2 GPUs).
# Agent 1 (first device) splits into per-token chunks at gaps > 1 ms; the other agent's clock is offset, so its
# per-token kernel count comes from windowing by agent 1's token starts and its times are summed over the same tail.
import csv, sys, collections, statistics as st

rows = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    rows[r["Agent_Id"]].append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
for k in rows: rows[k].sort()
agents = sorted(a for a in rows if rows[a])
a0 = agents[0]
ch = [[rows[a0][0]]]
for k in rows[a0][1:]:
    (ch.append([k]) if k[0] - ch[-1][-1][1] > 1e6 else ch[-1].append(k))
n0 = collections.Counter(len(c) for c in ch).most_common(1)[0][0]
starts = [c[0][0] for c in ch if abs(len(c) - n0) <= 0.02 * n0]
ntok = len(starts) - 1
per_tok = {a0: n0}
for a in agents[1:]:
    cnt = [sum(1 for k in rows[a] if t0 <= k[0] < t1) for t0, t1 in zip(starts, starts[1:])]
    per_tok[a] = int(st.median(cnt))

def short(n):
    n = n.replace("void ", "")
    base = n.split("(")[0]
    if base.startswith("mul_mat_vec_q<"):
        return "mul_mat_vec_q<" + n.split("(ggml_type)")[1].split(",")[0] + ">"
    return base.split("<")[0].strip()

# median per-call time x calls per token, so one stalled call cannot skew a kernel's share
durs, cnt = collections.defaultdict(list), collections.Counter()
for a in agents:
    tail = rows[a][-ntok * per_tok[a]:]
    for s, e, n in tail:
        durs[short(n)].append((e - s) / 1e6)
        cnt[short(n)] += 1 / ntok
tot = collections.Counter({k: st.median(v) * cnt[k] for k, v in durs.items()})
S = sum(tot.values())
print(f"{ntok} tokens; kernels/token {sum(per_tok.values())} ({', '.join(f'{a}: {n}' for a, n in per_tok.items())}); kernel time {S:.2f} ms/token")
print(f"{'ms/tok':>7} {'share':>6} {'calls':>6} {'us/call':>8}  kernel")
for k, t in tot.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 30):
    print(f"{t:7.3f} {100*t/S:5.1f}% {cnt[k]:6.0f} {1e3*t/cnt[k]:8.2f}  {k[:80]}")
