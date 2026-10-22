#!/usr/bin/env python3
# Where the idle time goes per decode token, per GPU (each card on its own clock).
# usage: gapsplit.py <kernel_trace.csv> [long_gap_us=100]
# Per card: busy (union of kernel intervals), short gaps (< long_gap_us, dispatch overhead) and long gaps (waits on
# the other card or the host). Long gaps are listed by the kernels on both sides.
import csv, sys, collections, statistics as st

LONG = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
rows = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    rows[r["Agent_Id"]].append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))

def short(n):
    n = n.replace("void ", "")
    if n.startswith("mul_mat_vec_q<"):
        return "mmvq<" + n.split("(ggml_type)")[1].split(",")[0] + ">"
    return n.split("(")[0].split("<")[0].strip()[:28]

for a in sorted(rows):
    ks = sorted(rows[a])
    if len(ks) < 100:
        continue
    # decode region: drop the first 20% (load, warmup) of the timeline
    ks = ks[len(ks) // 5:]
    t0, t1 = ks[0][0], ks[-1][1]
    busy = 0; end = ks[0][0]; short_gap = 0; long_gaps = collections.Counter(); long_cnt = collections.Counter()
    prev = None
    for s, e, n in ks:
        if s > end:
            g = (s - end) / 1e3
            if g < LONG:
                short_gap += g
            else:
                key = f"{short(prev)} -> {short(n)}"
                long_gaps[key] += g; long_cnt[key] += 1
        s = max(s, end); busy += max(0, e - s) / 1e3; end = max(end, e); prev = n
    total = (t1 - t0) / 1e3
    lg = sum(long_gaps.values())
    print(f"{a}: window {total/1e3:.1f} ms, {len(ks)} kernels | busy {100*busy/total:.1f}%, short gaps {100*short_gap/total:.1f}% "
          f"(avg {1e3*short_gap/len(ks):.2f} us/kernel... {short_gap/len(ks):.2f} us), long gaps {100*lg/total:.1f}%")
    for k, g in long_gaps.most_common(8):
        print(f"    {100*g/total:5.1f}%  {long_cnt[k]:5d}x  avg {g/long_cnt[k]:8.1f} us  {k}")
