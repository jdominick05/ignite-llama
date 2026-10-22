#!/usr/bin/env bash
# Standard Flash-Next decode A/B with fixed flags, d0 + d64k, alternating A/B/A/B.
# usage: fn-bench.sh <build>                          one plain run
#        fn-bench.sh <build> KNOB=off KNOB=on         one binary, env knob alternated
#        fn-bench.sh <build A> <build B>              two builds alternated (graph / model changes)
#   e.g. fn-bench.sh build-hip-therock GGML_CUDA_MMVQ_SMALL_K=0 GGML_CUDA_MMVQ_SMALL_K=1
# env: FN_MODEL (default ~/llm/Qwen3.8-Flash-Next-REAP-288-FN-R1.gguf), FN_DEPTHS (0,65536), FN_SM (layer), FN_EXTRA
# Adopt a change only if both B runs beat both A runs.
set -euo pipefail
MODEL=${FN_MODEL:-$HOME/llm/Qwen3.8-Flash-Next-REAP-288-FN-R1.gguf}
LOG=$HOME/llm/logs/fn-bench-$(date +%Y%m%d-%H%M%S).txt
ARGS=(-m "$MODEL" -lm dio -dev ROCm0/ROCm1 -sm "${FN_SM:-layer}" -ts 530/470 -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0
      -b 2048 -ub 256 -t 8 -p 0 -n 128 -d "${FN_DEPTHS:-0,65536}" -r 4 ${FN_EXTRA:-})
rev() { echo "$(git -C "$1/.." rev-parse --short HEAD 2>/dev/null)$(git -C "$1/.." diff HEAD --quiet 2>/dev/null || echo +dirty)"; }
# the first repetition after loading runs slow (first-use warm-up, up to -10% at d0): report reps 2..N, show rep 1 apart
run() { # <build> [ENV=value]
    echo "== $1 ${2:-} $(rev "$1")"
    env ${2:-} timeout 3600 "$1/bin/llama-bench" "${ARGS[@]}" -o jsonl 2>&1 | python3 -c '
import json, sys, statistics as st
for line in sys.stdin:
    if not line.startswith("{"):
        if "error" in line: print(line.rstrip())
        continue
    r = json.loads(line); t = r["samples_ts"]; rest = t[1:] or t
    d = r.get("n_depth", 0); name = "tg%d" % r["n_gen"] + (" @ d%d" % d if d else "")
    sd = st.stdev(rest) if len(rest) > 1 else 0.0
    print("| %15s | %12.2f ± %.2f |  (rep 1: %.2f)" % (name, st.mean(rest), sd, t[0]))
'
}
{
    echo "# $(date -Is) ${ARGS[*]}"
    if [ $# -eq 3 ]; then for kv in "$2" "$3" "$2" "$3"; do run "$1" "$kv"; done
    elif [ $# -eq 2 ]; then for b in "$1" "$2" "$1" "$2"; do run "$b"; done
    else run "$1"; fi
} | tee "$LOG"
echo "-> $LOG"
