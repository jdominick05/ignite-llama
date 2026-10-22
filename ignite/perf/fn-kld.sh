#!/usr/bin/env bash
# KLD of a Flash-Next candidate (requant or graph change) against FN-R1:
# eval2.txt, -c 1024, 24 chunks (~12k scored tokens, base file ~6 GB). Gate: Mean KLD < 0.02 and Same top p >= 97%.
# usage: fn-kld.sh base <build>                      -> writes the base logits from FN-R1
#        fn-kld.sh cmp  <build> [model.gguf] [ENV=..] -> compares (model defaults to FN-R1, for graph/kernel changes)
#        fn-kld.sh clean                             -> deletes the base file
# The base file is derived from chat text: local only, never in the repo; delete it when the phase is done.
set -euo pipefail
FN=${FN_MODEL:-$HOME/llm/Qwen3.8-Flash-Next-REAP-288-FN-R1.gguf}
TEXT=${FN_KLD_TEXT:?set FN_KLD_TEXT to a plain-text eval file}
BASE=$HOME/llm/kld-base-fn-r1.bin
mode=$1; shift
[ "$mode" = clean ] && { rm -f "$BASE"; echo "removed $BASE"; exit 0; }
B=$1; shift
ARGS=(-f "$TEXT" -c 1024 --chunks 24 -b 1024 -ub 256 -dev ROCm0,ROCm1 -sm layer -ngl 99 -ts 530,470 -fa on -t 8 --load-mode dio)
LOG=$HOME/llm/logs/fn-kld-$mode-$(date +%Y%m%d-%H%M%S).log
if [ "$mode" = base ]; then
    timeout 3600 "$B/bin/llama-perplexity" -m "$FN" "${ARGS[@]}" --kl-divergence-base "$BASE" > "$LOG" 2>&1
    grep -E 'Final estimate|error' "$LOG"; ls -la "$BASE"
else
    M=$FN; if [ $# -gt 0 ] && [[ "$1" == *.gguf ]]; then M=$1; shift; fi
    env "$@" timeout 3600 "$B/bin/llama-perplexity" -m "$M" "${ARGS[@]}" --kl-divergence-base "$BASE" --kl-divergence > "$LOG" 2>&1
    grep -aE 'Mean +KLD|99\.0% +KLD|Maximum KLD|Same top p|Mean +PPL\(Q\)|Mean +PPL\(base\)|error' "$LOG"
fi
echo "-> $LOG"
