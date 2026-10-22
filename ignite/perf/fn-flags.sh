#!/usr/bin/env bash
# HIP/CLR runtime flag sweep for Flash-Next decode at d0 (is launch latency tunable on Linux?). Baseline is repeated
# at start, middle and end to show run-to-run noise.
# usage: fn-flags.sh <build>
set -uo pipefail
B=$1
MODEL=${FN_MODEL:-$HOME/llm/Qwen3.8-Flash-Next-REAP-288-FN-R1.gguf}
LOG=$HOME/llm/logs/fn-flags-$(date +%Y%m%d-%H%M%S).txt
ARGS=(-m "$MODEL" -lm dio -dev ROCm0/ROCm1 -sm layer -ts 530/470 -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 256 -t 8
      -p 0 -n 128 -d 0 -r 3)
RUNS=(base HIP_FORCE_DEV_KERNARG=1 HIP_FORCE_DEV_KERNARG=0 DEBUG_CLR_GRAPH_PACKET_CAPTURE=1 DEBUG_CLR_GRAPH_PACKET_CAPTURE=0
      AMD_DIRECT_DISPATCH=1 AMD_DIRECT_DISPATCH=0 base GPU_MAX_HW_QUEUES=1 GPU_MAX_HW_QUEUES=2 ROC_ACTIVE_WAIT_TIMEOUT=0
      DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1 DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0 DEBUG_HIP_GRAPH_BATCH_SIZE=1024
      HSA_ENABLE_SDMA=0 "HIP_FORCE_DEV_KERNARG=1;DEBUG_CLR_GRAPH_PACKET_CAPTURE=1" base)
for r in "${RUNS[@]}"; do
    kv=(); [ "$r" != base ] && IFS=';' read -ra kv <<< "$r"
    out=$(env "${kv[@]}" timeout 900 "$B/bin/llama-bench" "${ARGS[@]}" 2>&1)
    t=$(grep -oE 'tg128 \| +[0-9.]+ ± [0-9.]+' <<< "$out" | sed -E 's/tg128 \| +//')
    printf '%-58s %s\n' "$r" "${t:-FAILED: $(grep -m1 -iE 'error|abort' <<< "$out")}" | tee -a "$LOG"
done
echo "-> $LOG"
