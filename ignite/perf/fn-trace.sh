#!/usr/bin/env bash
# rocprofv3 kernel trace of a graphed Flash-Next decode (d0, 32 tokens), then the kernel inventory and mmvq bandwidth.
# usage: fn-trace.sh <build dir> [ENV=value ...]
# Profiling slows the run; kernel durations stay valid, gaps do not (take gaps from fn-bench.sh token time).
set -euo pipefail
B=$1; shift
R=${ROCM_PATH:-$HOME/rocm-therock/10.2.0a20261011}
PY=${FN_PY:-python3}
MODEL=${FN_MODEL:-$HOME/llm/Qwen3.8-Flash-Next-REAP-288-FN-R1.gguf}
HERE=$(cd "$(dirname "$0")" && pwd)
O=$HOME/llm/logs/rocprof-$(date +%Y%m%d-%H%M%S)
env "$@" timeout 1200 "$R/bin/rocprofv3" --kernel-trace -f csv -d "$O" -o trace -- "$B/bin/llama-bench" \
    -m "$MODEL" -lm dio -dev ROCm0/ROCm1 -sm "${FN_SM:-layer}" -ts 530/470 -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
    -b 2048 -ub 256 -t 8 -p 0 -n 32 -d 0 -r 1 > "$O.log" 2>&1
grep tg32 "$O.log" | sed -E 's/\| +qwen4exp.*\| +dio \|/|/'
"$PY" -I "$HERE/kinventory.py" "$O/trace_kernel_trace.csv" 25 | tee "$O.inventory.txt"
"$PY" -I "$HERE/mmvq_bw.py" "$O/trace_kernel_trace.csv" "$MODEL" | tee "$O.mmvq.txt"
echo "-> $O"
