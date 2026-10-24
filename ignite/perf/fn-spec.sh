#!/usr/bin/env bash
# Delivered decode on real agent requests: llama-server (Flash-Next profile flags) + serve_bench.py, per config.
# usage: fn-spec.sh <build> "<label>|<ENV=v ...>|<server args>" ...
#   e.g. fn-spec.sh build-hip-therock "none||--spec-type none" "mod||--spec-type ngram-mod"
# serve_bench prints aggregate tok/s only (the replay prompts are private token ids and are never printed).
set -uo pipefail
B=$1; shift
MODEL=${FN_MODEL:-$HOME/llm/Qwen3.8-Flash-Next-REAP-288-FN-R1.gguf}
SB=${FN_SERVE_BENCH:?set FN_SERVE_BENCH to a request-replay script (not included: it replays private requests)}
PORT=8092
LOG=$HOME/llm/logs/fn-spec-$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOG"
for cfg in "$@"; do
    IFS='|' read -r label envs sargs <<< "$cfg"
    read -ra kv <<< "$envs"; read -ra extra <<< "$sargs"
    env "${kv[@]}" "$B/bin/llama-server" -m "$MODEL" --load-mode dio -dev ROCm0,ROCm1 -sm layer -ts ${FN_TS:-530,470} -ngl 99 \
        -c 32768 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 256 -t 8 --cache-ram 0 --poll 0 \
        --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --port $PORT "${extra[@]}" > "$LOG/server-$label.log" 2>&1 &
    pid=$!
    for _ in $(seq 300); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 2; done
    ${FN_PY:-python3} -I "$SB" "$label" --port $PORT --out "$LOG/serve-bench.jsonl" 2>&1 | tee -a "$LOG/summary.txt"
    kill $pid; wait $pid 2> /dev/null
done
echo "-> $LOG"
