#!/usr/bin/env bash
# Serve the main generator (and optionally a second, SQL-specialised model) with vLLM.
# Automatic prefix caching matters: every prompt starts with [contract+rulebook][business card].
set -euo pipefail
MAIN=${MAIN:-Qwen/Qwen3-30B-A3B-Instruct-2507}
SQLM=${SQLM:-}                       # e.g. seeklhy/OmniSQL-7B or Snowflake/Arctic-Text2SQL-R1-7B
GPU_UTIL=${GPU_UTIL:-0.85}

vllm serve "$MAIN" --port 8000 --enable-prefix-caching --max-model-len 16384 \
  --gpu-memory-utilization "$GPU_UTIL" --served-model-name "$MAIN" &
if [[ -n "$SQLM" ]]; then
  vllm serve "$SQLM" --port 8001 --enable-prefix-caching --max-model-len 16384 \
    --gpu-memory-utilization 0.10 --served-model-name "$SQLM" &
fi
wait
