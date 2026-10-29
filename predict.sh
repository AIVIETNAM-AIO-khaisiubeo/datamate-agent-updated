#!/usr/bin/env bash
# Competition entry point: reads the test split from /data, writes /result/submission.jsonl
# and per-question logs to /result/logs/. Runs fully offline (Rule 3).
#
#   DATA_DIR   (default /data)          split directory: questions.jsonl, rulebook.md, calendar.csv, data/<biz>/
#   RESULT_DIR (default /result)
#   MODEL_DIR  (default /models/Qwen3-14B)   local weights, served by vLLM on both GPUs (TP=2)
#   NO_LLM=1   answer with templates + rules only (no GPU needed)
set -euo pipefail
DATA_DIR=${DATA_DIR:-/data}
RESULT_DIR=${RESULT_DIR:-/result}
MODEL_DIR=${MODEL_DIR:-/models/Qwen3-14B}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-14B}
PORT=${PORT:-8000}
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$RESULT_DIR/logs"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
CACHE_DIR=${CACHE_DIR:-/tmp/datamate_cache}

EXTRA=()
if [[ "${NO_LLM:-0}" == "1" ]]; then
  EXTRA+=(--set llm.backend=none)
else
  # deterministic serving: fixed seed, no async scheduling surprises; prefix caching for the shared prompt prefix
  python -m vllm.entrypoints.openai.api_server --model "$MODEL_DIR" --served-model-name "$MODEL_NAME" \
      --tensor-parallel-size 2 --port "$PORT" --enable-prefix-caching --max-model-len 32768 \
      --gpu-memory-utilization 0.90 --seed 0 > "$RESULT_DIR/logs/vllm.log" 2>&1 &
  VLLM_PID=$!
  trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
  for i in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then break; fi
    sleep 2
  done
  EXTRA+=(--set "llm.base_url=http://127.0.0.1:$PORT/v1" --set "llm.main_model=$MODEL_NAME")
fi

python "$HERE/scripts/run.py" --data "$DATA_DIR" --out "$RESULT_DIR/submission.jsonl" \
    --logs "$RESULT_DIR/logs" --set "data.cache_dir=$CACHE_DIR" "${EXTRA[@]}"
