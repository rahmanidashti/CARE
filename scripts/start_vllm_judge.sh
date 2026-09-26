#!/usr/bin/env bash
# Serve the frozen model that writes and grades rubrics.
#
# Every stage of the pipeline -- pseudo-reference, near-miss, contrastive criteria,
# rubric, judge -- hits this one endpoint. It holds a frozen copy of the policy's
# INITIAL weights and is never updated: start it from the same checkpoint you are
# about to train, and leave it alone for the whole run. Nothing else is consulted.
#
# Usage:
#   bash scripts/start_vllm_judge.sh                          # Qwen3-4B, thinking off
#   JUDGE_MODEL=Qwen/Qwen3-8B bash scripts/start_vllm_judge.sh
#   JUDGE_MODEL=google/gemma-3-4b-it NO_THINK=0 bash scripts/start_vllm_judge.sh
#
# Then point training at it:  VLLM_BASE_URL=http://localhost:8061/v1
set -euo pipefail

JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3-4B}   # HF hub id or a local path
SERVED_NAME=${SERVED_NAME:-$(basename "$JUDGE_MODEL")}
PORT=${VLLM_PORT:-8061}
GPU_MEM=${JUDGE_GPU_MEMORY_UTILIZATION:-0.6}
NO_THINK=${NO_THINK:-1}                     # 1 for Qwen3, which thinks by default

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=${JUDGE_CUDA_VISIBLE_DEVICES:-1}

ARGS=(--model "$JUDGE_MODEL"
      --host 0.0.0.0
      --port "$PORT"
      --served-model-name "$SERVED_NAME"
      --gpu-memory-utilization "$GPU_MEM")

if [ "$NO_THINK" = "1" ]; then
  # Qwen3 emits a reasoning block by default. Suppress it with the bundled template,
  # which is Qwen3's own with `enable_thinking = false` prepended.
  #
  # Do NOT also pass --reasoning-parser. The two together make vLLM classify the
  # whole response as reasoning and return content=None on every call, which shows
  # up downstream as every rubric silently failing to parse.
  ARGS+=(--enable-auto-tool-choice --tool-call-parser hermes
         --chat-template scripts/qwen3_nothink.jinja)
fi

echo "[judge] serving ${JUDGE_MODEL} as '${SERVED_NAME}' on :${PORT} (no_think=${NO_THINK})"
exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
