#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Conda 环境
# ============================================================

# source /apps/miniconda3/etc/profile.d/conda.sh
# conda activate qwen

# ============================================================
# GPU 配置
# ============================================================

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# ============================================================
# 模型配置
# ============================================================

declare -A MODEL_PATH
declare -A MODEL_NAME
declare -A MODEL_DTYPE
declare -A GPU_UTIL
declare -A MAX_LEN

MODEL_PATH[4b]="Qwen/Qwen3-VL-4B-Instruct"
MODEL_NAME[4b]="qwen3-vl-4b"
MODEL_DTYPE[4b]="bfloat16"
GPU_UTIL[4b]="0.25"
MAX_LEN[4b]="4096"

MODEL_PATH[8b]="Qwen/Qwen3-VL-8B-Instruct"
MODEL_NAME[8b]="qwen3-vl-8b"
MODEL_DTYPE[8b]="bfloat16"
GPU_UTIL[8b]="0.35"
MAX_LEN[8b]="8192"

MODEL_PATH[30b]="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
MODEL_NAME[30b]="qwen3-vl-30b"
MODEL_DTYPE[30b]="auto"
GPU_UTIL[30b]="0.50"
MAX_LEN[30b]="8192"

# ============================================================
# 选择模型
# ============================================================

MODEL_KEY="${1:-8b}"

if [[ -z "${MODEL_PATH[$MODEL_KEY]+x}" ]]; then
    echo "Unknown model key: ${MODEL_KEY}"
    echo "Available: 4b 8b 30b"
    exit 1
fi

MODEL="${MODEL_PATH[$MODEL_KEY]}"
SERVED_NAME="${MODEL_NAME[$MODEL_KEY]}"
DTYPE="${MODEL_DTYPE[$MODEL_KEY]}"
MEM_UTIL="${GPU_UTIL[$MODEL_KEY]}"
MODEL_LEN="${MAX_LEN[$MODEL_KEY]}"

PORT=8001

echo "========================================"
echo "Model key:      ${MODEL_KEY}"
echo "Model:          ${MODEL}"
echo "Served name:    ${SERVED_NAME}"
echo "dtype:          ${DTYPE}"
echo "max model len:  ${MODEL_LEN}"
echo "GPU util:       ${MEM_UTIL}"
echo "Port:           ${PORT}"
echo "========================================"

vllm serve "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --dtype "${DTYPE}" \
  --max-model-len "${MODEL_LEN}" \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --gpu-memory-utilization "${MEM_UTIL}" \
  --seed 0 \
  --disable-log-stats \
  --api-key "qwenvlL666666qB9pM8yVZx2Kf7R4hN1uCe6Wsd3AjT0mGkPqX"