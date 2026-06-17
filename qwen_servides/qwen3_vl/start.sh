#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Conda 环境
# ============================================================

source /apps/miniconda3/etc/profile.d/conda.sh
conda activate qwen

# ============================================================
# GPU 配置
# ============================================================

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

# ============================================================
# 服务配置
# ============================================================

MODEL="Qwen/Qwen3-VL-8B-Instruct"
MEDIA_ROOT="/scrinvme/huilin/exchange/qwen3_vl"
PORT=8001

vllm serve "${MODEL}" \
  --served-model-name qwen3-vl-8b \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --gpu-memory-utilization 0.90 \
  --seed 0 \
  --disable-log-stats \
  --allowed-local-media-path "${MEDIA_ROOT}" \
  --api-key "qwenvlL666666qB9pM8yVZx2Kf7R4hN1uCe6Wsd3AjT0mGkPqX"