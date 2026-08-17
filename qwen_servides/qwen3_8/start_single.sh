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

MODEL="Qwen/Qwen3.8-27B-FP8"
MEDIA_ROOT="/scrinvme/huilin"
PORT=8010

vllm serve "${MODEL}" \
  --served-model-name qwen3.8-27b-fp8 \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --allowed-local-media-path "${MEDIA_ROOT}" \
  --seed 0 \
  --disable-log-stats \
  --api-key "qwenL666666qB9pM8yVZx2Kf7R4hN1uCe6Wsd3AjT0mGkPqX"