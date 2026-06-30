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

MODEL="Qwen/Qwen3.5-9B"
PORT=8002

vllm serve "${MODEL}" \
  --served-model-name qwen3.5-9b \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.55 \
  --language-model-only \
  --generation-config vllm \
  --seed 0 \
  --disable-log-stats \
  --api-key "qwenL666666qB9pM8yVZx2Kf7R4hN1uCe6Wsd3AjT0mGkPqX"