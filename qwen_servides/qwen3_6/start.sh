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

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# ============================================================
# 服务配置
# ============================================================

MODEL="Qwen/Qwen3.6-27B-FP8"
PORT=8002

vllm serve "${MODEL}" \
  --served-model-name qwen3.6 \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --seed 0 \
  --disable-log-stats \
  --api-key "qwenL666666qB9pM8yVZx2Kf7R4hN1uCe6Wsd3AjT0mGkPqX"