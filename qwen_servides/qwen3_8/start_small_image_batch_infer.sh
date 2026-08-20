#!/usr/bin/env bash

set -eo pipefail

# ============================================================
# Conda
# ============================================================

source /apps/miniconda3/etc/profile.d/conda.sh
conda activate qwen38

set -u


# ============================================================
# GPU / Runtime
# ============================================================

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

# 你的系统 nvcc 太旧，避免 FlashInfer sampler JIT
export VLLM_USE_FLASHINFER_SAMPLER=0

# 清理之前安装时留下的变量
unset VLLM_VERSION || true
unset CUDA_VERSION || true


# ============================================================
# Model
# ============================================================

MODEL="Qwen/Qwen3.8-27B-FP8"
SERVED_MODEL_NAME="qwen3.8-27b-fp8"

PORT=8010
MEDIA_ROOT="/scrinvme/huilin"


# ============================================================
# Check
# ============================================================

echo "============================================================"
echo "Qwen3.8-27B-FP8 - Defect Classification"
echo "============================================================"
echo "CONDA_PREFIX : ${CONDA_PREFIX}"
echo "Python       : $(which python)"
echo "vLLM         : $(which vllm)"
echo "GPU          : ${CUDA_VISIBLE_DEVICES}"
echo "MODEL        : ${MODEL}"
echo "PORT         : ${PORT}"
echo "============================================================"

python - <<'PY'
import torch
import vllm

print("torch      :", torch.__version__)
print("torch CUDA :", torch.version.cuda)
print("vLLM       :", vllm.__version__)
print("CUDA       :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU        :", torch.cuda.get_device_name())
PY

echo "============================================================"


# ============================================================
# Start
# ============================================================

exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --dtype auto \
  --max-model-len 2048 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"min_pixels":65536,"max_pixels":200704}' \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --allowed-local-media-path "${MEDIA_ROOT}" \
  --generation-config vllm \
  --seed 0 \
  --disable-log-stats \
  --api-key "qwenL666666qB9pM8yVZx2Kf7R4hN1uCe6Wsd3AjT0mGkPqX"