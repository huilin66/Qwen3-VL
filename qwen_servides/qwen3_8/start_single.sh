#!/usr/bin/env bash

set -eo pipefail

# ============================================================
# Conda
# ============================================================

source /apps/miniconda3/etc/profile.d/conda.sh
conda activate qwen38

set -u


# ============================================================
# GPU
# ============================================================

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

# ============================================================
# Model
# ============================================================

MODEL="Qwen/Qwen3.8-27B-FP8"
SERVED_MODEL_NAME="qwen3.8-27b-fp8"

PORT=8010
MEDIA_ROOT="/scrinvme/huilin"


# ============================================================
# Environment
# ============================================================

echo "============================================================"
echo "Qwen3.8-27B-FP8 / vLLM 0.26.0"
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

print("torch       :", torch.__version__)
print("torch CUDA  :", torch.version.cuda)
print("vLLM        :", vllm.__version__)
print("CUDA        :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name())
    print("Capability  :", torch.cuda.get_device_capability())
PY

echo "============================================================"


# ============================================================
# Start server
# ============================================================

exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --allowed-local-media-path "${MEDIA_ROOT}" \
  --generation-config vllm \
  --seed 0 \
  --disable-log-stats \
  --api-key ""