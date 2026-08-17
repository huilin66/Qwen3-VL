#!/usr/bin/env bash

# ============================================================
# Qwen3.8-27B-FP8
# Single RTX 6000 Ada 48GB
# Text + Image + Reasoning + Tool Calling
# ============================================================

# conda activate 前不要开启 -u，
# 否则 conda hook 可能因为未定义变量报错。
set -eo pipefail


# ============================================================
# Conda 环境
# ============================================================

source /apps/miniconda3/etc/profile.d/conda.sh
conda activate qwen

# Conda 激活完成后再开启 nounset
set -u


# ============================================================
# CUDA 配置
# ============================================================

# 当前 qwen 环境中已安装 CUDA 12.8 nvcc
export CUDA_HOME="${CONDA_PREFIX}"

# 保证优先使用 Conda CUDA 12.8
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# 显式指定 CUDA 编译器
export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
export CUDA_NVCC_EXECUTABLE="${CONDA_PREFIX}/bin/nvcc"

# CUDA runtime / Conda libraries
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"


# ============================================================
# GPU 配置
# ============================================================

# 使用物理 GPU 1
export CUDA_VISIBLE_DEVICES=1

export TOKENIZERS_PARALLELISM=false


# ============================================================
# 服务配置
# ============================================================

MODEL="Qwen/Qwen3.8-27B-FP8"

SERVED_MODEL_NAME="qwen3.8-27b-fp8"

PORT=8010

# 允许读取本地图片
MEDIA_ROOT="/scrinvme/huilin"


# ============================================================
# 环境检查
# ============================================================

echo "============================================================"
echo "Qwen3.8-27B-FP8 Deployment"
echo "============================================================"

echo "CONDA_PREFIX : ${CONDA_PREFIX}"
echo "CUDA_HOME    : ${CUDA_HOME}"
echo "NVCC         : $(which nvcc)"
echo "GPU          : ${CUDA_VISIBLE_DEVICES}"
echo "MODEL        : ${MODEL}"
echo "PORT         : ${PORT}"
echo "MEDIA_ROOT   : ${MEDIA_ROOT}"

echo "============================================================"

nvcc --version

echo "============================================================"


# ============================================================
# 检查 nvcc
# ============================================================

NVCC_PATH="$(which nvcc)"

EXPECTED_NVCC="${CONDA_PREFIX}/bin/nvcc"

if [[ "${NVCC_PATH}" != "${EXPECTED_NVCC}" ]]; then
    echo "ERROR: Wrong nvcc detected."
    echo
    echo "Current:"
    echo "  ${NVCC_PATH}"
    echo
    echo "Expected:"
    echo "  ${EXPECTED_NVCC}"
    echo
    exit 1
fi


# ============================================================
# 启动 vLLM
# ============================================================

exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --attention-backend TRITON_ATTN \
  --enforce-eager \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --allowed-local-media-path "${MEDIA_ROOT}" \
  --seed 0 \
  --disable-log-stats \
  --api-key "qwenL666666qB9pM8yVZx2Kf7R4hN1uCe6Wsd3AjT0mGkPqX"