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

# 可选：强制只使用本地缓存
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ============================================================
# 模型与 LoRA 路径
# ============================================================

# 必须是训练 LoRA 时使用的 Qwen3-VL-4B 基座
BASE_MODEL="/home/23039356r/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"

# 必须直接指向包含 adapter_config.json 的 checkpoint
CROP_LORA="/localnvme/project/Qwen3-VL/qwen_servides/qwen3vl_yolo_finetune/output/qwen3vl_crop_cls_lora/v1-20260722-201719/checkpoint-1389"
FULL_LORA="/localnvme/project/Qwen3-VL/qwen_servides/qwen3vl_yolo_finetune/output/qwen3vl_full_det_lora/v0-20260722-202209/checkpoint-675"

MEDIA_ROOT="/scrinvme/huilin/traffic_sign/defect/detection/data_det_3_damaged_traffic_signs"

PORT=8010
API_KEY="qwenvlloraL666666qB9pM8yVZx2Kf7R4hN1uCe6Wsd3AjT0mGkPqX"

# 两个 LoRA 中最大的 rank
MAX_LORA_RANK=16

# ============================================================
# 路径检查
# ============================================================

for path in \
    "${BASE_MODEL}/config.json" \
    "${CROP_LORA}/adapter_config.json" \
    "${CROP_LORA}/adapter_model.safetensors" \
    "${FULL_LORA}/adapter_config.json" \
    "${FULL_LORA}/adapter_model.safetensors"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing file: ${path}" >&2
        exit 1
    fi
done

# ============================================================
# 启动服务
# ============================================================

exec vllm serve "${BASE_MODEL}" \
  --served-model-name qwen3-vl-4b-instruct \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --gpu-memory-utilization 0.85 \
  --enable-lora \
  --lora-modules \
    "{\"name\":\"qwen3-vl-4b-crop\",\"path\":\"${CROP_LORA}\",\"base_model_name\":\"${BASE_MODEL}\"}" \
    "{\"name\":\"qwen3-vl-4b-full\",\"path\":\"${FULL_LORA}\",\"base_model_name\":\"${BASE_MODEL}\"}" \
  --max-loras 2 \
  --max-cpu-loras 2 \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --seed 0 \
  --disable-log-stats \
  --allowed-local-media-path "${MEDIA_ROOT}" \
  --api-key "${API_KEY}"