#!/usr/bin/env bash
set -euo pipefail

# Example:
# MODEL_PATH=/models/Qwen3-VL-8B-Instruct \
# TRAIN_JSONL=/data/qwen_full/train.jsonl \
# VAL_JSONL=/data/qwen_full/val.jsonl \
# CUDA_VISIBLE_DEVICES=0,1 \
# bash train_full_detection.sh

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"

MODEL_PATH="${MODEL_PATH:-/home/23039356r/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17}"
TRAIN_JSONL="${TRAIN_JSONL:-/scrinvme/huilin/traffic_sign/defect/detection/data_det_3_damaged_traffic_signs/qwen_full/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-/scrinvme/huilin/traffic_sign/defect/detection/data_det_3_damaged_traffic_signs/qwen_full/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen3vl_full_det_lora}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-5}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
GRAD_ACC="${GRAD_ACC:-8}"
IMAGE_TOKENS="${IMAGE_TOKENS:-1024}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
SAVE_STEPS="${SAVE_STEPS:-200}"
EVAL_STEPS="${EVAL_STEPS:-200}"

export CUDA_VISIBLE_DEVICES
export IMAGE_MAX_TOKEN_NUM="${IMAGE_TOKENS}"
export QWENVL_BBOX_FORMAT="${QWENVL_BBOX_FORMAT:-new}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"

EXTRA_ARGS=()
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  EXTRA_ARGS+=(--deepspeed zero2)
fi

exec swift sft \
  --model "${MODEL_PATH}" \
  --dataset "${TRAIN_JSONL}" \
  --val_dataset "${VAL_JSONL}" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs "${EPOCHS}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate "${LR}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 3 \
  --logging_steps 10 \
  --max_length "${MAX_LENGTH}" \
  --warmup_ratio 0.05 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --output_dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
