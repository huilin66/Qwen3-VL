# Qwen3-VL + YOLO LoRA fine-tuning

This bundle converts a standard YOLO detection or segmentation dataset into two Qwen3-VL SFT tasks:

1. **Crop classification**: one cropped target per sample; answer is exactly one class name.
2. **Full-image detection**: one full image per sample; answer is a JSON list containing labels and bounding boxes.

## Expected YOLO layout

```text
dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

`names` in `data.yaml` can be a list or dict. YOLO segmentation polygons are supported and are converted to enclosing bounding boxes.

## Install

```bash
pip install -r requirements.txt
```

## Crop classification

```bash
python convert_crop_classification.py \
  --data /data/my_yolo/data.yaml \
  --output /data/qwen_crop \
  --splits train val \
  --padding 0.10 \
  --square
```

Optional background crops for false-positive rejection:

```bash
python convert_crop_classification.py \
  --data /data/my_yolo/data.yaml \
  --output /data/qwen_crop_with_bg \
  --splits train val \
  --padding 0.10 \
  --square \
  --background-per-image 1
```

Without background samples, this is a closed-set classifier and it will tend to always select a known class.

Train:

```bash
MODEL_PATH=/models/Qwen3-VL-8B-Instruct \
TRAIN_JSONL=/data/qwen_crop/train.jsonl \
VAL_JSONL=/data/qwen_crop/val.jsonl \
CUDA_VISIBLE_DEVICES=0,1 \
bash train_crop_classification.sh
```

## Full-image detection

```bash
python convert_full_detection.py \
  --data /data/my_yolo/data.yaml \
  --output /data/qwen_full \
  --splits train val
```

The generated `objects.bbox` values are absolute source-image `xyxy` coordinates. The training script sets `QWENVL_BBOX_FORMAT=new`, allowing ms-swift to convert them to the Qwen3-VL normalized-1000 representation.

Train:

```bash
MODEL_PATH=/models/Qwen3-VL-8B-Instruct \
TRAIN_JSONL=/data/qwen_full/train.jsonl \
VAL_JSONL=/data/qwen_full/val.jsonl \
CUDA_VISIBLE_DEVICES=0,1 \
bash train_full_detection.sh
```

## Useful overrides

```bash
# More visual tokens for small objects; increases VRAM usage
IMAGE_TOKENS=1600 bash train_full_detection.sh

# Change LoRA or batch settings
LORA_RANK=32 LORA_ALPHA=64 GRAD_ACC=4 bash train_full_detection.sh

# Local base model
MODEL_PATH=/your/local/Qwen3-VL-8B-Instruct bash train_full_detection.sh
```

Both initial scripts freeze ViT and aligner. This provides a stable language-side LoRA baseline. If the full-image model learns class names and JSON formatting but bbox quality remains poor, test visual adaptation as a separate second-stage experiment rather than changing several variables at once.
