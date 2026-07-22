#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from yolo_common import (
    collect_images,
    label_path_for_image,
    load_dataset_config,
    parse_yolo_label,
    stable_image_id,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert YOLO detection/segmentation data to cropped Qwen3-VL classification SFT data."
    )
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--padding", type=float, default=0.10)
    p.add_argument("--square", action="store_true")
    p.add_argument("--min-side", type=int, default=8)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--background-per-image", type=int, default=0)
    p.add_argument("--background-label", default="background")
    p.add_argument("--background-max-iou", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-invalid", action="store_true")
    return p.parse_args()


def expand_box(box, width: int, height: int, padding: float, square: bool):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    tw, th = bw * (1 + 2 * padding), bh * (1 + 2 * padding)
    if square:
        tw = th = max(tw, th)
    return (
        max(0, int(round(cx - tw / 2))),
        max(0, int(round(cy - th / 2))),
        min(width, int(round(cx + tw / 2))),
        min(height, int(round(cy + th / 2))),
    )


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def sample_background_box(rng, width, height, gt_boxes, max_iou):
    min_dim = min(width, height)
    for _ in range(100):
        scale = rng.uniform(0.12, 0.40)
        aspect = rng.uniform(0.75, 1.33)
        cw = max(16, min(width, int(min_dim * scale * aspect**0.5)))
        ch = max(16, min(height, int(min_dim * scale / aspect**0.5)))
        x1 = rng.randint(0, max(0, width - cw))
        y1 = rng.randint(0, max(0, height - ch))
        box = (x1, y1, x1 + cw, y1 + ch)
        if all(iou_xyxy(box, gt) <= max_iou for gt in gt_boxes):
            return box
    return None


def sample(image_path: Path, prompt: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"<image>{prompt}"},
            {"role": "assistant", "content": answer},
        ],
        "images": [str(image_path.resolve())],
    }


def main() -> None:
    args = parse_args()
    if args.padding < 0:
        raise ValueError("--padding must be >= 0")

    cfg = load_dataset_config(args.data)
    out_root = Path(args.output).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    labels = list(cfg.names)
    if args.background_per_image > 0:
        if args.background_label in labels:
            raise ValueError(
                f'Background label "{args.background_label}" already exists'
            )
        labels.append(args.background_label)

    labels_json = json.dumps(labels, ensure_ascii=False)
    prompt = (
        "Classify the main target in this cropped image. "
        f"Choose exactly one label from {labels_json}. "
        "Return only the label, without explanation or JSON."
    )
    rng = random.Random(args.seed)
    all_counts = Counter()

    for split in args.splits:
        source = cfg.split_sources.get(split)
        if source is None:
            print(f"[WARN] split '{split}' is absent; skipped")
            continue

        images = collect_images(source, cfg.dataset_root)
        crop_root = out_root / "crops" / split
        crop_root.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_root / f"{split}.jsonl"
        counts, written, skipped = Counter(), 0, 0

        with jsonl_path.open("w", encoding="utf-8") as dst:
            for image_path in tqdm(images, desc=f"Converting {split}", unit="image"):
                objects = parse_yolo_label(
                    label_path_for_image(image_path, cfg.dataset_root),
                    cfg.names,
                    strict=not args.skip_invalid,
                )
                with Image.open(image_path) as raw:
                    image = raw.convert("RGB")
                    width, height = image.size
                    image_id = stable_image_id(image_path, cfg.dataset_root)
                    gt_abs = [o.to_absolute(width, height) for o in objects]

                    for i, (obj, box) in enumerate(zip(objects, gt_abs)):
                        crop_box = expand_box(
                            box, width, height, args.padding, args.square
                        )
                        x1, y1, x2, y2 = crop_box
                        if x2 - x1 < args.min_side or y2 - y1 < args.min_side:
                            skipped += 1
                            continue
                        class_dir = crop_root / f"{obj.class_id:03d}_{obj.class_name}"
                        class_dir.mkdir(parents=True, exist_ok=True)
                        crop_path = class_dir / f"{image_id}__obj{i:04d}.jpg"
                        image.crop(crop_box).save(
                            crop_path, "JPEG", quality=args.jpeg_quality, subsampling=0
                        )
                        dst.write(
                            json.dumps(
                                sample(crop_path, prompt, obj.class_name),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        counts[obj.class_name] += 1
                        written += 1

                    for i in range(args.background_per_image):
                        crop_box = sample_background_box(
                            rng, width, height, gt_abs, args.background_max_iou
                        )
                        if crop_box is None:
                            skipped += 1
                            continue
                        bg_dir = crop_root / f"background_{args.background_label}"
                        bg_dir.mkdir(parents=True, exist_ok=True)
                        crop_path = bg_dir / f"{image_id}__bg{i:03d}.jpg"
                        image.crop(crop_box).save(
                            crop_path, "JPEG", quality=args.jpeg_quality, subsampling=0
                        )
                        dst.write(
                            json.dumps(
                                sample(crop_path, prompt, args.background_label),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        counts[args.background_label] += 1
                        written += 1

        all_counts.update(counts)
        print(
            f"[{split}] source_images={len(images)} samples={written} skipped={skipped}"
        )
        print(f"[{split}] class_counts={dict(counts)}")

    metadata = {
        "task": "crop_classification",
        "source_yaml": str(cfg.yaml_path),
        "labels": labels,
        "padding": args.padding,
        "square": args.square,
        "background_per_image": args.background_per_image,
        "counts": dict(all_counts),
    }
    (out_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] {out_root}")


if __name__ == "__main__":
    main()
