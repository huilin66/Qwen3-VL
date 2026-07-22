#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from yolo_common import (
    collect_images,
    label_path_for_image,
    load_dataset_config,
    parse_yolo_label,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert YOLO detection/segmentation data to full-image Qwen3-VL grounding SFT data."
    )
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--decimals", type=int, default=2)
    p.add_argument("--exclude-empty", action="store_true")
    p.add_argument("--skip-invalid", action="store_true")
    return p.parse_args()


def make_prompt(names: list[str]) -> str:
    return (
        "Detect every target belonging to the following classes: "
        f"{json.dumps(names, ensure_ascii=False)}. "
        "Return only a JSON array. Each item must contain exactly "
        '"bbox_2d": [x1, y1, x2, y2] and "label". '
        "Return [] when no target exists."
    )


def make_answer(n: int) -> str:
    if n == 0:
        return "[]"
    row = '  {"bbox_2d": <bbox>, "label": "<ref-object>"}'
    return "[\n" + ",\n".join([row] * n) + "\n]"


def main() -> None:
    args = parse_args()
    cfg = load_dataset_config(args.data)
    out_root = Path(args.output).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    prompt = make_prompt(cfg.names)

    global_counts = Counter()
    global_empty = 0

    for split in args.splits:
        source = cfg.split_sources.get(split)
        if source is None:
            print(f"[WARN] split '{split}' is absent; skipped")
            continue

        images = collect_images(source, cfg.dataset_root)
        jsonl_path = out_root / f"{split}.jsonl"
        counts, written, empty_count, box_count = Counter(), 0, 0, 0

        with jsonl_path.open("w", encoding="utf-8") as dst:
            for image_path in tqdm(images, desc=f"Converting {split}", unit="image"):
                objects = parse_yolo_label(
                    label_path_for_image(image_path, cfg.dataset_root),
                    cfg.names,
                    strict=not args.skip_invalid,
                )
                if not objects and args.exclude_empty:
                    continue

                with Image.open(image_path) as image:
                    width, height = image.size

                objects = sorted(
                    objects,
                    key=lambda o: (
                        o.to_absolute(width, height)[1],
                        o.to_absolute(width, height)[0],
                        o.class_id,
                    ),
                )
                refs, boxes = [], []
                for obj in objects:
                    x1, y1, x2, y2 = obj.to_absolute(width, height)
                    refs.append(obj.class_name)
                    boxes.append(
                        [
                            round(x1, args.decimals),
                            round(y1, args.decimals),
                            round(x2, args.decimals),
                            round(y2, args.decimals),
                        ]
                    )
                    counts[obj.class_name] += 1
                    box_count += 1

                if not objects:
                    empty_count += 1

                row = {
                    "messages": [
                        {"role": "user", "content": f"<image>{prompt}"},
                        {"role": "assistant", "content": make_answer(len(objects))},
                    ],
                    "images": [str(image_path.resolve())],
                    "objects": {"ref": refs, "bbox": boxes, "bbox_type": "real"},
                }
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

        global_counts.update(counts)
        global_empty += empty_count
        print(
            f"[{split}] source_images={len(images)} samples={written} "
            f"boxes={box_count} empty_images={empty_count}"
        )
        print(f"[{split}] class_counts={dict(counts)}")

    metadata = {
        "task": "full_detection",
        "source_yaml": str(cfg.yaml_path),
        "labels": cfg.names,
        "bbox_type": "real",
        "coordinate_note": (
            "Absolute source-image xyxy coordinates. ms-swift converts them to "
            "Qwen3-VL normalized-1000 coordinates during preprocessing."
        ),
        "empty_images": global_empty,
        "counts": dict(global_counts),
    }
    (out_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] {out_root}")


if __name__ == "__main__":
    main()
