from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class YoloObject:
    class_id: int
    class_name: str
    xyxy_norm: tuple[float, float, float, float]

    def to_absolute(self, width: int, height: int) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.xyxy_norm
        return x1 * width, y1 * height, x2 * width, y2 * height


@dataclass(frozen=True)
class DatasetConfig:
    yaml_path: Path
    dataset_root: Path
    names: list[str]
    split_sources: dict[str, object]


def load_dataset_config(yaml_path: str | Path) -> DatasetConfig:
    yaml_path = Path(yaml_path).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    names_raw = data.get("names")
    if isinstance(names_raw, list):
        names = [str(x) for x in names_raw]
    elif isinstance(names_raw, dict):
        keyed = {int(k): str(v) for k, v in names_raw.items()}
        if sorted(keyed) != list(range(len(keyed))):
            raise ValueError("data.yaml names dict must use continuous class IDs from 0")
        names = [keyed[i] for i in range(len(keyed))]
    else:
        raise ValueError("data.yaml must contain names as a list or dict")

    root_raw = data.get("path", ".")
    dataset_root = Path(root_raw).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (yaml_path.parent / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()

    split_sources = {k: data.get(k) for k in ("train", "val", "test") if data.get(k) is not None}
    if not split_sources:
        raise ValueError("data.yaml does not contain train/val/test paths")

    return DatasetConfig(yaml_path, dataset_root, names, split_sources)


def _resolve_source_path(source: str | Path, dataset_root: Path, list_parent: Path | None = None) -> Path:
    p = Path(source).expanduser()
    if p.is_absolute():
        return p.resolve()

    candidate = (dataset_root / p).resolve()
    if candidate.exists():
        return candidate

    if list_parent is not None:
        candidate2 = (list_parent / p).resolve()
        if candidate2.exists():
            return candidate2
    return candidate


def _iter_source(source: object, dataset_root: Path) -> Iterable[Path]:
    if isinstance(source, (list, tuple)):
        for item in source:
            yield from _iter_source(item, dataset_root)
        return
    if not isinstance(source, (str, Path)):
        raise TypeError(f"Unsupported split source: {source!r}")

    p = _resolve_source_path(source, dataset_root)
    if p.is_dir():
        for q in sorted(p.rglob("*")):
            if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES:
                yield q.resolve()
        return
    if p.is_file() and p.suffix.lower() == ".txt":
        with p.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    q = _resolve_source_path(line, dataset_root, p.parent)
                    if q.suffix.lower() in IMAGE_SUFFIXES:
                        yield q.resolve()
        return
    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
        yield p.resolve()
        return
    raise FileNotFoundError(f"Split source not found or unsupported: {p}")


def collect_images(source: object, dataset_root: Path) -> list[Path]:
    seen, result = set(), []
    for p in _iter_source(source, dataset_root):
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def label_path_for_image(image_path: Path, dataset_root: Path) -> Path:
    image_path = image_path.resolve()
    dataset_root = dataset_root.resolve()
    try:
        rel = image_path.relative_to(dataset_root)
    except ValueError:
        rel = Path(image_path.name)

    parts = list(rel.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return (dataset_root / Path(*parts)).with_suffix(".txt")
    if image_path.parent.name == "images":
        return image_path.parent.parent.joinpath("labels", image_path.name).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def parse_yolo_label(label_path: Path, names: Sequence[str], strict: bool = True) -> list[YoloObject]:
    if not label_path.exists():
        return []

    objects = []
    with label_path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            tokens = line.split()
            try:
                class_id = int(float(tokens[0]))
                values = [float(x) for x in tokens[1:]]
            except (ValueError, IndexError) as e:
                if strict:
                    raise ValueError(f"{label_path}:{line_no}: invalid row: {line}") from e
                continue

            if not 0 <= class_id < len(names):
                if strict:
                    raise ValueError(f"{label_path}:{line_no}: class_id {class_id} out of range")
                continue

            if len(values) == 4:
                xc, yc, bw, bh = values
                x1, y1, x2, y2 = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2
            elif len(values) >= 6 and len(values) % 2 == 0:
                xs, ys = values[0::2], values[1::2]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            else:
                if strict:
                    raise ValueError(f"{label_path}:{line_no}: unsupported YOLO row")
                continue

            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
            if x2 <= x1 or y2 <= y1:
                if strict:
                    raise ValueError(f"{label_path}:{line_no}: degenerate box")
                continue
            objects.append(YoloObject(class_id, names[class_id], (x1, y1, x2, y2)))
    return objects


def stable_image_id(image_path: Path, dataset_root: Path) -> str:
    try:
        rel = image_path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        rel = image_path.resolve().as_posix()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    return f"{image_path.stem}_{digest}"
