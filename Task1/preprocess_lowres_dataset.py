#!/usr/bin/env python3
"""Rewrite the writable dataset to match a 640x480 low-resolution camera.

This is an offline preprocessing step.  It intentionally modifies only the
writable ``Task1/dataset_nongfu_checked`` copy, never the read-only dataset in
``/mnt/dataset``.  Base images are resized in place to 640x480, labels are
normalized to detection boxes, and two deterministic camera-degradation
variants are added to the training split.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from train_detector import CLASS_NAMES, convert_label_row


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "dataset_nongfu_checked"
DEFAULT_REPORT = PROJECT_ROOT / "experiments/yolov8n_lowres_640x480/configs/preprocess_report.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--augment-count", type=int, default=2)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.stem.startswith("aug_blur_")
        and not path.stem.startswith("aug_lowlight_")
    )


def atomic_write_image(path: Path, image: np.ndarray, quality: int) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if not cv2.imwrite(str(temporary), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]):
        raise OSError(f"无法写入图片：{temporary}")
    os.replace(temporary, path)


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def normalize_label(path: Path, stats: Counter) -> str:
    rows: list[str] = []
    if path.exists():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                class_id, x_center, y_center, width, height, is_polygon = convert_label_row(
                    line.split(), len(CLASS_NAMES)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}：{exc}") from exc
            stats["polygon_rows" if is_polygon else "box_rows"] += 1
            stats[f"class_{class_id}"] += 1
            rows.append(f"{class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}")
    content = "\n".join(rows) + ("\n" if rows else "")
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        atomic_write_text(path, content)
    return content


def make_blur_variant(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    noise = rng.normal(0.0, 3.0, size=blurred.shape)
    return np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def make_lowlight_variant(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alpha = float(rng.uniform(0.68, 0.88))
    beta = float(rng.uniform(-12.0, -2.0))
    result = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(result, (3, 3), 0)


def update_data_yaml(dataset: Path) -> Path:
    yaml_path = dataset / "data.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) if yaml_path.exists() else {}
    data.update(
        {
            "path": str(dataset.resolve()),
            "train": str((dataset / "train" / "images").resolve()),
            "val": str((dataset / "valid" / "images").resolve()),
            "test": str((dataset / "test" / "images").resolve()),
            "nc": len(CLASS_NAMES),
            "names": CLASS_NAMES,
        }
    )
    atomic_write_text(yaml_path, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    return yaml_path


def process_dataset(dataset: Path, width: int, height: int, augment_count: int) -> dict:
    dataset = dataset.resolve()
    readonly_dataset = Path("/mnt/dataset/dataset_nongfu_checked").resolve()
    if dataset == readonly_dataset:
        raise ValueError("拒绝修改 /mnt/dataset 中的只读数据集，请使用 Task1/dataset_nongfu_checked 副本")
    if not dataset.is_dir():
        raise FileNotFoundError(f"找不到数据集：{dataset}")

    report: Counter = Counter()
    for split in ("train", "valid", "test"):
        images_dir = dataset / split / "images"
        labels_dir = dataset / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        base_images = image_files(images_dir)
        if not base_images:
            raise RuntimeError(f"图片目录为空：{images_dir}")
        report[f"{split}_base_images_before"] = len(base_images)

        for image_path in base_images:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"无法读取图片：{image_path}")
            if image.shape[1] != width or image.shape[0] != height:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                atomic_write_image(image_path, image, quality=82)
                report["resized_images"] += 1
            else:
                report["already_target_size"] += 1

            label_path = labels_dir / f"{image_path.stem}.txt"
            label_content = normalize_label(label_path, report)

            if split == "train":
                seed = sum(bytearray(image_path.stem.encode("utf-8")))
                rng = np.random.default_rng(seed)
                variants = (
                    ("aug_blur", make_blur_variant(image, rng), 62),
                    ("aug_lowlight", make_lowlight_variant(image, rng), 70),
                )
                for variant_index, (prefix, variant, quality) in enumerate(variants[:augment_count], start=1):
                    variant_path = images_dir / f"{prefix}_{image_path.stem}.jpg"
                    variant_label = labels_dir / f"{variant_path.stem}.txt"
                    if not variant_path.exists():
                        atomic_write_image(variant_path, variant, quality=quality)
                        atomic_write_text(variant_label, label_content)
                        report[f"{prefix}_created"] += 1
                    elif not variant_label.exists():
                        atomic_write_text(variant_label, label_content)
                    report[f"{prefix}_images"] += 1

        report[f"{split}_images_after"] = len(
            [path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        )

    update_data_yaml(dataset)
    report["width"] = width
    report["height"] = height
    report["augment_count"] = augment_count
    return dict(report)


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("目标尺寸必须为正数")
    if not 0 <= args.augment_count <= 2:
        raise ValueError("augment-count 只能是 0、1 或 2")
    report = process_dataset(args.dataset, args.width, args.height, args.augment_count)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[完成] 已离线处理数据集：{args.dataset.resolve()}")
    print(f"[完成] 数据配置：{(args.dataset / 'data.yaml').resolve()}")
    print(f"[完成] 处理报告：{args.report.resolve()}")


if __name__ == "__main__":
    main()
