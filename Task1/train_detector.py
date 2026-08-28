#!/usr/bin/env python3
"""Train a lightweight YOLO detector for the Nongfu desktop-object dataset.

The source dataset is kept read-only.  At startup this script creates a small
working dataset containing symlinks to the source images and normalized YOLO
box labels.  Polygon rows in the source labels are converted to their
axis-aligned bounding boxes so that mixed detection/segmentation annotations
can be used by a detection model.

The model performs resize and augmentation online during training.  After
training, the script evaluates the test split, writes a loss plot, and saves a
small number of processed 640x480 error examples with ground-truth and
predicted boxes overlaid.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


DEFAULT_DATASET = Path("/mnt/dataset/dataset_nongfu_checked")
DEFAULT_OUTPUT = Path("experiments/yolov8n_640x480")
CLASS_NAMES = ["keyboard", "nongfu_spring", "phone"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Tee:
    """Write terminal output to both the terminal and a log file."""

    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.console.write(text)
        self.log_file.write(text)
        self.console.flush()
        self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.log_file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"原始 YOLO 数据集目录，默认：{DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"实验输出目录，默认：{DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="预训练模型或模型配置，例如 yolov8n.pt",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数，默认 100")
    parser.add_argument("--batch", type=int, default=16, help="batch size，默认 16")
    parser.add_argument("--workers", type=int, default=4, help="数据加载进程数，默认 4")
    parser.add_argument(
        "--device",
        default="0",
        help="训练设备，默认 GPU 0；CPU 可传 cpu",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument(
        "--max-error-images",
        type=int,
        default=12,
        help="最多保存多少张典型错误图片，默认 12",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="错误样例推理的置信度阈值，默认 0.25",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查数据并生成规范化标签，不启动训练",
    )
    return parser.parse_args()


def iter_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def convert_label_row(parts: list[str], num_classes: int) -> tuple[int, float, float, float, float, bool]:
    """Convert one YOLO box or polygon row to a normalized xywh box."""
    if len(parts) < 5:
        raise ValueError(f"字段数量不足：{len(parts)}")

    class_value = float(parts[0])
    class_id = int(class_value)
    if class_value != class_id or not 0 <= class_id < num_classes:
        raise ValueError(f"非法类别 ID：{parts[0]}")

    coordinates = [float(value) for value in parts[1:]]
    if any(not np.isfinite(value) for value in coordinates):
        raise ValueError("坐标包含 NaN 或无穷值")
    if any(value < -0.01 or value > 1.01 for value in coordinates):
        raise ValueError("坐标不是归一化坐标")

    # Five fields is the standard YOLO detection format.  More fields are a
    # polygon row: class x1 y1 x2 y2 ...; use its enclosing box for detection.
    is_polygon = len(parts) > 5
    if not is_polygon:
        x_center, y_center, width, height = coordinates
    else:
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError(f"多边形坐标数量不合法：{len(coordinates)}")
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

    x_center = float(np.clip(x_center, 0.0, 1.0))
    y_center = float(np.clip(y_center, 0.0, 1.0))
    width = float(np.clip(width, 0.0, 1.0))
    height = float(np.clip(height, 0.0, 1.0))
    if width <= 0.0 or height <= 0.0:
        raise ValueError("目标框宽度或高度为 0")

    # Keep the resulting box inside the image after clipping.
    x_min = max(0.0, x_center - width / 2.0)
    y_min = max(0.0, y_center - height / 2.0)
    x_max = min(1.0, x_center + width / 2.0)
    y_max = min(1.0, y_center + height / 2.0)
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0.0 or height <= 0.0:
        raise ValueError("裁剪后目标框宽度或高度为 0")

    return class_id, x_center, y_center, width, height, is_polygon


def normalize_label_file(
    source_label: Path | None,
    target_label: Path,
    num_classes: int,
    stats: Counter,
) -> None:
    """Write a standard five-column YOLO label file."""
    normalized_rows: list[str] = []
    if source_label is not None and source_label.exists():
        for line_number, line in enumerate(source_label.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = convert_label_row(line.split(), num_classes)
            except (TypeError, ValueError) as exc:
                stats["invalid_rows"] += 1
                print(f"[警告] 跳过非法标签 {source_label}:{line_number}：{exc}")
                continue
            class_id, x_center, y_center, width, height, is_polygon = row
            stats["polygon_rows" if is_polygon else "box_rows"] += 1
            normalized_rows.append(
                f"{class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"
            )
            stats[f"class_{class_id}"] += 1
    else:
        stats["missing_labels"] += 1

    target_label.parent.mkdir(parents=True, exist_ok=True)
    target_label.write_text("\n".join(normalized_rows) + ("\n" if normalized_rows else ""), encoding="utf-8")


def ensure_image_link(source: Path, target: Path) -> None:
    """Create a symlink to a source image without copying the large originals."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return
        target.unlink()
    elif target.exists():
        raise FileExistsError(f"目标路径已存在且不是预期的符号链接：{target}")
    target.symlink_to(source.resolve())


def prepare_dataset(source_root: Path, output_root: Path) -> tuple[Path, Counter]:
    """Create a normalized, lightweight working dataset under the experiment."""
    if not source_root.is_dir():
        raise FileNotFoundError(f"找不到数据集目录：{source_root}")

    num_classes = len(CLASS_NAMES)
    normalized_root = output_root / "normalized_dataset"
    stats: Counter = Counter()
    split_mapping = {"train": "train", "valid": "val", "test": "test"}

    for source_split, yaml_split in split_mapping.items():
        source_images = source_root / source_split / "images"
        source_labels = source_root / source_split / "labels"
        if not source_images.is_dir():
            raise FileNotFoundError(f"找不到图片目录：{source_images}")

        target_images = normalized_root / source_split / "images"
        target_labels = normalized_root / source_split / "labels"
        images = iter_images(source_images)
        stats[f"{source_split}_images"] = len(images)
        if not images:
            raise RuntimeError(f"图片目录为空：{source_images}")

        for image_path in images:
            ensure_image_link(image_path, target_images / image_path.name)
            source_label = source_labels / f"{image_path.stem}.txt"
            normalize_label_file(
                source_label if source_label.exists() else None,
                target_labels / f"{image_path.stem}.txt",
                num_classes,
                stats,
            )

    data_yaml = output_root / "configs" / "normalized_data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    yaml_data = {
        "path": str(normalized_root.resolve()),
        "train": str((normalized_root / "train" / "images").resolve()),
        "val": str((normalized_root / "valid" / "images").resolve()),
        "test": str((normalized_root / "test" / "images").resolve()),
        "nc": num_classes,
        "names": CLASS_NAMES,
    }
    data_yaml.write_text(
        yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return data_yaml, stats


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def print_epoch_progress(trainer) -> None:
    """Fallback textual progress in addition to Ultralytics' own progress bar."""
    epoch = int(getattr(trainer, "epoch", 0)) + 1
    epochs = int(getattr(trainer, "epochs", 0))
    metrics = getattr(trainer, "metrics", {}) or {}
    tloss = getattr(trainer, "tloss", None)
    loss_text = ""
    if tloss is not None:
        try:
            values = tloss.detach().float().cpu().tolist()
            loss_text = " train_loss=" + ",".join(f"{float(value):.4f}" for value in values)
        except (AttributeError, TypeError, ValueError):
            loss_text = ""
    map50 = metrics.get("metrics/mAP50(B)")
    map5095 = metrics.get("metrics/mAP50-95(B)")
    metric_text = ""
    if map50 is not None:
        metric_text += f" mAP50={float(map50):.4f}"
    if map5095 is not None:
        metric_text += f" mAP50-95={float(map5095):.4f}"
    print(f"[训练进度] epoch {epoch}/{epochs}{loss_text}{metric_text}", flush=True)


def load_results_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(file)]


def save_loss_curves(run_root: Path, results_root: Path) -> Path | None:
    """Plot train/validation losses from Ultralytics' results.csv."""
    csv_path = run_root / "results.csv"
    if not csv_path.exists():
        print(f"[警告] 找不到训练指标文件，无法绘制 loss 曲线：{csv_path}")
        return None

    rows = load_results_csv(csv_path)
    if not rows:
        print("[警告] results.csv 为空，无法绘制 loss 曲线")
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[警告] 未安装 matplotlib，跳过 loss 曲线绘制")
        return None

    epochs = [float(row.get("epoch", index + 1)) + 1 for index, row in enumerate(rows)]
    loss_keys = [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    ]
    available_keys = [key for key in loss_keys if key in rows[0]]
    if not available_keys:
        print("[警告] results.csv 中没有识别到 loss 列")
        return None

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    train_keys = [key for key in available_keys if key.startswith("train/")]
    val_keys = [key for key in available_keys if key.startswith("val/")]
    for key in train_keys:
        values = [float(row[key]) for row in rows]
        axes[0].plot(epochs, values, label=key.removeprefix("train/"))
    for key in val_keys:
        values = [float(row[key]) for row in rows]
        axes[1].plot(epochs, values, label=key.removeprefix("val/"))
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("YOLOv8n loss curves")
    figure.tight_layout()
    output_path = results_root / "loss_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"[结果] loss 曲线已保存：{output_path}")
    return output_path


def letterbox_to_camera(image: np.ndarray, target_height: int = 480, target_width: int = 640):
    """Resize without distortion to the camera's 640x480 canvas."""
    source_height, source_width = image.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas, scale, offset_x, offset_y


def normalized_box_to_xyxy(
    box: Iterable[float], source_width: int, source_height: int, scale: float, offset_x: int, offset_y: int
) -> np.ndarray:
    x_center, y_center, width, height = map(float, box)
    x1 = (x_center - width / 2) * source_width * scale + offset_x
    y1 = (y_center - height / 2) * source_height * scale + offset_y
    x2 = (x_center + width / 2) * source_width * scale + offset_x
    y2 = (y_center + height / 2) * source_height * scale + offset_y
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def read_normalized_boxes(label_path: Path) -> list[tuple[int, np.ndarray]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        boxes.append((class_id, np.array([float(value) for value in parts[1:]], dtype=np.float32)))
    return boxes


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_predictions(
    ground_truth: list[tuple[int, np.ndarray]], predictions: list[tuple[int, float, np.ndarray]], iou_threshold: float = 0.5
) -> dict:
    """Greedily match predictions to ground truth and classify errors."""
    used_gt: set[int] = set()
    matched: list[tuple[int, int, bool, float]] = []
    false_positives: list[int] = []
    wrong_class: list[tuple[int, int, float]] = []

    for prediction_index, (pred_class, _confidence, pred_box) in enumerate(predictions):
        candidates = [
            (box_iou(pred_box, gt_box), gt_index)
            for gt_index, (_gt_class, gt_box) in enumerate(ground_truth)
            if gt_index not in used_gt
        ]
        if not candidates:
            false_positives.append(prediction_index)
            continue
        best_iou, best_gt_index = max(candidates)
        if best_iou < iou_threshold:
            false_positives.append(prediction_index)
            continue
        used_gt.add(best_gt_index)
        gt_class = ground_truth[best_gt_index][0]
        is_correct = pred_class == gt_class
        matched.append((prediction_index, best_gt_index, is_correct, best_iou))
        if not is_correct:
            wrong_class.append((prediction_index, best_gt_index, best_iou))

    false_negatives = [index for index in range(len(ground_truth)) if index not in used_gt]
    return {
        "matched": matched,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "wrong_class": wrong_class,
    }


def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int], text: str) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text_y = max(18, y1 - 5)
    cv2.putText(image, text, (max(0, x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def save_error_examples(
    model,
    normalized_root: Path,
    output_root: Path,
    device: str,
    confidence: float,
    max_images: int,
) -> None:
    """Run test inference on processed 640x480 images and save representative errors."""
    error_root = output_root / "results" / "error_examples"
    error_root.mkdir(parents=True, exist_ok=True)
    test_images = iter_images(normalized_root / "test" / "images")
    candidates: list[tuple[int, Path, np.ndarray, list[tuple[int, np.ndarray]], list[tuple[int, float, np.ndarray]], dict]] = []

    for image_index, image_path in enumerate(test_images, start=1):
        original = cv2.imread(str(image_path))
        if original is None:
            print(f"[警告] 无法读取测试图片：{image_path}")
            continue
        processed, scale, offset_x, offset_y = letterbox_to_camera(original)
        source_height, source_width = original.shape[:2]
        gt_boxes = [
            (class_id, normalized_box_to_xyxy(box, source_width, source_height, scale, offset_x, offset_y))
            for class_id, box in read_normalized_boxes(normalized_root / "test" / "labels" / f"{image_path.stem}.txt")
        ]

        prediction_result = model.predict(
            source=processed,
            imgsz=[480, 640],
            conf=confidence,
            iou=0.7,
            device=device,
            verbose=False,
        )[0]
        predictions: list[tuple[int, float, np.ndarray]] = []
        if prediction_result.boxes is not None:
            boxes = prediction_result.boxes.xyxy.detach().cpu().numpy()
            classes = prediction_result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = prediction_result.boxes.conf.detach().cpu().numpy()
            order = np.argsort(-confidences)
            predictions = [(int(classes[i]), float(confidences[i]), boxes[i]) for i in order]

        matching = match_predictions(gt_boxes, predictions)
        fp_count = len(matching["false_positives"])
        fn_count = len(matching["false_negatives"])
        wrong_count = len(matching["wrong_class"])
        score = fp_count + 2 * fn_count + 3 * wrong_count
        if score > 0:
            candidates.append((score, image_path, processed, gt_boxes, predictions, matching))
        if image_index == 1 or image_index % 10 == 0 or image_index == len(test_images):
            print(f"[错误样例进度] {image_index}/{len(test_images)}", flush=True)

    candidates.sort(key=lambda item: (-item[0], item[1].name))
    summary = []
    for rank, (score, image_path, processed, gt_boxes, predictions, matching) in enumerate(
        candidates[:max_images], start=1
    ):
        canvas = processed.copy()
        wrong_gt = {gt_index for _pred_index, gt_index, _iou in matching["wrong_class"]}
        for gt_index, (class_id, box) in enumerate(gt_boxes):
            color = (0, 165, 255) if gt_index in matching["false_negatives"] or gt_index in wrong_gt else (0, 190, 0)
            draw_box(canvas, box, color, f"GT:{CLASS_NAMES[class_id]}")

        wrong_predictions = {pred_index for pred_index, _gt_index, _iou in matching["wrong_class"]}
        wrong_predictions.update(matching["false_positives"])
        matched_prediction_indices = {pred_index for pred_index, _gt_index, is_correct, _iou in matching["matched"] if is_correct}
        for pred_index, (class_id, confidence, box) in enumerate(predictions):
            if pred_index in wrong_predictions:
                color = (0, 0, 220)
            elif pred_index in matched_prediction_indices:
                color = (220, 120, 0)
            else:
                color = (0, 0, 220)
            draw_box(canvas, box, color, f"P:{CLASS_NAMES[class_id]} {confidence:.2f}")

        header = f"score={score}  FP={len(matching['false_positives'])}  FN={len(matching['false_negatives'])}  CLS={len(matching['wrong_class'])}"
        cv2.rectangle(canvas, (0, 0), (640, 28), (35, 35, 35), -1)
        cv2.putText(canvas, header, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        target_path = error_root / f"{rank:02d}_{image_path.stem}.jpg"
        cv2.imwrite(str(target_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        summary.append(
            {
                "rank": rank,
                "image": image_path.name,
                "output": str(target_path),
                "score": score,
                "false_positives": len(matching["false_positives"]),
                "false_negatives": len(matching["false_negatives"]),
                "wrong_class": len(matching["wrong_class"]),
                "ground_truth_count": len(gt_boxes),
                "prediction_count": len(predictions),
            }
        )

    write_json(error_root / "summary.json", summary)
    print(f"[结果] 已保存 {len(summary)} 张典型错误图片：{error_root}")


def run_training(args: argparse.Namespace, data_yaml: Path, output_root: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.add_callback("on_fit_epoch_end", print_epoch_progress)
    image_size = [480, 640]
    print(f"[训练] 模型：{args.model}")
    print(f"[训练] 输入尺寸：{image_size[0]}x{image_size[1]}，GPU：{args.device}，batch：{args.batch}")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=image_size,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(output_root.parent.resolve()),
        name=output_root.name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        patience=30,
        seed=args.seed,
        deterministic=True,
        amp=True,
        cache=False,
        plots=True,
        verbose=True,
        save=True,
        save_period=10,
        rect=False,
        degrees=7.0,
        translate=0.08,
        scale=0.30,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.40,
        mosaic=0.50,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10,
    )
    best_weight = output_root / "weights" / "best.pt"
    last_weight = output_root / "weights" / "last.pt"
    if best_weight.exists():
        return best_weight
    if last_weight.exists():
        return last_weight
    raise FileNotFoundError(f"训练结束但没有找到模型权重：{output_root / 'weights'}")


def evaluate_and_save_results(
    weights: Path,
    data_yaml: Path,
    normalized_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> None:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    print("[评估] 开始在 test 集上评估……")
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=[480, 640],
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(output_root),
        name="test_eval",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    metric_dict = {}
    for key, value in (getattr(metrics, "results_dict", {}) or {}).items():
        try:
            metric_dict[key] = float(value)
        except (TypeError, ValueError):
            metric_dict[key] = str(value)
    write_json(output_root / "results" / "test_metrics.json", metric_dict)
    print(f"[结果] 测试指标已保存：{output_root / 'results' / 'test_metrics.json'}")
    save_error_examples(
        model=model,
        normalized_root=normalized_root,
        output_root=output_root,
        device=args.device,
        confidence=args.conf,
        max_images=args.max_error_images,
    )


def main() -> None:
    args = parse_args()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "configs").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    (output_root / "results").mkdir(parents=True, exist_ok=True)

    log_path = output_root / "logs" / "train.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("a", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print("=" * 72)
            print("Task1 目标检测训练")
            print(f"数据集：{args.dataset.resolve()}")
            print(f"输出目录：{output_root}")
            data_yaml, stats = prepare_dataset(args.dataset.resolve(), output_root)
            write_json(output_root / "configs" / "dataset_report.json", dict(stats))
            config = vars(args).copy()
            config.update(
                {
                    "dataset": str(args.dataset.resolve()),
                    "output": str(output_root),
                    "normalized_data_yaml": str(data_yaml),
                    "image_size": [480, 640],
                    "classes": CLASS_NAMES,
                }
            )
            write_json(output_root / "configs" / "train_config.json", config)
            print(f"[数据] 训练图片：{stats['train_images']}，验证图片：{stats['valid_images']}，测试图片：{stats['test_images']}")
            print(f"[数据] 标准框：{stats['box_rows']}，多边形转框：{stats['polygon_rows']}，非法标签行：{stats['invalid_rows']}")
            print(f"[数据] 规范化配置：{data_yaml}")

            if args.check_only:
                print("[检查] 完成。由于指定了 --check-only，未启动训练。")
                return

            weights = run_training(args, data_yaml, output_root)
            save_loss_curves(output_root, output_root / "results")
            print(f"[结果] 最佳模型：{weights}")
            evaluate_and_save_results(weights, data_yaml, output_root / "normalized_dataset", output_root, args)
            print("[完成] 训练、测试评估、loss 曲线和典型错误图片均已生成。")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
