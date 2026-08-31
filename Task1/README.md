# Task1 — Desktop Object Detection and ROS2 Deployment

[简体中文](README_zh-CN.md)

## Overview

Task1 trains a lightweight YOLOv8n detector for three desktop-object classes and runs it on a Jetson camera node. The runtime displays bounding boxes, class names, confidence scores, and smoothed FPS while publishing every frame's detections as JSON on a ROS2 topic. It can also record the annotated video.

The current deployment class contract is:

| Class ID | Name | Plot color |
| ---: | --- | --- |
| 0 | `phone` | blue |
| 1 | `keyboard` | green |
| 2 | `bottle` | red |


## Repository layout

```text
Task1/
├── README.md
├── README_zh-CN.md
├── task.md
├── train_detector.py
├── preprocess_lowres_dataset.py
├── xby.py
└── experiments/
    └── <run-name>/
```

`train_detector.py` validates and normalizes labels, trains YOLOv8n, evaluates the test split, plots loss curves, and saves representative errors. `preprocess_lowres_dataset.py` is an optional in-place preprocessing utility for a writable dataset copy. `xby.py` is the Jetson/ROS2 camera runtime.

## Environment

### Training environment

The project was developed and tested with:

- Ubuntu with Python 3.10.12
- Ultralytics 8.4.127
- PyTorch 2.5.1 with CUDA 12.4
- NVIDIA GeForce RTX 4060, 8 GB

Python 3.10 or newer is recommended because the code uses modern type syntax. Install a PyTorch build appropriate for the host and CUDA version first, then install the Python dependencies:

```bash
cd /path/to/2026Summer/Task1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "ultralytics==8.4.127" opencv-python numpy PyYAML matplotlib
```

The first use of `yolov8n.pt` may download pretrained weights. Pass a local model path with `--model` when the training machine has no network access.

### Jetson and ROS2 environment

The Jetson runtime additionally requires:

- an NVIDIA Jetson with a JetPack-compatible PyTorch build;
- ROS2 with Python packages `rclpy` and `std_msgs`;
- Ultralytics and OpenCV with V4L2 camera support;
- a camera exposed as `/dev/videoN`;
- a graphical display for `cv2.imshow`.

Use the ROS2 distribution installed on the Jetson. For example, with ROS2 Humble:

```bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install "ultralytics==8.4.127"
```

On Jetson, install PyTorch using NVIDIA's JetPack-compatible instructions rather than replacing it with a generic desktop wheel. A system OpenCV package is often preferable because it preserves the board's camera integration.

## Dataset contract

The trainer expects this YOLO directory layout:

```text
dataset_XBY/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each detection row must use normalized YOLO coordinates:

```text
class_id x_center y_center width height
```

Polygon rows (`class_id x1 y1 x2 y2 ...`) are accepted and converted to enclosing axis-aligned boxes. Invalid rows are reported and skipped. Missing label files are treated as images without objects.

The XBY labels must already follow the deployment IDs:

```text
0 -> phone
1 -> keyboard
2 -> bottle
```

The saved remapping report records the historical conversion:

```text
old: 0 keyboard, 1 nongfu_spring, 2 phone
new: 0 phone,    1 keyboard,       2 bottle
map: old 0 -> 1, old 1 -> 2, old 2 -> 0
```

The trainer does **not** perform this semantic ID remapping. It only normalizes geometry. Use the prepared `dataset_XBY` or remap labels before training. The no-argument dataset default still points to the older Nongfu path, so always pass `--dataset` explicitly for the current three-class model.


## Validate the dataset

Run a data-only pass before a long training job:

```bash
cd /path/to/2026Summer/Task1
source .venv/bin/activate
python train_detector.py \
  --dataset ./dataset_XBY \
  --output ./experiments/xby_dataset_check \
  --check-only
```

This creates normalized labels, image symlinks, a generated data YAML, a dataset report, and a log under the selected experiment directory. Check:

```text
experiments/xby_dataset_check/configs/dataset_report.json
experiments/xby_dataset_check/configs/normalized_data.yaml
experiments/xby_dataset_check/logs/train.log
```

Review the counts and any skipped-label warnings in the report before starting training.

## Optional low-resolution preprocessing

`preprocess_lowres_dataset.py` modifies a writable dataset copy in place. It resizes base images, normalizes labels, and can add deterministic blur and low-light variants to the training split. It refuses to modify `/mnt/dataset/dataset_nongfu_checked` directly.

```bash
python preprocess_lowres_dataset.py \
  --dataset ./dataset_nongfu_checked \
  --width 640 \
  --height 480 \
  --augment-count 2 \
  --report ./experiments/lowres_prepare/configs/preprocess_report.json
```

`--augment-count` must be 0, 1, or 2. This utility currently imports the same `phone, keyboard, bottle` class contract as the trainer and rewrites `data.yaml` accordingly. Do not run it on an older class layout until its label IDs have been remapped. Keep the source dataset in `/mnt/dataset` unchanged and operate only on a disposable writable copy.

## Train and evaluate

Use a new output directory for each run. Reusing an existing directory appends to `logs/train.log` and may mix old and new artifacts.

```bash
python train_detector.py \
  --dataset ./dataset_XBY \
  --output ./experiments/yolov8n_XBY_phone_keyboard_bottle_repro01 \
  --model yolov8n.pt \
  --epochs 100 \
  --batch 16 \
  --workers 4 \
  --device 0 \
  --seed 42 \
  --conf 0.25 \
  --max-error-images 12
```

For CPU training, use `--device cpu`. Reduce `--batch` and `--workers` when GPU memory or host RAM is limited.

The script automatically performs the following sequence:

1. checks the `train`, `valid`, and `test` splits;
2. creates `normalized_dataset/` with image symlinks and normalized labels;
3. trains YOLOv8n with deterministic seed 42 and camera-oriented augmentation;
4. saves `best.pt` and `last.pt` under `weights/`;
5. evaluates the selected weight on the test split;
6. writes metrics, curves, confusion matrices, and up to 12 representative errors.

Important: the code requests `[480, 640]`, but Ultralytics 8.4.127 logs that train/validation image size is coerced to scalar `640`. The saved error examples are explicitly letterboxed to a 640×480 canvas, and the Jetson camera defaults to 640×480.

### Output layout

```text
experiments/<run-name>/
├── args.yaml
├── configs/
│   ├── dataset_report.json
│   ├── normalized_data.yaml
│   └── train_config.json
├── logs/train.log
├── normalized_dataset/       # generated working dataset
├── weights/                  # trained model weights
│   ├── best.pt
│   └── last.pt
├── results.csv
├── results.png
├── confusion_matrix*.png
├── test_eval/
└── results/
    ├── test_metrics.json
    ├── loss_curves.png
    └── error_examples/
```


## Deploy on Jetson with ROS2

Provide a trained weight through `--model`. The runtime has a default path, but an explicit path is safer:

```bash
cd /path/to/2026Summer/Task1
source /opt/ros/humble/setup.bash
source .venv/bin/activate

python xby.py \
  --model /home/nvidia/models/xby_best.pt \
  --camera 2 \
  --width 640 \
  --height 480 \
  --imgsz 640 \
  --conf 0.70 \
  --device 0 \
  --half \
  --topic /yolo/detections \
  --node-name yolo_detector
```

Press `q` in the OpenCV window or use `Ctrl+C` in the terminal to stop. If the selected weight does not contain the class names `phone`, `keyboard`, and `bottle` in that order, the node logs a warning.

For CPU inference:

```bash
python xby.py --model /path/to/best.pt --device cpu --no-half
```

### Inspect ROS2 output

In another sourced ROS2 terminal:

```bash
ros2 node list
ros2 topic list
ros2 topic echo /yolo/detections std_msgs/msg/String
```

Each message contains a JSON string like:

```json
{
  "fps": 12.4,
  "object_count": 1,
  "objects": [
    {
      "class_id": 0,
      "class_name": "phone",
      "confidence": 0.934,
      "bbox": {"x1": 101, "y1": 82, "x2": 330, "y2": 401}
    }
  ]
}
```

Coordinates are integer pixels in the captured frame. The publisher uses `std_msgs/msg/String` with queue depth 10.

### Record an annotated video

```bash
python xby.py \
  --model /home/nvidia/models/xby_best.pt \
  --camera 2 \
  --record \
  --record-dir ./recordings
```

The output name is `yolo_record_YYYYMMDD_HHMMSS.mp4`.

## Acceptance verification

The assignment requires at least two classes, at least 80% correct recognition over 20 tested objects, at least 5 FPS on Jetson, saved test results, and representative errors.

Before submission, collect the following evidence on the target Jetson:

1. test 20 physical objects and record the number correctly recognized;
2. record sustained FPS from the on-screen value or ROS2 messages;
3. save an annotated result video;
4. document the Jetson model, JetPack/ROS2 versions, camera index, confidence threshold, and weight filename.

Do not use desktop RTX 4060 timing or offline mAP as a substitute for these two live acceptance measurements.

## Troubleshooting

- **`Unable to open /dev/videoN`:** list devices with `ls /dev/video*` or `v4l2-ctl --list-devices`, then change `--camera`.
- **`rclpy` or `std_msgs` cannot be imported:** source the correct ROS2 setup script and use a virtual environment created with `--system-site-packages`.
- **FP16/CUDA failure:** use a JetPack-compatible PyTorch build; for CPU, pass `--device cpu --no-half`.
- **Model-class warning:** the weight was trained with a different class order. Use a weight trained for `phone`, `keyboard`, and `bottle` in that order.
- **No display over SSH:** `xby.py` always calls `cv2.imshow`; run with a graphical session/X forwarding or adapt the script for headless use.
- **Missing `best.pt`:** provide a compatible trained weight or train one with the command above.
- **Training output mixes runs:** choose a new `--output` directory instead of reusing an old experiment.
- **Unexpected labels:** confirm that XBY labels are already remapped to IDs 0/1/2 = phone/keyboard/bottle.
