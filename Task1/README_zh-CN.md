# Task1——桌面物体检测与 ROS2 部署

[English](README.md)

## 项目概述

Task1 使用轻量级 YOLOv8n 训练三类桌面物体检测模型，并在 Jetson 摄像头节点上运行。运行程序实时显示检测框、类别、置信度和平滑 FPS，同时将每帧检测结果以 JSON 字符串发布到 ROS2 话题，并可选录制带标注视频。

当前部署类别约定如下：

| 类别 ID | 名称 | 绘图颜色 |
| ---: | --- | --- |
| 0 | `phone` | 蓝色 |
| 1 | `keyboard` | 绿色 |
| 2 | `bottle` | 红色 |


## 目录结构

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

`train_detector.py` 负责标签检查与规范化、YOLOv8n 训练、测试集评估、loss 曲线和典型错误样例生成。`preprocess_lowres_dataset.py` 是可选的可写数据集离线预处理工具。`xby.py` 是 Jetson/ROS2 摄像头实时运行程序。

## 环境要求

### 训练环境

项目开发与测试使用了以下环境：

- Ubuntu，Python 3.10.12
- Ultralytics 8.4.127
- PyTorch 2.5.1，CUDA 12.4
- NVIDIA GeForce RTX 4060，8 GB 显存

代码使用了较新的类型语法，建议使用 Python 3.10 或更高版本。先安装与训练主机和 CUDA 匹配的 PyTorch，再安装 Python 依赖：

```bash
cd /path/to/2026Summer/Task1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "ultralytics==8.4.127" opencv-python numpy PyYAML matplotlib
```

首次使用 `yolov8n.pt` 时可能会自动下载预训练权重。训练机无法联网时，请通过 `--model` 指定本地模型路径。

### Jetson 与 ROS2 环境

Jetson 实时运行还需要：

- 安装了与 JetPack 兼容 PyTorch 的 NVIDIA Jetson；
- ROS2，以及 Python 包 `rclpy`、`std_msgs`；
- Ultralytics 和支持 V4L2 摄像头的 OpenCV；
- 映射为 `/dev/videoN` 的摄像头；
- 可供 `cv2.imshow` 使用的图形显示环境。

应使用 Jetson 上实际安装的 ROS2 发行版。例如 ROS2 Humble：

```bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install "ultralytics==8.4.127"
```

Jetson 上应按 NVIDIA 的 JetPack 兼容说明安装 PyTorch，不要用通用桌面 wheel 覆盖。系统自带 OpenCV 通常更适合保留开发板的摄像头集成能力。

## 数据集约定

训练程序要求以下 YOLO 目录结构：

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

标准检测标签行使用归一化 YOLO 坐标：

```text
class_id x_center y_center width height
```

程序也接受多边形行（`class_id x1 y1 x2 y2 ...`），并转换为外接轴对齐矩形。非法行会记录警告并跳过；标签文件缺失时，图片按无目标样本处理。

XBY 标签必须事先符合当前部署 ID：

```text
0 -> phone
1 -> keyboard
2 -> bottle
```

保存的重映射报告记录了历史转换关系：

```text
旧：0 keyboard，1 nongfu_spring，2 phone
新：0 phone，   1 keyboard，       2 bottle
映射：旧 0 -> 1，旧 1 -> 2，旧 2 -> 0
```

训练程序本身**不会**执行语义类别 ID 重映射，只会规范化框的几何格式。请使用已经处理好的 `dataset_XBY`，或在训练前先完成标签重映射。无参数运行时的数据集默认值仍指向旧 Nongfu 路径，因此训练当前三类模型时必须显式传入 `--dataset`。


## 检查数据集

开始长时间训练前，先执行仅检查流程：

```bash
cd /path/to/2026Summer/Task1
source .venv/bin/activate
python train_detector.py \
  --dataset ./dataset_XBY \
  --output ./experiments/xby_dataset_check \
  --check-only
```

该命令会在指定实验目录中创建规范化标签、图片符号链接、生成的数据 YAML、数据集报告和日志。重点检查：

```text
experiments/xby_dataset_check/configs/dataset_report.json
experiments/xby_dataset_check/configs/normalized_data.yaml
experiments/xby_dataset_check/logs/train.log
```

开始训练前，请检查报告中的样本数量及被跳过标签的警告信息。

## 可选低清预处理

`preprocess_lowres_dataset.py` 会原地修改可写数据集副本：调整基础图片尺寸、规范化标签，并可在训练集增加确定性的模糊和低照度变体。程序拒绝直接修改 `/mnt/dataset/dataset_nongfu_checked`。

```bash
python preprocess_lowres_dataset.py \
  --dataset ./dataset_nongfu_checked \
  --width 640 \
  --height 480 \
  --augment-count 2 \
  --report ./experiments/lowres_prepare/configs/preprocess_report.json
```

`--augment-count` 只能是 0、1 或 2。该工具当前与训练程序共用 `phone, keyboard, bottle` 类别约定，并会据此重写 `data.yaml`。旧类别布局的标签未重映射前，不要运行该工具。必须保持 `/mnt/dataset` 中的源数据不变，仅操作一次性的可写副本。

## 训练与评估

每次实验应使用新的输出目录。复用已有目录会向 `logs/train.log` 追加内容，也可能混合新旧产物。

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

CPU 训练使用 `--device cpu`。显存或主机内存不足时可降低 `--batch` 和 `--workers`。

脚本会自动执行：

1. 检查 `train`、`valid`、`test` 三个划分；
2. 创建包含图片符号链接和规范化标签的 `normalized_dataset/`；
3. 使用固定随机种子 42 和面向摄像头场景的数据增强训练 YOLOv8n；
4. 在 `weights/` 保存 `best.pt` 和 `last.pt`；
5. 使用选定权重评估测试集；
6. 写出指标、曲线、混淆矩阵和最多 12 张典型错误样例。

注意：代码请求的输入尺寸为 `[480, 640]`，但 Ultralytics 8.4.127 日志显示训练和验证尺寸会被转换为标量 `640`。保存的错误样例明确 letterbox 到 640×480 画布，Jetson 摄像头默认分辨率也是 640×480。

### 输出结构

```text
experiments/<run-name>/
├── args.yaml
├── configs/
│   ├── dataset_report.json
│   ├── normalized_data.yaml
│   └── train_config.json
├── logs/train.log
├── normalized_dataset/       # 自动生成的工作数据集
├── weights/                  # 训练得到的模型权重
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


## 在 Jetson 上通过 ROS2 运行

通过 `--model` 指定训练好的权重。运行程序带有默认模型路径，但建议显式指定：

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

在 OpenCV 窗口中按 `q`，或在终端按 `Ctrl+C` 退出。若权重中的类别名及顺序不是 `phone`、`keyboard`、`bottle`，节点会输出警告。

CPU 推理使用：

```bash
python xby.py --model /path/to/best.pt --device cpu --no-half
```

### 检查 ROS2 输出

在另一个已 source ROS2 的终端中执行：

```bash
ros2 node list
ros2 topic list
ros2 topic echo /yolo/detections std_msgs/msg/String
```

每条消息包含类似以下内容的 JSON 字符串：

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

坐标为采集画面中的整数像素值。发布器使用 `std_msgs/msg/String`，队列深度为 10。

### 录制带标注视频

```bash
python xby.py \
  --model /home/nvidia/models/xby_best.pt \
  --camera 2 \
  --record \
  --record-dir ./recordings
```

输出文件名为 `yolo_record_YYYYMMDD_HHMMSS.mp4`。

## 验收复核

任务要求：不少于两类、20 个实物识别正确率不低于 80%、Jetson 不低于 5 FPS、保存测试结果和典型错误案例。

提交前，请在目标 Jetson 上记录以下实测证据：

1. 测试 20 个实物并记录正确识别数量；
2. 从画面或 ROS2 消息记录持续 FPS；
3. 保存一段带标注结果视频；
4. 记录 Jetson 型号、JetPack/ROS2 版本、摄像头编号、置信度阈值和权重文件名。

不能使用桌面 RTX 4060 速度或离线 mAP 替代上述两项目标机验收数据。

## 常见问题

- **无法打开 `/dev/videoN`：** 使用 `ls /dev/video*` 或 `v4l2-ctl --list-devices` 查看设备，并调整 `--camera`。
- **无法导入 `rclpy` 或 `std_msgs`：** source 正确的 ROS2 环境，并使用带 `--system-site-packages` 的虚拟环境。
- **FP16/CUDA 错误：** 安装与 JetPack 兼容的 PyTorch；CPU 模式传入 `--device cpu --no-half`。
- **模型类别警告：** 权重使用了不同类别顺序，应使用按 `phone`、`keyboard`、`bottle` 顺序训练的权重。
- **SSH 下没有窗口：** `xby.py` 始终调用 `cv2.imshow`，需使用图形会话/X 转发，或修改脚本支持无界面模式。
- **找不到 `best.pt`：** 请提供兼容的训练权重，或使用上面的命令重新训练。
- **训练结果混合：** 不要复用旧实验目录，应设置新的 `--output`。
- **标签含义异常：** 确认 XBY 标签已经重映射为 0/1/2 = phone/keyboard/bottle。
