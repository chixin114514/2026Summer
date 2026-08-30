import cv2
import time
import json
import argparse

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ultralytics import YOLO


# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/home/nvidia/best_gjs_1.pt")
parser.add_argument("--camera", type=int, default=2)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--conf", type=float, default=0.7)
parser.add_argument("--device", default=0)
parser.add_argument("--half", dest="half", action="store_true")
parser.add_argument("--no-half", dest="half", action="store_false")
parser.set_defaults(half=True)
parser.add_argument("--topic", default="/yolo/detections")
parser.add_argument("--node-name", default="yolo_detector")
args = parser.parse_args()


# =========================
# ROS2 初始化
# =========================
rclpy.init()

node = Node(args.node_name)

publisher = node.create_publisher(
    String,
    args.topic,
    10
)

node.get_logger().info("YOLO ROS2 detector started")
node.get_logger().info(f"Publishing: {args.topic}")


# =========================
# 1. 加载 YOLO 模型
# =========================
model = YOLO(args.model)


# =========================
# 2. 打开摄像头
# =========================
cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)

if not cap.isOpened():
    node.destroy_node()
    rclpy.shutdown()
    raise RuntimeError(f"无法打开摄像头 /dev/video{args.camera}")


# 设置摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)


prev_time = time.perf_counter()
fps = 0.0


try:
    while rclpy.ok():

        ret, frame = cap.read()

        if not ret:
            print("读取摄像头失败")
            continue


        # =========================
        # 3. YOLO 推理
        # =========================
        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            half=args.half,
            verbose=False
        )

        result = results[0]


        # =========================
        # 4. 解析检测结果
        # =========================
        detections = []

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

            class_name = model.names[class_id]

            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 3),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            }

            detections.append(detection)


        # =========================
        # 5. 计算 FPS
        # =========================
        current_time = time.perf_counter()

        delta = current_time - prev_time
        prev_time = current_time

        if delta > 0:

            current_fps = 1.0 / delta

            # 平滑 FPS
            fps = fps * 0.9 + current_fps * 0.1


        # =========================
        # 6. ROS2 发布检测结果
        # =========================
        ros_data = {
            "fps": round(fps, 1),
            "object_count": len(detections),
            "objects": detections
        }

        msg = String()

        msg.data = json.dumps(
            ros_data,
            ensure_ascii=False
        )

        publisher.publish(msg)


        # =========================
        # 7. 绘制 YOLO 检测框
        # =========================
        annotated_frame = result.plot()


        # 显示 FPS
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


        # 显示 ROS2 状态
        cv2.putText(
            annotated_frame,
            f"ROS2: {len(detections)} objects",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


        # =========================
        # 8. 显示画面
        # =========================
        cv2.imshow(
            "YOLO Detection",
            annotated_frame
        )


        # 按 q 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


except KeyboardInterrupt:
    pass


finally:

    cap.release()

    cv2.destroyAllWindows()

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()
