#!/usr/bin/env python3

from ultralytics import YOLO
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from yolov8_msgs.msg import InferenceResult
from yolov8_msgs.msg import Yolov8Inference

bridge = CvBridge()


class CameraSubscriber(Node):
    def __init__(self):
        super().__init__("camera_subscriber")

        model_path = Path(__file__).parent / "best.pt"
        self.model = YOLO(str(model_path))

        # Tuned for CPU + prevent NMS explosion
        self.conf_thres = 0.35
        self.iou_thres = 0.5
        self.max_det = 10
        self.imgsz = 640  # try 480 if still slow

        self.subscription = self.create_subscription(
            Image,
            "/image_raw",
            self.camera_callback,
            10,
        )

        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 1)
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)

        self.get_logger().info(f"Loaded model. names={self.model.names}")

    def camera_callback(self, data: Image):
        img = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

        results = self.model.predict(
            source=img,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            imgsz=self.imgsz,
            verbose=False,
        )

        yolov8_msg = Yolov8Inference()
        yolov8_msg.header.frame_id = "inference"
        yolov8_msg.header.stamp = self.get_clock().now().to_msg()

        det_count = 0

        for r in results:
            if getattr(r, "obb", None) is not None:
                boxes = r.obb
                is_obb = True
            elif getattr(r, "boxes", None) is not None:
                boxes = r.boxes
                is_obb = False
            else:
                continue

            for box in boxes:
                conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else 1.0
                if conf < self.conf_thres:
                    continue

                if is_obb:
                    coordinates = box.xyxyxyxy[0].cpu().numpy().copy().reshape(8).tolist()
                else:
                    coordinates = box.xyxy[0].cpu().numpy().copy().tolist()

                # Single-class model → force class_name to "banana"
                inf = InferenceResult()
                inf.class_name = "banana"
                inf.coordinates = coordinates

                yolov8_msg.yolov8_inference.append(inf)
                det_count += 1

        self.get_logger().info(f"Published {det_count} detections")
        self.yolov8_pub.publish(yolov8_msg)

        # Annotated image
        try:
            annotated = results[0].plot()
            img_msg = bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            self.img_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Could not publish annotated image: {e}")


def main():
    rclpy.init(args=None)
    node = CameraSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()