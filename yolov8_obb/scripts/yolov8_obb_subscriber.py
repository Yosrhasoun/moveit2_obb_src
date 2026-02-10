#!/usr/bin/env python3

import cv2
import numpy as np
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from yolov8_msgs.msg import Yolov8Inference

bridge = CvBridge()

# shared latest frame
img = None


class CameraSubscriber(Node):
    def __init__(self):
        super().__init__("camera_subscriber")
        self.subscription = self.create_subscription(
            Image,
            "/image_raw",              # FIX 1: match exact topic
            self.camera_callback,
            10,
        )

    def camera_callback(self, data):
        global img
        img = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")


class YoloSubscriber(Node):
    def __init__(self):
        super().__init__("yolo_subscriber")
        self.subscription = self.create_subscription(
            Yolov8Inference,
            "/Yolov8_Inference",
            self.yolo_callback,
            10,
        )
        self.img_pub = self.create_publisher(Image, "/inference_result_cv2", 1)

    def yolo_callback(self, data):
        global img
        if img is None:
            return

        # FIX 3: draw on a fresh copy every time (no accumulation)
        frame = img.copy()

        for r in data.yolov8_inference:
            coords = np.array(r.coordinates, dtype=np.float32)

            # FIX 4: support OBB (8 vals) OR bbox (4 vals)
            if coords.size == 8:
                pts = coords.reshape(4, 2).astype(np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            elif coords.size == 4:
                x1, y1, x2, y2 = coords.astype(np.int32).tolist()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                # unexpected format, skip
                continue

            # optional: label
            cv2.putText(frame, r.class_name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # FIX 2: specify encoding
        img_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.img_pub.publish(img_msg)


def main():
    rclpy.init(args=None)
    yolo_subscriber = YoloSubscriber()
    camera_subscriber = CameraSubscriber()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(yolo_subscriber)
    executor.add_node(camera_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        while rclpy.ok():
            pass
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    executor_thread.join()


if __name__ == "__main__":
    main()