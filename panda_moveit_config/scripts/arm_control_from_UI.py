#!/usr/bin/env python3

import time
import threading
import math

import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray

from control_msgs.action import GripperCommand

from moveit.planning import MoveItPy


def quat_from_euler(roll: float, pitch: float, yaw: float):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def plan_and_execute(robot: MoveItPy, planning_component, logger, sleep_time: float = 0.0, **kwargs) -> bool:
    logger.info("Planning trajectory")
    plan_result = planning_component.plan(**kwargs) if kwargs else planning_component.plan()
    if not plan_result:
        logger.error("Planning failed")
        return False

    logger.info("Executing plan")
    try:
        robot.execute(plan_result.trajectory, controllers=[])
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return False

    if sleep_time > 0.0:
        time.sleep(sleep_time)
    return True


class Controller(Node):
    def __init__(self):
        super().__init__("commander")

        self.subscription = self.create_subscription(
            Float64MultiArray, "/target_point", self.listener_callback, 10
        )

        self.pose_goal = PoseStamped()
        self.pose_goal.header.frame_id = "panda_link0"

        # MoveIt for ARM only
        self.panda = MoveItPy(node_name="moveit_py_core")
        self.panda_arm = self.panda.get_planning_component("panda_arm")
        self.logger = get_logger("moveit_py.pose_goal")

        # Gripper action client
        self.gripper_client = ActionClient(self, GripperCommand, "/panda_hand_controller/gripper_cmd")

        # Banana pick parameters
        self.pregrasp_height = 0.20
        self.pick_height = 0.105
        self.carrying_height = 0.30

        self.grasp_open = 0.04     # meters
        self.grasp_close = 0.012   # meters (close enough; you can also try 0.0)

        # NEW: effort settings (based on your successful terminal test)
        self.open_effort = 100.0
        self.close_effort = 200.0

        # Orientation: top-down + yaw from UI
        self.roll_offset = math.pi
        self.pitch_offset = 0.0
        self.yaw_offset = 0.0

        self._busy_lock = threading.Lock()

    def move_to(self, x: float, y: float, z: float, yaw: float = 0.0) -> bool:
        self.panda_arm.set_start_state_to_current_state()

        self.pose_goal.header.stamp = self.get_clock().now().to_msg()
        self.pose_goal.pose.position.x = float(x)
        self.pose_goal.pose.position.y = float(y)
        self.pose_goal.pose.position.z = float(z)

        qx, qy, qz, qw = quat_from_euler(
            self.roll_offset,
            self.pitch_offset,
            float(yaw) + self.yaw_offset,
        )

        self.pose_goal.pose.orientation.x = float(qx)
        self.pose_goal.pose.orientation.y = float(qy)
        self.pose_goal.pose.orientation.z = float(qz)
        self.pose_goal.pose.orientation.w = float(qw)

        self.panda_arm.set_goal_state(pose_stamped_msg=self.pose_goal, pose_link="panda_link8")
        return plan_and_execute(self.panda, self.panda_arm, self.logger, sleep_time=0.2)

    def gripper_command(self, width: float, max_effort: float, timeout_sec: float = 10.0) -> bool:
        """
        Send gripper width command via action server.

        - No rclpy.spin_until_future_complete() here because the node is already being spun
          by MultiThreadedExecutor in another thread.
        - Poll futures until done or timeout.
        """
        width = max(0.0, min(float(width), 0.04))

        if not self.gripper_client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error("Gripper action server not available: /panda_hand_controller/gripper_cmd")
            return False

        goal = GripperCommand.Goal()
        goal.command.position = width
        goal.command.max_effort = float(max_effort)

        send_future = self.gripper_client.send_goal_async(goal)

        # Wait for goal response safely (no spinning here)
        t0 = time.time()
        while rclpy.ok() and not send_future.done():
            if time.time() - t0 > timeout_sec:
                self.get_logger().error("Timeout waiting for gripper goal response.")
                return False
            time.sleep(0.01)

        goal_handle = send_future.result()
        if goal_handle is None:
            self.get_logger().error("No goal handle returned (timeout/communication issue).")
            return False

        if not goal_handle.accepted:
            self.get_logger().error("Gripper goal rejected by server.")
            return False

        # Wait for result (optional)
        result_future = goal_handle.get_result_async()
        t0 = time.time()
        while rclpy.ok() and not result_future.done():
            if time.time() - t0 > timeout_sec:
                self.get_logger().warning("Timeout waiting for gripper result; gripper may still have moved.")
                return True
            time.sleep(0.01)

        return True

    def listener_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warning("target_point must contain at least [x, y] (and optionally yaw)")
            return

        x, y = float(msg.data[0]), float(msg.data[1])
        yaw = float(msg.data[2]) if len(msg.data) >= 3 else 0.0

        if not self._busy_lock.acquire(blocking=False):
            self.get_logger().warning("Robot is busy; ignoring new target.")
            return

        try:
            self.get_logger().info(f"New target: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f} rad")

            # Pregrasp
            self.move_to(x, y, self.pregrasp_height, yaw=yaw)
            self.gripper_command(self.grasp_open, max_effort=self.open_effort)

            # Approach
            self.move_to(x, y, self.pick_height, yaw=yaw)
            time.sleep(0.3)

            # Close (HIGH effort)
            self.gripper_command(self.grasp_close, max_effort=self.close_effort)
            time.sleep(0.2)

            # Lift
            self.move_to(x, y, self.pregrasp_height, yaw=yaw)
            self.move_to(x, y, self.carrying_height, yaw=yaw)

            # Place
            self.move_to(0.3, -0.3, self.carrying_height, yaw=yaw)
            self.gripper_command(self.grasp_open, max_effort=self.open_effort)

        finally:
            self._busy_lock.release()


def main():
    rclpy.init(args=None)

    controller = Controller()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(controller)
    threading.Thread(target=executor.spin, daemon=True).start()

    try:
        while rclpy.ok():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()