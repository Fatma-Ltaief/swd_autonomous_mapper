#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


class WanderAvoidNode(Node):
    def __init__(self):
        super().__init__('wander_avoid')

        # Tunable parameters
        self.declare_parameter('forward_speed', 0.12)
        self.declare_parameter('slow_speed', 0.05)
        self.declare_parameter('rotation_speed', 0.6)

        self.declare_parameter('front_stop_distance', 0.75)
        self.declare_parameter('front_slow_distance', 1.10)
        self.declare_parameter('side_stop_distance', 0.45)

        self.declare_parameter('front_sector_deg', 35.0)   # +/- around front
        self.declare_parameter('side_sector_deg', 70.0)    # lateral guard sectors
        self.declare_parameter('publish_period', 0.1)

        self.forward_speed = float(self.get_parameter('forward_speed').value)
        self.slow_speed = float(self.get_parameter('slow_speed').value)
        self.rotation_speed = float(self.get_parameter('rotation_speed').value)

        self.front_stop_distance = float(self.get_parameter('front_stop_distance').value)
        self.front_slow_distance = float(self.get_parameter('front_slow_distance').value)
        self.side_stop_distance = float(self.get_parameter('side_stop_distance').value)

        self.front_sector_deg = float(self.get_parameter('front_sector_deg').value)
        self.side_sector_deg = float(self.get_parameter('side_sector_deg').value)
        self.publish_period = float(self.get_parameter('publish_period').value)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        self.state = 'FORWARD'   # FORWARD / SLOW / ROTATE
        self.rotate_direction = 1.0  # +1 left, -1 right

        self.last_front_min = float('inf')
        self.last_left_min = float('inf')
        self.last_right_min = float('inf')

        self.cmd_timer = self.create_timer(self.publish_period, self.publish_cmd)

        self.get_logger().info('Safer WanderAvoid node started.')

    def _valid_ranges_in_sector(self, msg: LaserScan, min_angle: float, max_angle: float):
        vals = []
        angle = msg.angle_min

        for d in msg.ranges:
            if min_angle <= angle <= max_angle:
                if math.isfinite(d) and msg.range_min < d < msg.range_max:
                    vals.append(d)
            angle += msg.angle_increment

        return vals

    def _sector_min(self, msg: LaserScan, min_angle: float, max_angle: float):
        vals = self._valid_ranges_in_sector(msg, min_angle, max_angle)
        return min(vals) if vals else float('inf')

    def scan_callback(self, msg: LaserScan):
        front_rad = math.radians(self.front_sector_deg)
        side_rad = math.radians(self.side_sector_deg)

        # Front sector centered at 0
        front_min = self._sector_min(msg, -front_rad, front_rad)

        # Left guard: from front-left to left side
        left_min = self._sector_min(msg, front_rad, side_rad)

        # Right guard: from right side to front-right
        right_min = self._sector_min(msg, -side_rad, -front_rad)

        self.last_front_min = front_min
        self.last_left_min = left_min
        self.last_right_min = right_min

        obstacle_front_close = front_min < self.front_stop_distance
        obstacle_side_close = (left_min < self.side_stop_distance) or (right_min < self.side_stop_distance)
        obstacle_front_near = front_min < self.front_slow_distance

        previous_state = self.state

        if obstacle_front_close or obstacle_side_close:
            self.state = 'ROTATE'

            # Turn toward the freer side
            # If left side has more clearance, rotate left (+)
            # else rotate right (-)
            self.rotate_direction = 1.0 if left_min > right_min else -1.0

        elif obstacle_front_near:
            self.state = 'SLOW'

            # Gentle steering away from the closer side
            # if right side is tighter, steer left, and vice versa
            self.rotate_direction = 1.0 if left_min > right_min else -1.0

        else:
            self.state = 'FORWARD'

        if self.state != previous_state:
            self.get_logger().info(
                f"State {previous_state} -> {self.state} | "
                f"front={front_min:.2f} m, left={left_min:.2f} m, right={right_min:.2f} m"
            )

    def publish_cmd(self):
        cmd = Twist()

        if self.state == 'FORWARD':
            cmd.linear.x = self.forward_speed
            cmd.angular.z = 0.0

        elif self.state == 'SLOW':
            cmd.linear.x = self.slow_speed
            cmd.angular.z = 0.35 * self.rotate_direction

        elif self.state == 'ROTATE':
            cmd.linear.x = 0.0
            cmd.angular.z = self.rotation_speed * self.rotate_direction

        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = WanderAvoidNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()