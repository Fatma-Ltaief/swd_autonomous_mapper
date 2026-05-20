import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class WanderNode(Node):
    def __init__(self):
        super().__init__('wander_node')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('front_obstacle_distance', 0.45)
        self.declare_parameter('forward_speed', 0.08)
        self.declare_parameter('turn_speed', 0.45)
        self.declare_parameter('front_sector_degrees', 15.0)
        self.declare_parameter('centering_gain', 0.25)
        self.declare_parameter('max_centering_turn', 0.18)

        scan_topic = self.get_parameter('scan_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.front_obstacle_distance = float(
            self.get_parameter('front_obstacle_distance').value)
        self.forward_speed = float(self.get_parameter('forward_speed').value)
        self.turn_speed = float(self.get_parameter('turn_speed').value)
        self.front_sector = math.radians(
            float(self.get_parameter('front_sector_degrees').value))
        self.centering_gain = float(self.get_parameter('centering_gain').value)
        self.max_centering_turn = float(
            self.get_parameter('max_centering_turn').value)

        self.latest_scan = None
        self.last_log_time = self.get_clock().now()
        self.state = 'waiting_for_scan'

        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            f'Wander node started: scan={scan_topic}, cmd_vel={cmd_vel_topic}')

    def scan_callback(self, msg):
        self.latest_scan = msg

    def timer_callback(self):
        if self.latest_scan is None:
            self.publish_stop()
            self.set_state('waiting_for_scan')
            return

        now = self.get_clock().now()
        front_min = self.min_range_in_sector(
            self.latest_scan, -self.front_sector, self.front_sector)
        left_min = self.min_range_in_sector(
            self.latest_scan, math.radians(60.0), math.radians(120.0))
        right_min = self.min_range_in_sector(
            self.latest_scan, math.radians(-120.0), math.radians(-60.0))

        if front_min < self.front_obstacle_distance:
            turn_direction = self.choose_turn_direction(left_min, right_min)
            self.set_state('turning')
            action = 'turn_left' if turn_direction > 0.0 else 'turn_right'
            self.publish_turn(turn_direction)
            self.log_status(now, front_min, left_min, right_min, action)
            return

        self.set_state('moving')
        angular_z, action = self.centering_turn(left_min, right_min)
        self.publish_forward(angular_z)
        self.log_status(now, front_min, left_min, right_min, action)

    def min_range_in_sector(self, scan, min_angle, max_angle):
        best = math.inf

        for index, distance in enumerate(scan.ranges):
            if not math.isfinite(distance):
                continue
            if distance < scan.range_min or distance > scan.range_max:
                continue

            angle = scan.angle_min + index * scan.angle_increment
            if min_angle <= angle <= max_angle:
                best = min(best, distance)

        return best

    def choose_turn_direction(self, left_min, right_min):
        if math.isfinite(left_min) and math.isfinite(right_min):
            return 1.0 if left_min >= right_min else -1.0
        if math.isfinite(left_min):
            return 1.0
        if math.isfinite(right_min):
            return -1.0
        return 1.0

    def centering_turn(self, left_min, right_min):
        if not math.isfinite(left_min) or not math.isfinite(right_min):
            return 0.0, 'forward'

        clearance_error = left_min - right_min
        angular_z = self.centering_gain * clearance_error
        angular_z = max(
            -self.max_centering_turn,
            min(self.max_centering_turn, angular_z)
        )

        if abs(angular_z) < 0.03:
            return 0.0, 'forward'
        if angular_z > 0.0:
            return angular_z, 'forward_left'
        return angular_z, 'forward_right'

    def publish_forward(self, angular_z):
        twist = Twist()
        twist.linear.x = self.forward_speed
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)

    def publish_turn(self, turn_direction):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = turn_direction * self.turn_speed
        self.cmd_pub.publish(twist)

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def set_state(self, state):
        if state == self.state:
            return
        self.state = state
        self.get_logger().info(f'Wander state: {state}')

    def log_status(self, now, front_min, left_min, right_min, action):
        if (now - self.last_log_time).nanoseconds < 1_000_000_000:
            return

        self.last_log_time = now
        self.get_logger().info(
            'front_min=%s m, left_min=%s m, right_min=%s m, action=%s' % (
                self.format_distance(front_min),
                self.format_distance(left_min),
                self.format_distance(right_min),
                action,
            )
        )

    def format_distance(self, distance):
        if not math.isfinite(distance):
            return 'inf'
        return f'{distance:.2f}'


def main(args=None):
    rclpy.init(args=args)
    node = WanderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
