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
        self.declare_parameter('reverse_speed', -0.04)
        self.declare_parameter('angular_speed', 0.35)
        self.declare_parameter('steering_gain', 0.20)
        self.declare_parameter('max_steering_speed', 0.15)
        self.declare_parameter('stuck_turn_time', 5.0)
        self.declare_parameter('reverse_time', 1.0)

        scan_topic = self.get_parameter('scan_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.front_obstacle_distance = float(
            self.get_parameter('front_obstacle_distance').value)
        self.forward_speed = float(self.get_parameter('forward_speed').value)
        self.reverse_speed = float(self.get_parameter('reverse_speed').value)
        self.angular_speed = float(self.get_parameter('angular_speed').value)
        self.steering_gain = float(self.get_parameter('steering_gain').value)
        self.max_steering_speed = float(
            self.get_parameter('max_steering_speed').value)
        self.stuck_turn_time = float(self.get_parameter('stuck_turn_time').value)
        self.reverse_time = float(self.get_parameter('reverse_time').value)

        self.latest_scan = None
        self.last_log_time = self.get_clock().now()
        self.state = 'waiting_for_scan'
        self.turn_started_at = None
        self.turn_direction = 1.0
        self.reverse_started_at = None
        self.recovery_turn_direction = -1.0

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
        sectors = self.scan_sectors(self.latest_scan)
        front_min = sectors['front']
        left_min = sectors['left']
        right_min = sectors['right']

        if self.state == 'reversing':
            if (
                self.seconds_since(now, self.reverse_started_at)
                < self.reverse_time
            ):
                self.publish_reverse()
                self.log_status(now, front_min, left_min, right_min, 'reverse')
                return

            self.turn_started_at = now
            self.turn_direction = self.recovery_turn_direction
            self.reverse_started_at = None
            self.set_state('turning')
            action = 'turn_left' if self.turn_direction > 0.0 else 'turn_right'
            self.publish_turn(self.turn_direction)
            self.log_status(now, front_min, left_min, right_min, action)
            return

        if front_min < self.front_obstacle_distance:
            if self.state != 'turning':
                self.turn_direction = self.choose_turn_direction(
                    sectors['front_left'],
                    sectors['front_right'],
                    left_min,
                    right_min,
                )
                self.turn_started_at = now

            elif (
                self.seconds_since(now, self.turn_started_at)
                > self.stuck_turn_time
            ):
                self.recovery_turn_direction = -self.turn_direction
                self.reverse_started_at = now
                self.turn_started_at = None
                self.set_state('reversing')
                self.publish_reverse()
                self.log_status(now, front_min, left_min, right_min, 'reverse')
                return

            self.set_state('turning')
            action = 'turn_left' if self.turn_direction > 0.0 else 'turn_right'
            self.publish_turn(self.turn_direction)
            self.log_status(now, front_min, left_min, right_min, action)
            return

        self.turn_started_at = None
        self.reverse_started_at = None
        self.set_state('moving')
        angular_z, action = self.steering_correction(left_min, right_min)
        self.publish_forward(angular_z)
        self.log_status(now, front_min, left_min, right_min, action)

    def scan_sectors(self, scan):
        return {
            'front': self.min_range_in_sector(
                scan, math.radians(-20.0), math.radians(20.0)),
            'front_left': self.min_range_in_sector(
                scan, math.radians(20.0), math.radians(70.0)),
            'front_right': self.min_range_in_sector(
                scan, math.radians(-70.0), math.radians(-20.0)),
            'left': self.min_range_in_sector(
                scan, math.radians(70.0), math.radians(120.0)),
            'right': self.min_range_in_sector(
                scan, math.radians(-120.0), math.radians(-70.0)),
        }

    def min_range_in_sector(self, scan, min_angle, max_angle):
        best = math.inf

        for index, distance in enumerate(scan.ranges):
            if not math.isfinite(distance):
                continue
            if distance < scan.range_min or distance > scan.range_max:
                continue

            angle = self.normalize_angle(
                scan.angle_min + index * scan.angle_increment)
            if min_angle <= angle <= max_angle:
                best = min(best, distance)

        return best

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def choose_turn_direction(self, front_left_min, front_right_min, left_min,
                              right_min):
        left_space = self.closest_valid(front_left_min, left_min)
        right_space = self.closest_valid(front_right_min, right_min)

        if math.isfinite(left_space) and math.isfinite(right_space):
            return 1.0 if left_space >= right_space else -1.0
        if math.isfinite(left_space):
            return 1.0
        if math.isfinite(right_space):
            return -1.0
        return 1.0

    def closest_valid(self, *distances):
        valid = [distance for distance in distances if math.isfinite(distance)]
        if not valid:
            return math.inf
        return min(valid)

    def steering_correction(self, left_min, right_min):
        if not math.isfinite(left_min) or not math.isfinite(right_min):
            return 0.0, 'forward'

        clearance_error = left_min - right_min
        angular_z = self.steering_gain * clearance_error
        angular_z = max(
            -self.max_steering_speed,
            min(self.max_steering_speed, angular_z)
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
        twist.angular.z = turn_direction * self.angular_speed
        self.cmd_pub.publish(twist)

    def publish_reverse(self):
        twist = Twist()
        twist.linear.x = self.reverse_speed
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def seconds_since(self, now, start_time):
        if start_time is None:
            return 0.0
        return (now - start_time).nanoseconds / 1_000_000_000.0

    def set_state(self, state):
        if state == self.state:
            return
        self.state = state

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
