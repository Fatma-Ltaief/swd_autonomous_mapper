#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from builtin_interfaces.msg import Time


class VelocityMultiplexer(Node):
    def __init__(self):
        super().__init__('multiplexer')

        # Déclaration des paramètres
        self.declare_parameter('cmd_joystick_topic', '/cmd_vel_joystick')
        self.declare_parameter('cmd_nav_topic', '/cmd_vel_nav')
        self.declare_parameter('cmd_output_topic', '/cmd_vel')
        self.declare_parameter('laser_topic', '/scan')
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('safety_enabled', True)
        self.declare_parameter('safety_resume_delay', 2.0)  # secondes
        self.declare_parameter('velocity_scale_with_obstacle', 0.5)

        joystick_topic = self.get_parameter('cmd_joystick_topic').get_parameter_value().string_value
        nav_topic = self.get_parameter('cmd_nav_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('cmd_output_topic').get_parameter_value().string_value
        laser_topic = self.get_parameter('laser_topic').get_parameter_value().string_value
        self.safety_distance = self.get_parameter('safety_distance').get_parameter_value().double_value
        self.safety_enabled = self.get_parameter('safety_enabled').get_parameter_value().bool_value
        self.safety_resume_delay = self.get_parameter('safety_resume_delay').get_parameter_value().double_value
        self.velocity_scale = self.get_parameter('velocity_scale_with_obstacle').get_parameter_value().double_value

        # Publications & abonnements
        self.cmd_vel_pub = self.create_publisher(Twist, output_topic, 10)
        self.joystick_sub = self.create_subscription(Twist, joystick_topic, self.joystick_callback, 10)
        self.nav_sub = self.create_subscription(Twist, nav_topic, self.nav_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, laser_topic, self.laser_callback, 10)

        self.last_joystick_time = None
        self.timeout = 0.5

        self.obstacle_detected = False
        self.obstacle_since = None

    def laser_callback(self, msg: LaserScan):
        valid_ranges = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if not valid_ranges:
            return

        min_distance = min(valid_ranges)

        if min_distance < self.safety_distance:
            if not self.obstacle_detected:
                self.get_logger().warn(f"[SÉCURITÉ] Obstacle détecté à {min_distance:.2f} m — arrêt.")
                self.obstacle_since = self.get_clock().now()
            self.obstacle_detected = True
        else:
            if self.obstacle_detected:
                self.get_logger().info("[SÉCURITÉ] Zone dégagée — reprise normale.")
            self.obstacle_detected = False
            self.obstacle_since = None

    def joystick_callback(self, msg):
        self.last_joystick_time = self.get_clock().now()
        self.publish_with_safety(msg, source="joystick")

    def nav_callback(self, msg):
        now = self.get_clock().now()
        if self.last_joystick_time:
            dt = (now - self.last_joystick_time).nanoseconds * 1e-9
            if dt < self.timeout:
                self.get_logger().debug("Joystick prioritaire — nav ignorée.")
                return
        self.publish_with_safety(msg, source="nav")

    def publish_with_safety(self, msg: Twist, source=""):
        if not self.safety_enabled or not self.obstacle_detected:
            self.cmd_vel_pub.publish(msg)
            self.get_logger().debug(f"{source} command published (normal).")
            return

        now = self.get_clock().now()
        elapsed = (now - self.obstacle_since).nanoseconds * 1e-9 if self.obstacle_since else 0.0

        if elapsed < self.safety_resume_delay:
            self.stop_robot(msg)
            self.get_logger().debug(f"[SÉCURITÉ] {source} bloqué (attente de {self.safety_resume_delay:.1f}s).")
        else:
            reduced_msg = self.scale_twist(msg, self.velocity_scale)
            self.cmd_vel_pub.publish(reduced_msg)
            self.get_logger().warn(f"[SÉCURITÉ] {source} ralentie à {self.velocity_scale*100:.0f}%")

    def stop_robot(self, msg: Twist):
        scaled = Twist()
        scaled.linear.x = 0.0
        scaled.linear.y = 0.0
        scaled.linear.z = 0.0
        scaled.angular.x = msg.angular.x
        scaled.angular.y = msg.angular.y
        scaled.angular.z = msg.angular.z
        self.cmd_vel_pub.publish(scaled)

    def scale_twist(self, msg: Twist, factor: float) -> Twist:
        scaled = Twist()
        scaled.linear.x = msg.linear.x * factor
        scaled.linear.y = msg.linear.y * factor
        scaled.linear.z = msg.linear.z * factor
        scaled.angular.x = msg.angular.x
        scaled.angular.y = msg.angular.y
        scaled.angular.z = msg.angular.z
        return scaled


def main(args=None):
    rclpy.init(args=args)
    node = VelocityMultiplexer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
