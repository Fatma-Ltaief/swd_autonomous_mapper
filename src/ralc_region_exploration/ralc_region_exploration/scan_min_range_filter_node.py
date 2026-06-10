#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanMinRangeFilter(Node):
    def __init__(self):
        super().__init__("scan_min_range_filter")

        self.declare_parameter("input_scan_topic", "/scan")
        self.declare_parameter("output_scan_topic", "/scan_filtered")
        self.declare_parameter("min_valid_range", 0.25)

        self.input_scan_topic = self.get_parameter("input_scan_topic").value
        self.output_scan_topic = self.get_parameter("output_scan_topic").value
        self.min_valid_range = float(self.get_parameter("min_valid_range").value)

        self.pub = self.create_publisher(
            LaserScan,
            self.output_scan_topic,
            qos_profile_sensor_data,
        )

        self.sub = self.create_subscription(
            LaserScan,
            self.input_scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"Filtering {self.input_scan_topic} -> {self.output_scan_topic}, "
            f"removing finite ranges < {self.min_valid_range:.3f} m"
        )

    def scan_callback(self, msg: LaserScan):
        out = LaserScan()
        out.header = msg.header

        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time

        out.range_min = max(float(msg.range_min), self.min_valid_range)
        out.range_max = msg.range_max

        filtered_ranges = []
        removed = 0

        for r in msg.ranges:
            if math.isfinite(r) and 0.0 < r < self.min_valid_range:
                filtered_ranges.append(float("inf"))
                removed += 1
            else:
                filtered_ranges.append(r)

        out.ranges = filtered_ranges
        out.intensities = msg.intensities

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ScanMinRangeFilter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
