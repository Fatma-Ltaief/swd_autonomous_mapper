#!/usr/bin/env python3

import argparse
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class StopRobot(Node):
    def __init__(self, topics, count, period):
        super().__init__('stop_robot')
        self.publishers = [
            self.create_publisher(Twist, topic, 10)
            for topic in topics
        ]
        self.count = count
        self.period = period

    def stop(self):
        msg = Twist()
        for _ in range(self.count):
            for publisher in self.publishers:
                publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(self.period)
        self.get_logger().info(f'Published {self.count} zero Twist messages.')


def main():
    parser = argparse.ArgumentParser(
        description='Publish repeated zero velocity commands to stop the robot.')
    parser.add_argument(
        '--topics',
        nargs='+',
        default=['/cmd_vel', '/cmd_vel_smoothed', '/cmd_vel_nav', '/cmd_vel_nav2'])
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--period', type=float, default=0.1)
    args = parser.parse_args()

    rclpy.init()
    node = StopRobot(args.topics, args.count, args.period)
    node.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
