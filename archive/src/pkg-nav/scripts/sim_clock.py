#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from time import sleep

class ClockPublisher(Node):
    def __init__(self):
        super().__init__('clock_publisher')
        self.publisher_ = self.create_publisher(Clock, '/clock', 10)
        self.declare_parameter('start_time', 1749833584)
        self.declare_parameter('rate', 100.0)

        self.time = self.get_parameter('start_time').get_parameter_value().double_value
        self.rate = self.get_parameter('rate').get_parameter_value().double_value

    def start(self):
        period = 1.0 / self.rate
        while rclpy.ok():
            msg = Clock()
            msg.clock.sec = int(self.time)
            msg.clock.nanosec = int((self.time - int(self.time)) * 1e9)
            self.publisher_.publish(msg)
            self.time += period
            sleep(period)

def main(args=None):
    rclpy.init(args=args)
    clock_publisher = ClockPublisher()
    clock_publisher.start()
    clock_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
