#!/usr/bin/env python3

import sys
import math
import time
import argparse
import cv2
import rclpy
from rclpy.node import Node
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from agents_msgs.msg import Agent, AgentArray
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

DISTANCE_ROBOT_HUMAN = 0.0
BASE_TIME_WAIT = 0.0

parser = argparse.ArgumentParser(description='Follow Demo')
parser.add_argument('--goal_topic', dest='goal_topic',
                    help='goal topic name', default="/goal")
parser.add_argument('--time_wait', dest='time_wait',
                    help='time to wait before start node', default=0.0)

args, unknown = parser.parse_known_args()

class GoalDelay(Node):

    def __init__(self):
        super().__init__('HumanDetection')

        goal_topic_name = args.goal_topic

        self.goal_subscriber = self.create_subscription(PoseStamped, 'goal', self.goal_callback, 10)
        self.goal_publisher_ = self.create_publisher(PoseStamped, goal_topic_name, 1)
        
        self.time_delay = float(args.time_wait)

        print("Start Node...")

    def goal_callback(self, goal_msg):
        print('Hey')
        time.sleep(self.time_delay)
        print("Publish Goal ...")
        self.goal_publisher_.publish(goal_msg)


def main(args=None):
    rclpy.init(args=args)

    goal_delay_publisher = GoalDelay()
    rclpy.spin(goal_delay_publisher)

    goal_delay_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
