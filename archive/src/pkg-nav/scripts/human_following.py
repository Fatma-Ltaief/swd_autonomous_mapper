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

class HumanDetection(Node):

    def __init__(self):
        super().__init__('HumanDetection')

        goal_topic_name = args.goal_topic

        self.agents_subscriber = self.create_subscription(AgentArray, 'agents', self.agents_callback, 10)
        self.goal_publisher_ = self.create_publisher(PoseStamped, goal_topic_name, 1)
        self.goal_marker_publisher_ = self.create_publisher(Marker, goal_topic_name + '/marker', 1)
        
        time.sleep(float(args.time_wait))

        print("Start Node...")

    def agents_callback(self, agents_msg):
        if len(agents_msg.agents) <= 0:
            return

        agent = agents_msg.agents[0]

        x_agent = agent.pose.position.x
        y_agent = agent.pose.position.y
        z_agent = agent.pose.position.z

        x_orientation_agent = agent.pose.orientation.x
        y_orientation_agent = agent.pose.orientation.y
        z_orientation_agent = agent.pose.orientation.z
        w_orientation_agent = agent.pose.orientation.w
        
        r = R.from_quat([x_orientation_agent, y_orientation_agent, z_orientation_agent, w_orientation_agent])

        x, y, z = np.array([x_agent, y_agent, z_agent]) - r.apply([DISTANCE_ROBOT_HUMAN, 0, 0])

        
        msg_goal_marker = Marker()
        msg_goal_marker.header.stamp = self.get_clock().now().to_msg()
        msg_goal_marker.header.frame_id = "camera_link"

        msg_goal_marker.id = 100
        msg_goal_marker.type = msg_goal_marker.SPHERE

        msg_goal_marker.color.r = 1.0
        msg_goal_marker.color.g = 0.0
        msg_goal_marker.color.b = 0.0
        msg_goal_marker.color.a = 1.0

        msg_goal_marker.scale.x = 0.1
        msg_goal_marker.scale.y = 0.1
        msg_goal_marker.scale.z = 0.1

        msg_goal_marker.pose.position.x = x
        msg_goal_marker.pose.position.y = y

        self.goal_marker_publisher_.publish(msg_goal_marker)


        msg_goal = PoseStamped()
        msg_goal.header.stamp = self.get_clock().now().to_msg()
        msg_goal.header.frame_id = "camera_link"

        msg_goal.pose.position.x = float(x)
        msg_goal.pose.position.y = float(y)

        self.goal_publisher_.publish(msg_goal)
        


def main(args=None):
    rclpy.init(args=args)

    human_detection_publisher = HumanDetection()
    rclpy.spin(human_detection_publisher)

    human_detection_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
