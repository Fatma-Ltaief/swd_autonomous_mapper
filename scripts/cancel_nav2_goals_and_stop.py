#!/usr/bin/env python3

import argparse
import time

import rclpy
from action_msgs.msg import GoalInfo
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import Twist
from rclpy.node import Node


DEFAULT_ACTIONS = (
    '/follow_waypoints',
    '/navigate_to_pose',
    '/navigate_through_poses',
)

DEFAULT_CMD_TOPICS = (
    '/cmd_vel',
    '/cmd_vel_smoothed',
    '/cmd_vel_nav',
    '/cmd_vel_nav2',
)


class Nav2SafeStop(Node):
    def __init__(self, actions, cmd_topics, count, period, service_timeout):
        super().__init__('cancel_nav2_goals_and_stop')
        self.actions = actions
        self.count = count
        self.period = period
        self.service_timeout = service_timeout
        self.publishers = [
            self.create_publisher(Twist, topic, 10)
            for topic in cmd_topics
        ]

    def cancel_goals(self):
        request = CancelGoal.Request()
        request.goal_info = GoalInfo()

        for action_name in self.actions:
            service_name = f'{action_name}/_action/cancel_goal'
            client = self.create_client(CancelGoal, service_name)
            if not client.wait_for_service(timeout_sec=self.service_timeout):
                self.get_logger().warn(f'Cancel service not available: {service_name}')
                continue

            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=self.service_timeout)
            if future.done() and future.result() is not None:
                goals = len(future.result().goals_canceling)
                self.get_logger().info(f'Requested cancel on {action_name}; goals canceling: {goals}')
            else:
                self.get_logger().warn(f'No cancel response from {service_name}')

    def publish_stop(self):
        msg = Twist()
        for _ in range(self.count):
            for publisher in self.publishers:
                publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(self.period)
        self.get_logger().info('Nav2 goals canceled and zero velocity published repeatedly.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Cancel active Nav2 actions and repeatedly publish zero velocity.')
    parser.add_argument('--actions', nargs='+', default=list(DEFAULT_ACTIONS))
    parser.add_argument('--cmd-topics', nargs='+', default=list(DEFAULT_CMD_TOPICS))
    parser.add_argument('--count', type=int, default=30)
    parser.add_argument('--period', type=float, default=0.1)
    parser.add_argument('--service-timeout', type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = Nav2SafeStop(
        actions=args.actions,
        cmd_topics=args.cmd_topics,
        count=args.count,
        period=args.period,
        service_timeout=args.service_timeout)
    node.cancel_goals()
    node.publish_stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
