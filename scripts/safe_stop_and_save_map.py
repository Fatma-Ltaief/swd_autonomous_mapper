#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

from cancel_nav2_goals_and_stop import DEFAULT_ACTIONS
from cancel_nav2_goals_and_stop import DEFAULT_CMD_TOPICS
from cancel_nav2_goals_and_stop import Nav2SafeStop
import rclpy


def parse_args():
    parser = argparse.ArgumentParser(
        description='Cancel exploration/Nav2 goals, stop the robot, and save the SLAM map.')
    parser.add_argument('--map', default='~/autonomous_slam_map')
    parser.add_argument('--actions', nargs='+', default=list(DEFAULT_ACTIONS))
    parser.add_argument('--cmd-topics', nargs='+', default=list(DEFAULT_CMD_TOPICS))
    parser.add_argument('--count', type=int, default=30)
    parser.add_argument('--period', type=float, default=0.1)
    parser.add_argument('--service-timeout', type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    map_path = os.path.expanduser(args.map)

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

    result = subprocess.run(
        ['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', map_path],
        check=False)
    if result.returncode != 0:
        print('Map save failed. SLAM was left running.', file=sys.stderr)
        return result.returncode

    print(f'Map saved to {map_path}. SLAM was left running.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
