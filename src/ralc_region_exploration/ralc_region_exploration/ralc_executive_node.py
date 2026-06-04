import json
import time
from typing import List, Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseArray, PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker


class RalcExecutive(Node):
    """Robot motion executor only.

    It accepts pose goals and waypoint sequences from the exploration manager,
    sends them to Nav2 NavigateToPose, and publishes a JSON execution result.
    It does not decide when to run ALC, refinement, or region transitions.
    """

    IDLE = 'IDLE'
    EXECUTING_POSE = 'EXECUTING_POSE'
    EXECUTING_WAYPOINTS = 'EXECUTING_WAYPOINTS'

    def __init__(self):
        super().__init__('ralc_executive')

        self.declare_parameter('enable_motion', True)
        self.enable_motion = bool(self.get_parameter('enable_motion').value)

        self.state = self.IDLE
        self.current_source = ''
        self.active_goal: Optional[PoseStamped] = None
        self.goal_start_time: Optional[float] = None
        self.waypoints: List[PoseStamped] = []
        self.waypoint_index = 0

        self.nav_to_pose_client = ActionClient(
            self, NavigateToPose, '/navigate_to_pose'
        )
        self.create_subscription(
            PoseStamped,
            '/ralc/execute_pose_goal',
            self.execute_pose_goal_callback,
            10,
        )
        self.create_subscription(
            PoseArray,
            '/ralc/execute_waypoint_sequence',
            self.execute_waypoint_sequence_callback,
            10,
        )

        self.result_pub = self.create_publisher(String, '/ralc/execution_result', 10)
        self.failed_marker_pub = self.create_publisher(
            Marker, '/ralc/failed_goal_marker', 10
        )

        self.get_logger().info(
            '[RALC] Executive ready: /ralc/execute_pose_goal, '
            '/ralc/execute_waypoint_sequence -> Nav2 NavigateToPose.'
        )

    def execute_pose_goal_callback(self, msg: PoseStamped):
        if not self.enable_motion:
            self.publish_result('pose', True, 'motion disabled')
            return
        if self.state != self.IDLE:
            self.get_logger().warn('[RALC] Executive busy; dropping pose goal.')
            return
        self.current_source = 'pose'
        self.state = self.EXECUTING_POSE
        self.send_goal(msg)

    def execute_waypoint_sequence_callback(self, msg: PoseArray):
        if not self.enable_motion:
            self.publish_result('waypoints', True, 'motion disabled')
            return
        if self.state != self.IDLE:
            self.get_logger().warn('[RALC] Executive busy; dropping waypoint sequence.')
            return
        self.waypoints = []
        for pose in msg.poses:
            waypoint = PoseStamped()
            waypoint.header = msg.header
            waypoint.pose = pose
            self.waypoints.append(waypoint)
        if not self.waypoints:
            self.publish_result('waypoints', False, 'empty waypoint sequence')
            return
        self.current_source = 'waypoints'
        self.state = self.EXECUTING_WAYPOINTS
        self.waypoint_index = 0
        self.send_goal(self.waypoints[self.waypoint_index])

    def send_goal(self, goal_pose: PoseStamped):
        if not self.nav_to_pose_client.server_is_ready():
            if not self.nav_to_pose_client.wait_for_server(timeout_sec=2.0):
                self.active_goal = goal_pose
                self.publish_result(
                    self.current_source,
                    False,
                    'Nav2 unavailable',
                    failure_type='NAV2_FAILED',
                )
                self.reset()
                return
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        self.active_goal = goal_pose
        self.goal_start_time = time.monotonic()
        self.get_logger().info(
            f'[RALC] Sending Nav2 goal source={self.current_source}: '
            f'x={goal_pose.pose.position.x:.2f}, y={goal_pose.pose.position.y:.2f}'
        )
        future = self.nav_to_pose_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.publish_result(
                self.current_source,
                False,
                'Nav2 rejected goal',
                failure_type='NAV2_FAILED',
            )
            self.publish_failed_goal_marker()
            self.reset()
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        result = future.result()
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            message, failure_type = self.describe_nav2_failure(result)
            self.publish_result(
                self.current_source,
                False,
                message,
                failure_type=failure_type,
            )
            self.publish_failed_goal_marker()
            self.reset()
            return
        if self.state == self.EXECUTING_WAYPOINTS:
            self.waypoint_index += 1
            if self.waypoint_index < len(self.waypoints):
                self.send_goal(self.waypoints[self.waypoint_index])
                return
        self.publish_result(self.current_source, True, 'succeeded')
        self.reset()

    def describe_nav2_failure(self, result):
        result_payload = getattr(result, 'result', None)
        detail_parts = [f'Nav2 status {result.status}']
        for attr in ('error_msg', 'message'):
            detail = getattr(result_payload, attr, '')
            if detail:
                detail_parts.append(str(detail))
        for attr in ('error_code',):
            code = getattr(result_payload, attr, None)
            if code is not None:
                detail_parts.append(f'{attr}={code}')
        message = '; '.join(detail_parts)
        lowered = message.lower()
        if result.status == GoalStatus.STATUS_CANCELED:
            failure_type = 'CANCELED'
        elif 'progress' in lowered or 'failed to make progress' in lowered:
            failure_type = 'NAV2_FAILED_TO_MAKE_PROGRESS'
        else:
            failure_type = 'NAV2_FAILED'
        return message, failure_type

    def publish_result(
        self,
        source: str,
        success: bool,
        message: str,
        failure_type: Optional[str] = None,
    ):
        failed_x = None
        failed_y = None
        elapsed = None
        if self.active_goal is not None:
            failed_x = float(self.active_goal.pose.position.x)
            failed_y = float(self.active_goal.pose.position.y)
        if self.goal_start_time is not None:
            elapsed = time.monotonic() - self.goal_start_time
        msg = String()
        payload = {
            'source': source,
            'success': success,
            'message': message,
            'failed_goal_x': failed_x,
            'failed_goal_y': failed_y,
            'execution_time_sec': elapsed,
        }
        if not success:
            payload['failure_type'] = failure_type or 'NAV2_FAILED'
        msg.data = json.dumps(payload)
        self.result_pub.publish(msg)
        self.get_logger().info(f'[RALC] execution_result: {msg.data}')
        terminal_status = 'SUCCESS' if success else payload.get('failure_type', 'FAILED')
        if elapsed is not None:
            self.get_logger().info(
                f'[RALC_TIMING] nav2_execution: source={source}, '
                f'status={terminal_status}, seconds={elapsed:.2f}'
            )

    def publish_failed_goal_marker(self):
        if self.active_goal is None:
            return
        marker = Marker()
        marker.header = self.active_goal.header
        marker.ns = 'ralc_failed_goals'
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = self.active_goal.pose.position
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.scale.z = 0.35
        marker.color.r = 1.0
        marker.color.a = 0.9
        marker.lifetime.sec = 15
        self.failed_marker_pub.publish(marker)

    def reset(self):
        self.state = self.IDLE
        self.current_source = ''
        self.active_goal = None
        self.goal_start_time = None
        self.waypoints = []
        self.waypoint_index = 0


def main(args=None):
    rclpy.init(args=args)
    node = RalcExecutive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
