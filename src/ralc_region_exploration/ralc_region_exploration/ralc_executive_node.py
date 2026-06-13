import json
import math
import time
from typing import List, Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseArray, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
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
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('fallback_base_frame', 'base_link')
        self.declare_parameter('nav_goal_timeout_sec', 90.0)
        self.declare_parameter('nav_stuck_timeout_sec', 12.0)
        self.declare_parameter('nav_stuck_progress_distance', 0.08)
        self.declare_parameter('cancel_nav_goal_on_stuck', True)
        self.declare_parameter('publish_zero_cmd_on_abort', True)
        self.enable_motion = bool(self.get_parameter('enable_motion').value)
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.fallback_base_frame = self.get_parameter('fallback_base_frame').value
        self.nav_goal_timeout_sec = float(
            self.get_parameter('nav_goal_timeout_sec').value
        )
        self.nav_stuck_timeout_sec = float(
            self.get_parameter('nav_stuck_timeout_sec').value
        )
        self.nav_stuck_progress_distance = float(
            self.get_parameter('nav_stuck_progress_distance').value
        )
        self.cancel_nav_goal_on_stuck = bool(
            self.get_parameter('cancel_nav_goal_on_stuck').value
        )
        self.publish_zero_cmd_on_abort = bool(
            self.get_parameter('publish_zero_cmd_on_abort').value
        )

        self.state = self.IDLE
        self.current_source = ''
        self.active_goal: Optional[PoseStamped] = None
        self.active_goal_handle = None
        self.cancel_requested = False
        self.goal_start_time: Optional[float] = None
        self.goal_start_robot_pose = None
        self.last_progress_robot_pose = None
        self.last_progress_time: Optional[float] = None
        self.waypoints: List[PoseStamped] = []
        self.waypoint_index = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.failed_marker_pub = self.create_publisher(
            Marker, '/ralc/failed_goal_marker', 10
        )
        self.create_timer(1.0, self.navigation_watchdog_callback)

        self.get_logger().info(
            '[RALC] Executive ready: /ralc/execute_pose_goal, '
            '/ralc/execute_waypoint_sequence -> Nav2 NavigateToPose.'
        )

    def execute_pose_goal_callback(self, msg: PoseStamped):
        if not self.enable_motion:
            self.publish_result('pose', True, 'motion disabled')
            return
        if self.state != self.IDLE:
            self.get_logger().warn(
                '[RALC] Executive busy; ignoring pose goal without publishing '
                f'an execution_result. state={self.state}, '
                f'ignored_goal=({msg.pose.position.x:.2f},'
                f'{msg.pose.position.y:.2f})'
            )
            return
        self.current_source = 'pose'
        self.state = self.EXECUTING_POSE
        self.send_goal(msg)

    def execute_waypoint_sequence_callback(self, msg: PoseArray):
        if not self.enable_motion:
            self.publish_result('waypoints', True, 'motion disabled')
            return
        first_pose = msg.poses[0] if msg.poses else None
        if self.state != self.IDLE:
            if first_pose is None:
                ignored_text = 'empty sequence'
            else:
                ignored_text = (
                    f'first_goal=({first_pose.position.x:.2f},'
                    f'{first_pose.position.y:.2f})'
                )
            self.get_logger().warn(
                '[RALC] Executive busy; ignoring waypoint sequence without '
                f'publishing an execution_result. state={self.state}, '
                f'{ignored_text}'
            )
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
            try:
                server_ready = self.nav_to_pose_client.wait_for_server(timeout_sec=2.0)
            except Exception as exc:
                self.active_goal = goal_pose
                self.publish_result(
                    self.current_source,
                    False,
                    f'Nav2 unavailable: {exc}',
                    failure_type='NAV2_FAILED',
                )
                self.reset('failure')
                return
            if not server_ready:
                self.active_goal = goal_pose
                self.publish_result(
                    self.current_source,
                    False,
                    'Nav2 unavailable',
                    failure_type='NAV2_FAILED',
                )
                self.reset('failure')
                return
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        self.active_goal = goal_pose
        self.active_goal_handle = None
        self.cancel_requested = False
        self.goal_start_time = time.monotonic()
        self.goal_start_robot_pose = self.lookup_robot_pose()
        self.last_progress_robot_pose = self.goal_start_robot_pose
        self.last_progress_time = self.goal_start_time
        self.get_logger().info(
            f'[RALC] Sending Nav2 goal source={self.current_source}: '
            f'x={goal_pose.pose.position.x:.2f}, y={goal_pose.pose.position.y:.2f}, '
            f'start_robot={self.pose_text(self.goal_start_robot_pose)}'
        )
        try:
            future = self.nav_to_pose_client.send_goal_async(goal_msg)
            future.add_done_callback(self.goal_response_callback)
        except Exception as exc:
            self.publish_result(
                self.current_source,
                False,
                f'Nav2 send_goal exception: {exc}',
                failure_type='NAV2_FAILED',
            )
            self.publish_failed_goal_marker()
            self.reset('failure')

    def lookup_robot_pose(self):
        for source_frame in (self.base_frame, self.fallback_base_frame):
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.05),
                )
                t = transform.transform.translation
                return float(t.x), float(t.y)
            except TransformException:
                continue
        return None

    def pose_text(self, pose):
        if pose is None:
            return 'unknown'
        return f'({pose[0]:.2f},{pose[1]:.2f})'

    def goal_response_callback(self, future):
        if self.state == self.IDLE or self.active_goal is None:
            self.get_logger().debug(
                '[RALC] Ignoring stale Nav2 goal response after executive reset.'
            )
            return
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.publish_result(
                self.current_source,
                False,
                f'Nav2 goal response exception: {exc}',
                failure_type='NAV2_FAILED',
            )
            self.publish_failed_goal_marker()
            self.reset('failure')
            return
        if not goal_handle.accepted:
            self.publish_result(
                self.current_source,
                False,
                'Nav2 rejected goal',
                failure_type='NAV2_FAILED',
            )
            self.publish_failed_goal_marker()
            self.reset('abort')
            return
        self.active_goal_handle = goal_handle
        try:
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.goal_result_callback)
        except Exception as exc:
            self.publish_result(
                self.current_source,
                False,
                f'Nav2 result request exception: {exc}',
                failure_type='NAV2_FAILED',
            )
            self.publish_failed_goal_marker()
            self.reset('failure')

    def goal_result_callback(self, future):
        if self.state == self.IDLE or self.active_goal is None:
            self.get_logger().debug(
                '[RALC] Ignoring stale Nav2 result after executive reset.'
            )
            return
        try:
            result = future.result()
        except Exception as exc:
            self.publish_result(
                self.current_source,
                False,
                f'Nav2 result exception: {exc}',
                failure_type='NAV2_FAILED',
            )
            self.publish_failed_goal_marker()
            self.reset('failure')
            return
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            message, failure_type = self.describe_nav2_failure(result)
            self.publish_result(
                self.current_source,
                False,
                message,
                failure_type=failure_type,
            )
            self.publish_failed_goal_marker()
            self.reset(failure_type)
            return
        if self.state == self.EXECUTING_WAYPOINTS:
            self.waypoint_index += 1
            if self.waypoint_index < len(self.waypoints):
                self.send_goal(self.waypoints[self.waypoint_index])
                return
        self.publish_result(self.current_source, True, 'succeeded')
        self.reset('success')

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

    def navigation_watchdog_callback(self):
        if self.state == self.IDLE or self.active_goal is None:
            return
        if self.goal_start_time is None:
            return
        now = time.monotonic()
        if self.nav_goal_timeout_sec > 0.0:
            elapsed = now - self.goal_start_time
            if elapsed > self.nav_goal_timeout_sec:
                self.abort_active_nav_goal(
                    'NAV2_TIMEOUT',
                    (
                        f'Nav2 goal timed out after {elapsed:.1f}s '
                        f'(limit {self.nav_goal_timeout_sec:.1f}s)'
                    ),
                )
                return
        if self.nav_stuck_timeout_sec <= 0.0:
            return
        pose = self.lookup_robot_pose()
        if pose is None:
            return
        if self.last_progress_robot_pose is None:
            self.last_progress_robot_pose = pose
            self.last_progress_time = now
            return
        progress = self.distance_2d(pose, self.last_progress_robot_pose)
        if progress >= self.nav_stuck_progress_distance:
            self.last_progress_robot_pose = pose
            self.last_progress_time = now
            return
        if self.last_progress_time is None:
            self.last_progress_time = now
            return
        stuck_duration = now - self.last_progress_time
        if stuck_duration > self.nav_stuck_timeout_sec:
            self.abort_active_nav_goal(
                'NAV2_STUCK_NO_PROGRESS',
                (
                    'Nav2 goal canceled by R-ALC watchdog: no robot '
                    f'translation >= {self.nav_stuck_progress_distance:.2f}m '
                    f'for {stuck_duration:.1f}s'
                ),
            )

    def abort_active_nav_goal(self, failure_type: str, message: str):
        if self.state == self.IDLE or self.active_goal is None:
            return
        self.get_logger().warn(
            f'[RALC] {message}; source={self.current_source}, '
            f'goal=({self.active_goal.pose.position.x:.2f},'
            f'{self.active_goal.pose.position.y:.2f})'
        )
        if self.cancel_nav_goal_on_stuck and self.active_goal_handle is not None:
            self.cancel_requested = True
            try:
                cancel_future = self.active_goal_handle.cancel_goal_async()
                cancel_future.add_done_callback(self.cancel_done_callback)
            except Exception as exc:
                self.get_logger().warn(
                    f'[RALC] Failed to request Nav2 goal cancel: {exc}'
                )
        elif self.cancel_nav_goal_on_stuck:
            self.get_logger().warn(
                '[RALC] Watchdog fired before Nav2 returned a goal handle; '
                'publishing failure without a cancel request.'
            )
        self.publish_zero_cmd_vel()
        self.publish_result(
            self.current_source,
            False,
            message,
            failure_type=failure_type,
        )
        self.publish_failed_goal_marker()
        self.reset(failure_type)

    def cancel_done_callback(self, future):
        try:
            future.result()
        except Exception as exc:
            self.get_logger().warn(f'[RALC] Nav2 cancel response exception: {exc}')
            return
        self.get_logger().info('[RALC] Nav2 cancel request completed.')

    def distance_2d(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def publish_zero_cmd_vel(self):
        if not self.publish_zero_cmd_on_abort:
            return
        self.cmd_vel_pub.publish(Twist())

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
        end_robot_pose = self.lookup_robot_pose()
        start_robot_x = None
        start_robot_y = None
        end_robot_x = None
        end_robot_y = None
        robot_travel_distance = None
        goal_yaw = None
        if self.active_goal is not None:
            failed_x = float(self.active_goal.pose.position.x)
            failed_y = float(self.active_goal.pose.position.y)
            goal_yaw = self.yaw_from_pose(self.active_goal)
        if self.goal_start_time is not None:
            elapsed = time.monotonic() - self.goal_start_time
        if self.goal_start_robot_pose is not None:
            start_robot_x = self.goal_start_robot_pose[0]
            start_robot_y = self.goal_start_robot_pose[1]
        if end_robot_pose is not None:
            end_robot_x = end_robot_pose[0]
            end_robot_y = end_robot_pose[1]
        if self.goal_start_robot_pose is not None and end_robot_pose is not None:
            robot_travel_distance = (
                (end_robot_pose[0] - self.goal_start_robot_pose[0]) ** 2 +
                (end_robot_pose[1] - self.goal_start_robot_pose[1]) ** 2
            ) ** 0.5
        msg = String()
        payload = {
            'source': source,
            'success': success,
            'message': message,
            'failed_goal_x': failed_x,
            'failed_goal_y': failed_y,
            'goal_x': failed_x,
            'goal_y': failed_y,
            'goal_yaw': goal_yaw,
            'start_robot_x': start_robot_x,
            'start_robot_y': start_robot_y,
            'end_robot_x': end_robot_x,
            'end_robot_y': end_robot_y,
            'robot_travel_distance': robot_travel_distance,
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
                f'status={terminal_status}, seconds={elapsed:.2f}, '
                f'start_robot={self.pose_text(self.goal_start_robot_pose)}, '
                f'end_robot={self.pose_text(end_robot_pose)}, '
                f'travel={robot_travel_distance if robot_travel_distance is not None else -1.0:.2f}m'
            )

    def yaw_from_pose(self, pose_stamped: PoseStamped):
        q = pose_stamped.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

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

    def reset(self, terminal_status='reset'):
        self.get_logger().info(
            f'[RALC] Executive reset after result: {terminal_status}'
        )
        self.state = self.IDLE
        self.current_source = ''
        self.active_goal = None
        self.active_goal_handle = None
        self.cancel_requested = False
        self.goal_start_time = None
        self.goal_start_robot_pose = None
        self.last_progress_robot_pose = None
        self.last_progress_time = None
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
