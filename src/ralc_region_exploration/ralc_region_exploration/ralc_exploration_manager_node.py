import json
import os
import subprocess
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String


class RalcExplorationManager(Node):
    """Owns the paper-faithful R-ALC state machine."""

    INIT = 'INIT'
    REGION_DISCOVERY = 'REGION_DISCOVERY'
    EXECUTING_ALC = 'EXECUTING_ALC'
    EXECUTING_FRONTIER = 'EXECUTING_FRONTIER'
    REGION_REFINEMENT = 'REGION_REFINEMENT'
    CHECKPOINT_AND_MARGINALIZATION = 'CHECKPOINT_AND_MARGINALIZATION'
    CREATE_NEXT_REGION_SEED = 'CREATE_NEXT_REGION_SEED'
    TRANSITION_TO_NEXT_REGION = 'TRANSITION_TO_NEXT_REGION'
    CREATE_NEXT_REGION_AT_ROBOT = 'CREATE_NEXT_REGION_AT_ROBOT'
    GLOBAL_PGS = 'GLOBAL_PGS'
    FINISHED = 'FINISHED'

    def __init__(self):
        super().__init__('ralc_exploration_manager')

        self.declare_parameter('checkpoint_root', 'maps/ralc_checkpoints')
        self.declare_parameter('enable_map_saving', True)
        self.declare_parameter(
            'require_region_at_max_before_no_actionable_completion',
            True,
        )
        self.declare_parameter('allow_pgs_unavailable_completion', True)
        self.declare_parameter('region_unknown_completion_threshold', 0.03)
        self.declare_parameter(
            'region_reachable_unknown_completion_threshold',
            0.03,
        )
        self.checkpoint_root = self.get_parameter('checkpoint_root').value
        self.enable_map_saving = bool(self.get_parameter('enable_map_saving').value)
        self.require_region_at_max_before_no_actionable_completion = bool(
            self.get_parameter(
                'require_region_at_max_before_no_actionable_completion'
            ).value
        )
        self.allow_pgs_unavailable_completion = bool(
            self.get_parameter('allow_pgs_unavailable_completion').value
        )
        self.region_unknown_completion_threshold = float(
            self.get_parameter('region_unknown_completion_threshold').value
        )
        self.region_reachable_unknown_completion_threshold = float(
            self.get_parameter(
                'region_reachable_unknown_completion_threshold'
            ).value
        )

        self.state = self.INIT
        self.current_region = None
        self.alc_goal: Optional[PoseStamped] = None
        self.alc_unavailable = True
        self.frontier_goal: Optional[PoseStamped] = None
        self.no_frontier_reason: Optional[str] = None
        self.no_frontier_report = None
        self.no_frontier_cells = None
        self.frontier_planner_status = None
        self.latest_region_stats = {}
        self.frontier_failure_count = 0
        self.frontier_report_seq = 0
        self.no_frontier_clusters = None
        self.last_nav_goal_success = False
        self.last_frontier_goal_success_report_seq = -1
        self.pgs_waypoints: Optional[PoseArray] = None
        self.pgs_status = None
        self.refinement_unavailable_accepted = False
        self.next_region_seed: Optional[PoseStamped] = None
        self.all_regions_explored = False
        self.waiting_for_execution = False
        self.checkpoint_done_for_region = None

        self.create_subscription(String, '/ralc/current_region', self.region_callback, 10)
        self.create_subscription(PoseStamped, '/ralc/alc_goal', self.alc_goal_callback, 10)
        self.create_subscription(Bool, '/ralc/alc_unavailable', self.alc_unavailable_callback, 10)
        self.create_subscription(PoseStamped, '/ralc/frontier_goal', self.frontier_goal_callback, 10)
        self.create_subscription(String, '/ralc/no_frontier_in_region', self.no_frontier_callback, 10)
        self.create_subscription(String, '/ralc/frontier_planner_status', self.frontier_status_callback, 10)
        self.create_subscription(String, '/ralc/execution_result', self.execution_result_callback, 10)
        self.create_subscription(PoseArray, '/ralc/pgs_waypoints', self.pgs_waypoints_callback, 10)
        self.create_subscription(String, '/ralc/pgs_status', self.pgs_status_callback, 10)
        self.create_subscription(PoseStamped, '/ralc/next_region_seed', self.next_region_seed_callback, 10)
        self.create_subscription(Bool, '/ralc/all_regions_explored', self.all_regions_callback, 10)

        self.execute_pose_pub = self.create_publisher(
            PoseStamped, '/ralc/execute_pose_goal', 10
        )
        self.execute_waypoints_pub = self.create_publisher(
            PoseArray, '/ralc/execute_waypoint_sequence', 10
        )
        self.create_region_pub = self.create_publisher(
            Empty, '/ralc/create_next_region', 10
        )
        self.create_region_at_robot_pub = self.create_publisher(
            Empty, '/ralc/create_region_at_robot', 10
        )
        self.mark_refinement_pub = self.create_publisher(
            Empty, '/ralc/mark_region_refinement', 10
        )
        self.mark_completed_pub = self.create_publisher(
            Empty, '/ralc/mark_region_completed', 10
        )
        self.expand_current_region_pub = self.create_publisher(
            Empty, '/ralc/expand_current_region', 10
        )
        self.recompute_frontiers_pub = self.create_publisher(
            Empty, '/ralc/recompute_frontiers', 10
        )
        self.frontier_goal_failed_pub = self.create_publisher(
            String, '/ralc/frontier_goal_failed', 10
        )
        self.request_pgs_pub = self.create_publisher(String, '/ralc/request_pgs', 10)
        self.marginalize_pub = self.create_publisher(
            String, '/ralc/marginalize_region_request', 10
        )
        self.state_pub = self.create_publisher(String, '/ralc/state', 10)
        self.region_completion_debug_pub = self.create_publisher(
            String, '/ralc/region_completion_debug', 10
        )

        self.timer = self.create_timer(1.0, self.tick)
        self.get_logger().info('[RALC] Exploration manager started.')

    def region_callback(self, msg: String):
        try:
            region = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.current_region = region if region else None

    def alc_goal_callback(self, msg: PoseStamped):
        self.alc_goal = msg
        self.alc_unavailable = False

    def alc_unavailable_callback(self, msg: Bool):
        self.alc_unavailable = bool(msg.data)
        if self.alc_unavailable:
            self.alc_goal = None

    def frontier_goal_callback(self, msg: PoseStamped):
        self.frontier_goal = msg
        self.no_frontier_reason = None
        self.frontier_planner_status = None
        self.frontier_failure_count = 0

    def no_frontier_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.no_frontier_report = data
            self.no_frontier_reason = data.get('reason', msg.data)
            self.no_frontier_cells = int(data.get('frontier_cells_in_region', -1))
            self.no_frontier_clusters = int(data.get('clusters', -1))
            self.update_latest_region_stats(data)
        except json.JSONDecodeError:
            self.no_frontier_report = {'reason': msg.data}
            self.no_frontier_reason = msg.data
            self.no_frontier_cells = -1
            self.no_frontier_clusters = -1
        self.frontier_goal = None
        self.frontier_report_seq += 1

    def frontier_status_callback(self, msg: String):
        try:
            self.frontier_planner_status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.frontier_planner_status = {'reason': msg.data}
        self.update_latest_region_stats(self.frontier_planner_status)
        self.frontier_report_seq += 1
        reason = self.frontier_planner_status.get('reason', 'UNKNOWN')
        if reason == 'REGION_STATS':
            self.frontier_planner_status = None
            return
        if reason in ('FRONTIERS_UNREACHABLE', 'FRONTIERS_REJECTED_BY_SAFETY'):
            self.frontier_failure_count += 1
            self.frontier_goal = None
            self.no_frontier_reason = None
            self.get_logger().warn(
                '[RALC] Frontier planner recovery case, not region completion: '
                f'reason={reason}, repeated={self.frontier_failure_count}'
            )
        elif reason == 'NO_ACTIONABLE_FRONTIER_IN_REGION':
            self.frontier_goal = None
            self.no_frontier_reason = None
            unknown_ratio = float(
                self.frontier_planner_status.get(
                    'reachable_unknown_ratio_in_region',
                    0.0,
                ) or 0.0
            )
            if unknown_ratio > self.region_reachable_unknown_completion_threshold:
                self.get_logger().warn(
                    '[RALC] NO_ACTIONABLE_FRONTIER_IN_REGION is not completion: '
                    f'reachable_unknown_ratio_in_region={unknown_ratio:.4f}, '
                    f'threshold='
                    f'{self.region_reachable_unknown_completion_threshold:.4f}'
                )
            else:
                self.get_logger().warn(
                    '[RALC] Region discovery may be complete with residual '
                    'non-actionable frontier artifacts; manager will verify '
                    'completion guards before refinement.'
                )
        elif reason == 'PLANNER_CLASSIFICATION_ERROR':
            self.frontier_goal = None
            self.no_frontier_reason = None
            self.get_logger().error(
                '[RALC] Frontier planner classification error; staying in '
                'REGION_DISCOVERY and requesting recomputation.'
            )

    def execution_result_callback(self, msg: String):
        try:
            result = json.loads(msg.data)
        except json.JSONDecodeError:
            result = {'success': False, 'message': msg.data, 'source': 'unknown'}
        self.waiting_for_execution = False
        self.get_logger().info(f'[RALC] execution finished: {result}')
        if self.state == self.EXECUTING_FRONTIER and not result.get('success', False):
            self.last_nav_goal_success = False
            self.publish_frontier_goal_failed(result)
            self.state = self.REGION_DISCOVERY
            self.clear_goals()
        elif self.state == self.EXECUTING_FRONTIER and result.get('success', False):
            self.last_nav_goal_success = True
            self.last_frontier_goal_success_report_seq = self.frontier_report_seq
            self.state = self.REGION_DISCOVERY
            self.clear_goals()
            self.recompute_frontiers_pub.publish(Empty())
            self.get_logger().info(
                '[RALC] Frontier goal reached; recomputing frontiers in same region.'
            )
        elif self.state in (self.EXECUTING_ALC, self.EXECUTING_FRONTIER):
            self.last_nav_goal_success = bool(result.get('success', False))
            self.state = self.REGION_DISCOVERY
            self.clear_goals()
        elif self.state == self.REGION_REFINEMENT:
            if result.get('success'):
                self.state = self.CHECKPOINT_AND_MARGINALIZATION
            else:
                self.get_logger().warn(
                    '[RALC] Refinement execution failed; region will remain '
                    'REFINEMENT and will not be checkpointed/completed.'
                )
                self.state = self.REGION_REFINEMENT
        elif self.state == self.TRANSITION_TO_NEXT_REGION:
            if result.get('success'):
                self.state = self.CREATE_NEXT_REGION_AT_ROBOT
            else:
                self.get_logger().warn(
                    '[RALC] Failed transition to next region seed; requesting '
                    'another seed instead of creating a remote region.'
                )
                self.state = self.CREATE_NEXT_REGION_SEED

    def pgs_waypoints_callback(self, msg: PoseArray):
        self.pgs_waypoints = msg

    def pgs_status_callback(self, msg: String):
        try:
            self.pgs_status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.pgs_status = {'success': False, 'message': msg.data}

    def next_region_seed_callback(self, msg: PoseStamped):
        self.next_region_seed = msg

    def all_regions_callback(self, msg: Bool):
        self.all_regions_explored = bool(msg.data)

    def tick(self):
        self.publish_state()
        if self.waiting_for_execution or self.state == self.FINISHED:
            return

        if self.state == self.INIT:
            if self.current_region is None:
                self.create_region_pub.publish(Empty())
                return
            self.state = self.REGION_DISCOVERY
            return

        if self.state == self.REGION_DISCOVERY:
            if self.current_region is None:
                self.state = self.CREATE_NEXT_REGION_SEED
                return
            if self.alc_goal is not None and not self.alc_unavailable:
                self.execute_pose_pub.publish(self.alc_goal)
                self.waiting_for_execution = True
                self.state = self.EXECUTING_ALC
                return
            if self.frontier_goal is not None:
                self.execute_pose_pub.publish(self.frontier_goal)
                self.waiting_for_execution = True
                self.state = self.EXECUTING_FRONTIER
                return
            if (
                self.no_frontier_reason == 'NO_FRONTIER_IN_REGION' and
                self.no_frontier_cells == 0 and
                self.no_frontier_clusters == 0
            ):
                if not self.fresh_frontier_report_after_success():
                    self.get_logger().warn(
                        '[RALC] Refusing REGION_REFINEMENT transition: last event '
                        'was FRONTIER_GOAL_SUCCESS and no fresh frontier report '
                        'has arrived after that success.'
                    )
                    self.no_frontier_reason = None
                    self.no_frontier_cells = None
                    self.no_frontier_clusters = None
                    self.recompute_frontiers_pub.publish(Empty())
                    return
                if not self.region_can_complete_after_no_actionable():
                    self.request_current_region_expansion()
                    self.no_frontier_reason = None
                    self.no_frontier_cells = None
                    self.no_frontier_clusters = None
                    self.no_frontier_report = None
                    return
                if not self.region_unknown_allows_completion(self.no_frontier_report):
                    self.refuse_completion_due_to_unknown(
                        self.no_frontier_report,
                        trigger_source='no_frontier_in_region',
                    )
                    self.no_frontier_reason = None
                    self.no_frontier_cells = None
                    self.no_frontier_clusters = None
                    self.no_frontier_report = None
                    return
                self.log_refinement_transition(
                    trigger_source='no_frontier_in_region',
                    frontier_status_reason=self.no_frontier_reason,
                    frontier_cells_in_region=self.no_frontier_cells,
                    clusters=self.no_frontier_clusters,
                    actionable_clusters=0,
                    frontier_report=self.no_frontier_report,
                )
                self.mark_refinement_pub.publish(Empty())
                self.request_pgs('regional')
                self.state = self.REGION_REFINEMENT
                return
            if self.frontier_planner_status is not None:
                reason = self.frontier_planner_status.get('reason', 'UNKNOWN')
                if reason in (
                    'FRONTIERS_REJECTED_BY_SAFETY',
                    'FRONTIERS_UNREACHABLE',
                    'NO_ACTIONABLE_FRONTIER_IN_REGION',
                ):
                    if not self.fresh_frontier_report_after_success():
                        self.get_logger().warn(
                            '[RALC] Refusing REGION_REFINEMENT transition: last '
                            'event was FRONTIER_GOAL_SUCCESS and no fresh '
                            'frontier_planner_status has arrived after that success.'
                        )
                        self.frontier_planner_status = None
                        self.recompute_frontiers_pub.publish(Empty())
                        return
                    if not self.region_can_complete_after_no_actionable():
                        self.get_logger().warn(
                            '[RALC] No actionable frontier, but region can still '
                            'grow; keeping current region active.'
                        )
                        self.request_current_region_expansion()
                        self.frontier_planner_status = None
                        return
                    if not self.region_unknown_allows_completion(
                        self.frontier_planner_status
                    ):
                        self.refuse_completion_due_to_unknown(
                            self.frontier_planner_status,
                            trigger_source='frontier_planner_status',
                        )
                        self.frontier_planner_status = None
                        return

                    self.get_logger().warn(
                        '[RALC] No actionable frontier and region is at max '
                        'size; entering refinement.'
                    )
                    self.log_refinement_transition(
                        trigger_source='frontier_planner_status',
                        frontier_status_reason=reason,
                        frontier_cells_in_region=self.frontier_planner_status.get(
                            'frontier_cells_in_region'
                        ),
                        clusters=self.frontier_planner_status.get('clusters'),
                        actionable_clusters=self.actionable_clusters_from_status(
                            self.frontier_planner_status
                        ),
                        frontier_report=self.frontier_planner_status,
                    )
                    self.mark_refinement_pub.publish(Empty())
                    self.request_pgs('regional')
                    self.frontier_planner_status = None
                    self.frontier_failure_count = 0
                    self.state = self.REGION_REFINEMENT
                    return
                if reason == 'PLANNER_CLASSIFICATION_ERROR':
                    self.publish_region_completion_debug(
                        event='COMPLETION_REFUSED_PLANNER_CLASSIFICATION_ERROR',
                        details={
                            'allowed_to_refine': False,
                            'reason': 'PLANNER_CLASSIFICATION_ERROR',
                            'frontier_status': self.frontier_planner_status,
                        },
                    )
                    self.frontier_planner_status = None
                    self.recompute_frontiers_pub.publish(Empty())
                    return
                self.get_logger().warn(
                    '[RALC] Staying in REGION_DISCOVERY because frontiers still '
                    f'exist but planner reported {reason}. This is recovery, '
                    'not region completion.'
                )
                self.frontier_planner_status = None
                return
            return

        if self.state == self.REGION_REFINEMENT:
            if self.pgs_status is not None and not self.pgs_status.get('success', False):
                status = self.pgs_status.get('status', '')
                if status == 'UNAVAILABLE' and self.allow_pgs_unavailable_completion:
                    self.refinement_unavailable_accepted = True
                    self.publish_region_completion_debug(
                        event='PGS_UNAVAILABLE_ACCEPTED',
                        details=self.pgs_status,
                    )
                    self.get_logger().warn(
                        '[RALC] PGS unavailable; accepting refinement as '
                        'unavailable for prototype testing and proceeding to '
                        'checkpoint/completion.'
                    )
                    self.pgs_status = None
                    self.state = self.CHECKPOINT_AND_MARGINALIZATION
                    return
                self.get_logger().warn(
                    '[RALC] PGS unavailable; refusing checkpoint/completion: '
                    f'{self.pgs_status}'
                )
                self.pgs_status = None
                self.request_pgs('regional')
                return
            if self.pgs_waypoints is None:
                self.request_pgs('regional')
                return
            if len(self.pgs_waypoints.poses) == 0:
                if self.allow_pgs_unavailable_completion:
                    self.refinement_unavailable_accepted = True
                    self.publish_region_completion_debug(
                        event='PGS_ZERO_WAYPOINTS_ACCEPTED',
                        details={'waypoint_count': 0},
                    )
                    self.get_logger().warn(
                        '[RALC] PGS produced zero waypoints; accepting '
                        'refinement unavailable for prototype testing.'
                    )
                    self.pgs_waypoints = None
                    self.state = self.CHECKPOINT_AND_MARGINALIZATION
                    return
                self.get_logger().warn(
                    '[RALC] PGS produced zero waypoints; region remains in '
                    'REFINEMENT and will not be completed.'
                )
                self.pgs_waypoints = None
                return
            self.execute_waypoints_pub.publish(self.pgs_waypoints)
            self.waiting_for_execution = True
            self.pgs_waypoints = None
            return

        if self.state == self.CHECKPOINT_AND_MARGINALIZATION:
            region = self.current_region
            if not self.save_checkpoint_and_request_marginalization(region):
                self.get_logger().warn(
                    '[RALC] Checkpoint failed; refusing to mark region COMPLETED.'
                )
                self.state = self.REGION_REFINEMENT
                return
            self.publish_region_completion_debug(
                event='REGION_COMPLETED',
                details={
                    'region_id': region.get('region_id') if region else None,
                    'refinement_unavailable_accepted': self.refinement_unavailable_accepted,
                },
            )
            self.mark_completed_pub.publish(Empty())
            self.refinement_unavailable_accepted = False
            self.current_region = None
            if self.all_regions_explored:
                self.state = self.GLOBAL_PGS
            else:
                self.state = self.CREATE_NEXT_REGION_SEED
            return

        if self.state == self.CREATE_NEXT_REGION_SEED:
            self.clear_goals()
            self.next_region_seed = None
            self.create_region_pub.publish(Empty())
            if self.all_regions_explored:
                self.state = self.GLOBAL_PGS
            else:
                self.state = self.TRANSITION_TO_NEXT_REGION
            return

        if self.state == self.TRANSITION_TO_NEXT_REGION:
            if self.next_region_seed is None:
                self.create_region_pub.publish(Empty())
                return
            self.execute_pose_pub.publish(self.next_region_seed)
            self.waiting_for_execution = True
            return

        if self.state == self.CREATE_NEXT_REGION_AT_ROBOT:
            self.create_region_at_robot_pub.publish(Empty())
            if self.current_region is not None:
                self.state = self.REGION_DISCOVERY
            return

        if self.state == self.GLOBAL_PGS:
            if self.pgs_waypoints is None:
                self.request_pgs('global')
                return
            self.execute_waypoints_pub.publish(self.pgs_waypoints)
            self.waiting_for_execution = True
            self.pgs_waypoints = None
            self.state = self.FINISHED
            return

    def request_pgs(self, mode: str):
        msg = String()
        msg.data = mode
        self.request_pgs_pub.publish(msg)

    def publish_region_completion_debug(self, event, details=None):
        msg = String()
        payload = {
            'event': event,
            'state': self.state,
            'region_id': (
                self.current_region.get('region_id')
                if self.current_region else None
            ),
            'unknown_cells_in_region': self.latest_region_stats.get(
                'unknown_cells_in_region'
            ),
            'unknown_ratio_in_region': self.latest_region_stats.get(
                'unknown_ratio_in_region'
            ),
            'rectangle_total_cells': self.latest_region_stats.get(
                'rectangle_total_cells'
            ),
            'raw_unknown_ratio_in_region': self.latest_region_stats.get(
                'raw_unknown_ratio_in_region'
            ),
            'reachable_free_cells_in_region': self.latest_region_stats.get(
                'reachable_free_cells_in_region'
            ),
            'reachable_frontier_cells_in_region': self.latest_region_stats.get(
                'reachable_frontier_cells_in_region'
            ),
            'reachable_unknown_ratio_in_region': self.latest_region_stats.get(
                'reachable_unknown_ratio_in_region'
            ),
            'reachable_unknown_cells_in_region': self.latest_region_stats.get(
                'reachable_unknown_cells_in_region'
            ),
            'blocked_unknown_cells_in_region': self.latest_region_stats.get(
                'blocked_unknown_cells_in_region'
            ),
            'blocked_unknown_cells_in_rectangle': self.latest_region_stats.get(
                'blocked_unknown_cells_in_rectangle'
            ),
            'frontier_cells_in_region': self.latest_region_stats.get(
                'frontier_cells_in_region'
            ),
            'clusters': self.latest_region_stats.get('clusters'),
            'actionable_clusters': self.latest_region_stats.get(
                'actionable_clusters'
            ),
            'non_actionable_clusters': self.latest_region_stats.get(
                'non_actionable_clusters'
            ),
            'details': details or {},
        }
        msg.data = json.dumps(payload)
        self.region_completion_debug_pub.publish(msg)
        self.get_logger().warn(f'[RALC] region_completion_debug: {msg.data}')

    def fresh_frontier_report_after_success(self):
        if not self.last_nav_goal_success:
            return True
        return self.frontier_report_seq > self.last_frontier_goal_success_report_seq

    def actionable_clusters_from_status(self, status):
        if not status:
            return None
        if 'actionable_clusters' in status:
            return int(status.get('actionable_clusters', 0) or 0)
        clusters = int(status.get('clusters', 0) or 0)
        rejections = status.get('rejections', {}) or {}
        rejected = (
            int(rejections.get('unreachable', 0) or 0) +
            int(rejections.get('too_close', 0) or 0) +
            int(rejections.get('safety', 0) or 0) +
            int(rejections.get('too_small', 0) or 0) +
            int(rejections.get('non_actionable', 0) or 0) +
            int(rejections.get('blacklisted', 0) or 0) +
            int(rejections.get('costmap', 0) or 0) +
            int(rejections.get('unknown', 0) or 0)
        )
        return max(0, clusters - rejected)

    def update_latest_region_stats(self, report):
        if not report:
            return
        keys = (
            'region_id',
            'rectangle_total_cells',
            'unknown_cells_in_region',
            'free_cells_in_region',
            'occupied_cells_in_region',
            'total_cells_in_region',
            'unknown_ratio_in_region',
            'raw_unknown_cells_in_region',
            'raw_unknown_ratio_in_region',
            'reachable_free_cells_in_region',
            'reachable_frontier_cells_in_region',
            'reachable_unknown_cells_in_region',
            'reachable_unknown_ratio_in_region',
            'blocked_unknown_cells_in_rectangle',
            'blocked_or_unreachable_unknown_cells_in_region',
            'blocked_unknown_cells_in_region',
            'robot_coverage_ratio_in_region',
            'frontier_cells_in_region',
            'clusters',
            'actionable_clusters',
            'non_actionable_clusters',
            'completion_allowed',
        )
        for key in keys:
            if key in report:
                self.latest_region_stats[key] = report[key]

    def region_unknown_allows_completion(self, report):
        if not report:
            return True
        if report.get('unknown_classification') == 'ENCLOSED_BY_OCCUPIED_WALLS':
            return True
        unknown_ratio = float(report.get(
            'reachable_unknown_ratio_in_region',
            report.get('unknown_ratio_in_region', 0.0),
        ) or 0.0)
        if unknown_ratio > self.region_reachable_unknown_completion_threshold:
            return False
        return bool(report.get('completion_allowed', True))

    def refuse_completion_due_to_unknown(self, report, trigger_source):
        raw_unknown_ratio = float(
            report.get('raw_unknown_ratio_in_region', 0.0) or 0.0
        )
        unknown_ratio = float(report.get(
            'reachable_unknown_ratio_in_region',
            report.get('unknown_ratio_in_region', 0.0),
        ) or 0.0)
        unknown_cells = int(report.get(
            'reachable_unknown_cells_in_region',
            report.get('unknown_cells_in_region', 0),
        ) or 0)
        reason = 'UNKNOWN_REMAINS_IN_REGION'
        if int(report.get('actionable_clusters', 0) or 0) == 0:
            reason = 'REACHABLE_UNKNOWN_REMAINS_BUT_NO_ACTIONABLE_FRONTIER'
        self.publish_region_completion_debug(
            event='COMPLETION_REFUSED_UNKNOWN_REMAINS',
            details={
                'allowed_to_refine': False,
                'reason': reason,
                'trigger_source': trigger_source,
                'frontier_status_reason': report.get('reason'),
                'reachable_unknown_cells_in_region': unknown_cells,
                'reachable_unknown_ratio_in_region': unknown_ratio,
                'raw_unknown_ratio_in_region': raw_unknown_ratio,
                'rectangle_total_cells': report.get('rectangle_total_cells'),
                'reachable_free_cells_in_region': report.get(
                    'reachable_free_cells_in_region'
                ),
                'reachable_frontier_cells_in_region': report.get(
                    'reachable_frontier_cells_in_region'
                ),
                'blocked_unknown_cells_in_region': report.get(
                    'blocked_unknown_cells_in_region'
                ),
                'blocked_unknown_cells_in_rectangle': report.get(
                    'blocked_unknown_cells_in_rectangle'
                ),
                'region_reachable_unknown_completion_threshold': (
                    self.region_reachable_unknown_completion_threshold
                ),
                'frontier_cells_in_region': report.get('frontier_cells_in_region'),
                'clusters': report.get('clusters'),
                'actionable_clusters': report.get('actionable_clusters'),
                'non_actionable_clusters': report.get('non_actionable_clusters'),
                'completion_allowed': False,
            },
        )
        self.get_logger().warn(
            '[RALC] Refusing completion: reachable unknown cells remain inside '
            f'active region. reachable_unknown_cells={unknown_cells}, '
            f'reachable_unknown_ratio={unknown_ratio:.4f}, '
            f'threshold='
            f'{self.region_reachable_unknown_completion_threshold:.4f}'
        )
        self.get_logger().warn(
            '[RALC] Refusing region completion: '
            f'reachable_unknown_ratio_in_region={unknown_ratio:.4f}, '
            f'threshold='
            f'{self.region_reachable_unknown_completion_threshold:.4f}'
        )
        if reason == 'REACHABLE_UNKNOWN_REMAINS_BUT_NO_ACTIONABLE_FRONTIER':
            self.get_logger().warn(
                '[RALC] Reachable unknown remains but no actionable frontier '
                'exists; planner recovery required.'
            )
        if self.current_region is not None and not bool(
            self.current_region.get('is_at_max_size', False)
        ):
            self.request_current_region_expansion()
        else:
            self.recompute_frontiers_pub.publish(Empty())

    def log_refinement_transition(
        self,
        trigger_source,
        frontier_status_reason,
        frontier_cells_in_region,
        clusters,
        actionable_clusters,
        frontier_report=None,
    ):
        is_at_max = None
        if self.current_region is not None:
            is_at_max = self.current_region.get('is_at_max_size')
        report = frontier_report or {}
        unknown_ratio = report.get(
            'reachable_unknown_ratio_in_region',
            report.get('unknown_ratio_in_region'),
        )
        unknown_cells = report.get(
            'reachable_unknown_cells_in_region',
            report.get('unknown_cells_in_region'),
        )
        raw_unknown_ratio = report.get('raw_unknown_ratio_in_region')
        completion_allowed = report.get('completion_allowed')
        self.get_logger().warn(
            '[RALC] REGION_REFINEMENT transition check: '
            f'trigger_source={trigger_source}, '
            f'frontier_status_reason={frontier_status_reason}, '
            f'frontier_cells_in_region={frontier_cells_in_region}, '
            f'clusters={clusters}, actionable_clusters={actionable_clusters}, '
            f'current_region.is_at_max_size={is_at_max}, '
            f'reachable_unknown_cells_in_region={unknown_cells}, '
            f'reachable_unknown_ratio_in_region={unknown_ratio}, '
            f'raw_unknown_ratio_in_region={raw_unknown_ratio}, '
            f'completion_allowed={completion_allowed}, '
            f'last_nav_goal_success={self.last_nav_goal_success}'
        )
        self.publish_region_completion_debug(
            event='REGION_REFINEMENT_TRANSITION',
            details={
                'trigger_source': trigger_source,
                'frontier_status_reason': frontier_status_reason,
                'frontier_cells_in_region': frontier_cells_in_region,
                'clusters': clusters,
                'actionable_clusters': actionable_clusters,
                'current_region_is_at_max_size': is_at_max,
                'reachable_unknown_cells_in_region': unknown_cells,
                'reachable_unknown_ratio_in_region': unknown_ratio,
                'raw_unknown_ratio_in_region': raw_unknown_ratio,
                'completion_allowed': completion_allowed,
                'last_nav_goal_success': self.last_nav_goal_success,
            },
        )

    def region_can_complete_after_no_actionable(self):
        if not self.require_region_at_max_before_no_actionable_completion:
            return True
        if self.current_region is None:
            return False
        return bool(self.current_region.get('is_at_max_size', False))

    def request_current_region_expansion(self):
        self.expand_current_region_pub.publish(Empty())

    def publish_frontier_goal_failed(self, result):
        failed_x = result.get('failed_goal_x')
        failed_y = result.get('failed_goal_y')
        if failed_x is None or failed_y is None:
            if self.frontier_goal is None:
                self.get_logger().warn(
                    '[RALC] Frontier goal failed, but no failed coordinates '
                    'were available to blacklist.'
                )
                return
            failed_x = float(self.frontier_goal.pose.position.x)
            failed_y = float(self.frontier_goal.pose.position.y)

        region_id = None
        if self.current_region is not None:
            region_id = self.current_region.get('region_id')

        msg = String()
        msg.data = json.dumps({
            'region_id': region_id,
            'failed_goal_x': float(failed_x),
            'failed_goal_y': float(failed_y),
            'failure_type': result.get('failure_type', 'NAV2_FAILED'),
            'message': result.get('message', ''),
        })
        self.frontier_goal_failed_pub.publish(msg)
        self.get_logger().warn(
            '[RALC] Frontier goal execution failed; staying in '
            f'REGION_DISCOVERY and reporting failed goal: {msg.data}'
        )

    def save_checkpoint_and_request_marginalization(self, region) -> bool:
        if not region:
            return False
        region_id = int(region['region_id'])
        checkpoint_dir = os.path.abspath(os.path.join(
            self.checkpoint_root, f'region_{region_id}'
        ))
        os.makedirs(checkpoint_dir, exist_ok=True)
        map_prefix = os.path.join(checkpoint_dir, 'map')
        if self.enable_map_saving:
            try:
                subprocess.run(
                    ['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', map_prefix],
                    check=True,
                    timeout=10.0,
                )
                self.get_logger().info(
                    f'[RALC] Saved region checkpoint to {map_prefix}.'
                )
            except Exception as exc:
                self.get_logger().warn(
                    f'[RALC] Map checkpoint save failed for {map_prefix}: {exc}'
                )
                return False
        request = String()
        request.data = json.dumps({
            'region_id': region_id,
            'checkpoint_path': checkpoint_dir,
            'status': 'REQUESTED',
            'note': (
                'SLAM Toolbox pose graph marginalization is unavailable in the '
                'current backend; interface kept for future integration.'
            ),
        })
        self.marginalize_pub.publish(request)
        self.get_logger().warn(
            '[RALC] Marginalization requested, but current SLAM backend does '
            'not expose a region marginalization service.'
        )
        return True

    def clear_goals(self):
        self.alc_goal = None
        self.frontier_goal = None
        self.no_frontier_reason = None
        self.no_frontier_report = None
        self.no_frontier_cells = None
        self.no_frontier_clusters = None
        self.frontier_planner_status = None
        self.pgs_status = None

    def publish_state(self):
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RalcExplorationManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
