import heapq
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class FrontierCluster:
    cluster_id: int
    cells: np.ndarray
    area_cells: int
    centroid_world: Tuple[float, float]
    goal_world: Tuple[float, float]


@dataclass
class FrontierScore:
    cluster: FrontierCluster
    total_cost: float
    path_cost: float
    angular_penalty: float
    switch_cost: float
    information_gain: float


class RalcFrontierPlanner(Node):
    """Detects and scores frontiers inside the active R-ALC region only."""

    def __init__(self):
        super().__init__('ralc_frontier_planner')

        self.declare_parameter('min_frontier_cluster_size', 3)
        self.declare_parameter('min_actionable_frontier_cluster_size_cells', 15)
        self.declare_parameter('max_non_actionable_rejections', 5)
        self.declare_parameter('failed_goal_blacklist_seconds', 30.0)
        self.declare_parameter('failed_goal_aliasing_distance', 0.5)
        self.declare_parameter('max_failed_goals_per_region', 5)
        self.declare_parameter('min_observation_travel_distance', 0.20)
        self.declare_parameter('min_frontier_reduction_ratio', 0.10)
        self.declare_parameter('ineffective_goal_blacklist_seconds', 15.0)
        self.declare_parameter('ineffective_goal_aliasing_distance', 0.5)
        self.declare_parameter('max_ineffective_observation_attempts_per_cluster', 2)
        self.declare_parameter('min_visible_unknown_gain', 5)
        self.declare_parameter('reject_observation_pose_no_visible_unknown', False)
        self.declare_parameter('deferred_frontier_recheck_distance', 1.0)
        self.declare_parameter('deferred_frontier_timeout_sec', 20.0)
        self.declare_parameter('max_deferred_frontiers_before_region_growth', 3)
        self.declare_parameter('beta_A', 0.5)
        self.declare_parameter('beta_S', 0.5)
        self.declare_parameter('beta_G', 0.02)
        self.declare_parameter('same_frontier_distance', 0.5)
        self.declare_parameter('frontier_goal_inward_shift', 0.15)
        self.declare_parameter('frontier_goal_candidate_shifts', [0.35, 0.55, 0.80, 1.10, 1.40])
        self.declare_parameter('max_frontier_observation_distance', 2.0)
        self.declare_parameter('min_goal_distance_from_robot', 0.35)
        self.declare_parameter('occupied_safety_margin', 0.10)
        self.declare_parameter('unknown_safety_margin', 0.10)
        self.declare_parameter('enable_recovery_frontier_without_astar', False)
        self.declare_parameter('region_unknown_completion_threshold', 0.03)
        self.declare_parameter('region_reachable_unknown_completion_threshold', 0.03)
        self.declare_parameter('robot_coverage_radius', 0.75)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('fallback_base_frame', 'base_link')

        self.min_frontier_cluster_size = int(
            self.get_parameter('min_frontier_cluster_size').value
        )
        self.min_actionable_frontier_cluster_size_cells = int(
            self.get_parameter(
                'min_actionable_frontier_cluster_size_cells'
            ).value
        )
        self.max_non_actionable_rejections = int(
            self.get_parameter('max_non_actionable_rejections').value
        )
        self.failed_goal_blacklist_seconds = float(
            self.get_parameter('failed_goal_blacklist_seconds').value
        )
        self.failed_goal_aliasing_distance = float(
            self.get_parameter('failed_goal_aliasing_distance').value
        )
        self.max_failed_goals_per_region = int(
            self.get_parameter('max_failed_goals_per_region').value
        )
        self.min_observation_travel_distance = float(
            self.get_parameter('min_observation_travel_distance').value
        )
        self.min_frontier_reduction_ratio = float(
            self.get_parameter('min_frontier_reduction_ratio').value
        )
        self.ineffective_goal_blacklist_seconds = float(
            self.get_parameter('ineffective_goal_blacklist_seconds').value
        )
        self.ineffective_goal_aliasing_distance = float(
            self.get_parameter('ineffective_goal_aliasing_distance').value
        )
        self.max_ineffective_observation_attempts_per_cluster = int(
            self.get_parameter(
                'max_ineffective_observation_attempts_per_cluster'
            ).value
        )
        self.min_visible_unknown_gain = int(
            self.get_parameter('min_visible_unknown_gain').value
        )
        self.reject_observation_pose_no_visible_unknown = bool(
            self.get_parameter('reject_observation_pose_no_visible_unknown').value
        )
        self.deferred_frontier_recheck_distance = float(
            self.get_parameter('deferred_frontier_recheck_distance').value
        )
        self.deferred_frontier_timeout_sec = float(
            self.get_parameter('deferred_frontier_timeout_sec').value
        )
        self.max_deferred_frontiers_before_region_growth = int(
            self.get_parameter('max_deferred_frontiers_before_region_growth').value
        )
        self.beta_a = float(self.get_parameter('beta_A').value)
        self.beta_s = float(self.get_parameter('beta_S').value)
        self.beta_g = float(self.get_parameter('beta_G').value)
        self.same_frontier_distance = float(
            self.get_parameter('same_frontier_distance').value
        )
        self.frontier_goal_inward_shift = float(
            self.get_parameter('frontier_goal_inward_shift').value
        )
        self.frontier_goal_candidate_shifts = [
            float(value)
            for value in self.get_parameter('frontier_goal_candidate_shifts').value
        ]
        self.max_frontier_observation_distance = float(
            self.get_parameter('max_frontier_observation_distance').value
        )
        self.min_goal_distance_from_robot = float(
            self.get_parameter('min_goal_distance_from_robot').value
        )
        self.occupied_safety_margin = float(
            self.get_parameter('occupied_safety_margin').value
        )
        self.unknown_safety_margin = float(
            self.get_parameter('unknown_safety_margin').value
        )
        self.enable_recovery_frontier_without_astar = bool(
            self.get_parameter('enable_recovery_frontier_without_astar').value
        )
        self.region_unknown_completion_threshold = float(
            self.get_parameter('region_unknown_completion_threshold').value
        )
        self.region_reachable_unknown_completion_threshold = float(
            self.get_parameter(
                'region_reachable_unknown_completion_threshold'
            ).value
        )
        self.robot_coverage_radius = float(
            self.get_parameter('robot_coverage_radius').value
        )
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.fallback_base_frame = self.get_parameter('fallback_base_frame').value

        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_global_costmap: Optional[OccupancyGrid] = None
        self.current_region = None
        self.previous_selected_frontier: Optional[Tuple[float, float]] = None
        self.map_update_count = 0
        self.last_planned_map_update_count = -1
        self.non_actionable_rejection_counts = {}
        self.non_actionable_cluster_ids = set()
        self.non_actionable_cluster_reasons = {}
        self.last_non_actionable_clusters: List[FrontierCluster] = []
        self.last_clusters: List[FrontierCluster] = []
        self.failed_goal_blacklist = {}
        self.ineffective_observation_attempts = {}
        self.pending_observation_goal = None
        self.pending_observation_execution = None
        self.last_observation_report = None
        self.rejected_goal_candidates = []
        self.valid_visible_goal_candidates = []
        self.deferred_frontier_clusters = {}
        self.last_deferred_clusters: List[FrontierCluster] = []
        self.last_deferred_recheck_robot_pose = None
        self.robot_trajectory_points: List[Tuple[float, float]] = []
        self.previous_cluster_snapshots = {}
        self.region_mask_cache = {}
        self.region_color_palette = [
            (0.20, 0.90, 1.00),  # cyan/light blue
            (1.00, 0.55, 0.10),  # orange
            (0.70, 0.35, 1.00),  # purple
            (1.00, 0.90, 0.15),  # yellow
            (1.00, 0.35, 0.75),  # pink
            (0.10, 0.75, 0.65),  # teal
            (0.45, 1.00, 0.35),  # light green
        ]

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.global_costmap_callback,
            10,
        )
        self.create_subscription(String, '/ralc/current_region', self.region_callback, 10)
        self.create_subscription(
            Empty,
            '/ralc/recompute_frontiers',
            self.recompute_frontiers_callback,
            10,
        )
        self.create_subscription(
            String,
            '/ralc/frontier_goal_failed',
            self.frontier_goal_failed_callback,
            10,
        )
        self.create_subscription(
            String,
            '/ralc/execution_result',
            self.execution_result_callback,
            10,
        )

        self.goal_pub = self.create_publisher(PoseStamped, '/ralc/frontier_goal', 10)
        self.no_frontier_pub = self.create_publisher(
            String, '/ralc/no_frontier_in_region', 10
        )
        self.status_pub = self.create_publisher(
            String, '/ralc/frontier_planner_status', 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, '/ralc/frontier_debug_markers', 10
        )
        self.non_actionable_marker_pub = self.create_publisher(
            MarkerArray, '/ralc/non_actionable_frontier_markers', 10
        )
        self.failed_goal_marker_pub = self.create_publisher(
            MarkerArray, '/ralc/failed_frontier_goal_markers', 10
        )
        self.region_mask_marker_pub = self.create_publisher(
            MarkerArray, '/ralc/region_mask_markers', 10
        )
        self.legacy_marker_pub = self.create_publisher(
            MarkerArray, '/ralc/frontier_markers', 10
        )

        self.timer = self.create_timer(1.0, self.plan_frontier_goal)
        self.get_logger().info(
            '[RALC] Frontier planner ready. It publishes /ralc/frontier_goal '
            'or /ralc/no_frontier_in_region and never changes regions.'
        )

    def map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg
        self.map_update_count += 1

    def global_costmap_callback(self, msg: OccupancyGrid):
        self.latest_global_costmap = msg

    def region_callback(self, msg: String):
        try:
            region = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        previous_region = self.current_region
        self.current_region = region if region else None
        self.update_cached_region_status(previous_region, self.current_region)
        if self.region_id(previous_region) != self.region_id(self.current_region):
            self.deferred_frontier_clusters.clear()
        self.last_planned_map_update_count = -1

    def region_id(self, region):
        if not region:
            return None
        return int(region.get('region_id', 0) or 0)

    def update_cached_region_status(self, previous_region, current_region):
        if previous_region is None:
            return
        previous_id = int(previous_region.get('region_id', 0) or 0)
        if previous_id <= 0 or previous_id not in self.region_mask_cache:
            return
        if current_region is None:
            self.region_mask_cache[previous_id]['status'] = 'COMPLETED'
            return
        current_id = int(current_region.get('region_id', 0) or 0)
        if current_id != previous_id:
            self.region_mask_cache[previous_id]['status'] = 'COMPLETED'
        else:
            status = current_region.get('status', 'ACTIVE')
            self.region_mask_cache[previous_id]['status'] = status

    def recompute_frontiers_callback(self, _msg: Empty):
        self.last_planned_map_update_count = -1
        self.get_logger().info(
            '[RALC] Received /ralc/recompute_frontiers; forcing frontier recompute.'
        )

    def frontier_goal_failed_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(
                f'[RALC] Ignoring malformed frontier_goal_failed: {msg.data}'
            )
            return
        failed_x = data.get('failed_goal_x')
        failed_y = data.get('failed_goal_y')
        if failed_x is None or failed_y is None:
            return
        region_id = data.get('region_id')
        if region_id is None and self.current_region is not None:
            region_id = self.current_region.get('region_id')
        if region_id is None:
            return

        failed_point = (float(failed_x), float(failed_y))
        nearest = self.nearest_cluster_to_point(
            self.last_clusters,
            failed_point,
            int(region_id),
        )
        if nearest is not None:
            cluster, distance = nearest
            cluster_key = self.cluster_identity_for_region(cluster, int(region_id))
            centroid = cluster.centroid_world
        else:
            distance = None
            cluster_key = self.point_identity_for_region(
                failed_point[0],
                failed_point[1],
                int(region_id),
            )
            centroid = failed_point

        failure_type = data.get('failure_type', 'NAV2_FAILED')
        blacklist_seconds = self.failed_goal_blacklist_seconds
        aliasing_distance = self.failed_goal_aliasing_distance
        if failure_type == 'INEFFECTIVE_OBSERVATION_GOAL':
            blacklist_seconds = self.ineffective_goal_blacklist_seconds
            aliasing_distance = self.ineffective_goal_aliasing_distance

        self.failed_goal_blacklist[cluster_key] = {
            'region_id': int(region_id),
            'centroid_x': float(centroid[0]),
            'centroid_y': float(centroid[1]),
            'failed_goal_x': failed_point[0],
            'failed_goal_y': failed_point[1],
            'failure_type': failure_type,
            'message': data.get('message', ''),
            'timestamp': self.now_sec(),
            'blacklist_seconds': blacklist_seconds,
            'aliasing_distance': aliasing_distance,
        }
        self.last_planned_map_update_count = -1
        distance_text = 'unknown' if distance is None else f'{distance:.2f}m'
        self.get_logger().warn(
            '[RALC] Blacklisted failed frontier goal for current region: '
            f'region={region_id}, failed_goal=({failed_point[0]:.2f},'
            f'{failed_point[1]:.2f}), nearest_cluster_distance={distance_text}, '
            f'failure={failure_type}, duration={blacklist_seconds:.1f}s'
        )

    def execution_result_callback(self, msg: String):
        if self.pending_observation_goal is None:
            return
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if data.get('source') != 'pose':
            return
        goal_x = data.get('goal_x', data.get('failed_goal_x'))
        goal_y = data.get('goal_y', data.get('failed_goal_y'))
        if goal_x is None or goal_y is None:
            return
        pending_goal = self.pending_observation_goal.get('goal_world')
        if pending_goal is None:
            return
        if math.hypot(float(goal_x) - pending_goal[0], float(goal_y) - pending_goal[1]) > 0.25:
            return
        if data.get('success', False):
            self.pending_observation_execution = data

    def lookup_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        for source_frame in (self.base_frame, self.fallback_base_frame):
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.05),
                )
                t = transform.transform.translation
                q = transform.transform.rotation
                yaw = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                )
                return float(t.x), float(t.y), yaw
            except TransformException:
                continue
        return None

    def plan_frontier_goal(self):
        if self.latest_map is None or self.current_region is None:
            return
        if self.current_region.get('status') != 'ACTIVE':
            return

        grid = self.latest_map
        robot_pose = self.lookup_robot_pose()
        if robot_pose is None:
            return
        if (
            self.map_update_count == self.last_planned_map_update_count and
            not self.should_recheck_deferred_frontiers(robot_pose)
        ):
            return
        if self.deferred_frontier_clusters:
            self.last_deferred_recheck_robot_pose = robot_pose[:2]

        cycle_start = time.monotonic()
        self.update_robot_coverage(robot_pose)
        free, unknown = self.occupancy_grid_to_masks(grid)
        global_raw_mask = self.detect_frontier_mask(free, unknown)
        active_region_mask = self.reachable_free_component(grid, free, robot_pose)
        raw_mask = self.detect_frontier_mask(active_region_mask, unknown)
        region_mask = raw_mask
        occupancy_stats = self.region_occupancy_stats(
            grid,
            active_region_mask,
            unknown,
            int(region_mask.sum()),
        )
        cluster_start = time.monotonic()
        clusters = self.connected_frontier_clusters(region_mask, grid, robot_pose)
        cluster_ms = (time.monotonic() - cluster_start) * 1000.0
        self.last_clusters = clusters
        self.prune_expired_failed_goal_blacklist()
        rejection_stats = {
            'unreachable': 0,
            'too_close': 0,
            'safety': 0,
            'too_small': 0,
            'non_actionable': 0,
            'blacklisted': 0,
            'ineffective_observation': 0,
            'deferred_visibility': 0,
            'observation_pose_no_visible_unknown': 0,
            'costmap': 0,
            'unknown': 0,
        }
        observation_report = self.evaluate_pending_observation(
            clusters,
            int(region_mask.sum()),
        )
        self.last_non_actionable_clusters = []
        self.last_deferred_clusters = []
        self.rejected_goal_candidates = []
        self.valid_visible_goal_candidates = []
        choose_start = time.monotonic()
        scores = self.score_frontiers(clusters, robot_pose, grid, rejection_stats)
        best = min(scores, key=lambda score: score.total_cost, default=None)
        recovery_score = None
        if best is None and clusters and self.enable_recovery_frontier_without_astar:
            recovery_score = self.recovery_frontier_score(
                clusters,
                robot_pose,
                grid,
                rejection_stats,
            )
            if recovery_score is not None:
                best = recovery_score
        choose_ms = (time.monotonic() - choose_start) * 1000.0

        self.publish_frontier_markers(
            grid,
            global_raw_mask,
            region_mask,
            clusters,
            best,
            active_region_mask,
            best is recovery_score,
        )
        self.update_region_mask_cache(grid, active_region_mask)
        self.publish_region_mask_markers(grid)
        self.publish_non_actionable_markers(grid, self.last_non_actionable_clusters)
        self.publish_failed_goal_markers(grid)
        self.log_debug(grid, raw_mask, region_mask, clusters, scores, robot_pose)
        self.log_unknown_ratio_debug(occupancy_stats)
        self.log_disappeared_frontier_clusters(grid, raw_mask, region_mask, clusters)
        self.get_logger().info(
            '[RALC_TIMING] frontier_planner: '
            f'clusters_ms={cluster_ms:.1f}, choose_goal_ms={choose_ms:.1f}, '
            f'total_ms={(time.monotonic() - cycle_start) * 1000.0:.1f}'
        )

        if best is None:
            reason = self.frontier_status_reason(
                raw_mask,
                region_mask,
                clusters,
                rejection_stats,
            )
            actionable_clusters = 0
            non_actionable_clusters = self.non_actionable_cluster_count(
                clusters,
                rejection_stats,
            )
            if reason == 'NO_FRONTIER_IN_REGION':
                self.publish_no_frontier(
                    reason,
                    len(clusters),
                    int(region_mask.sum()),
                    occupancy_stats,
                    actionable_clusters,
                    non_actionable_clusters,
                    observation_report=observation_report,
                )
            else:
                self.publish_frontier_status(
                    reason,
                    len(clusters),
                    int(region_mask.sum()),
                    rejection_stats,
                    occupancy_stats,
                    actionable_clusters,
                    non_actionable_clusters,
                    observation_report=observation_report,
                )
            self.last_planned_map_update_count = self.map_update_count
            return

        self.publish_frontier_status(
            'REGION_STATS',
            len(clusters),
            int(region_mask.sum()),
            rejection_stats,
            occupancy_stats,
            len(scores) + (1 if recovery_score is not None else 0),
            self.non_actionable_cluster_count(clusters, rejection_stats),
            selected_score=best,
            observation_report=observation_report,
        )
        goal = self.make_goal_msg(
            grid,
            best.cluster.goal_world,
            getattr(
                best.cluster,
                'observation_target_world',
                best.cluster.centroid_world,
            ),
        )
        self.goal_pub.publish(goal)
        self.previous_selected_frontier = best.cluster.goal_world
        self.pending_observation_goal = {
            'region_id': int(self.current_region.get('region_id')),
            'selected_cluster_id': int(best.cluster.cluster_id),
            'selected_goal_x': float(goal.pose.position.x),
            'selected_goal_y': float(goal.pose.position.y),
            'selected_centroid_x': float(best.cluster.centroid_world[0]),
            'selected_centroid_y': float(best.cluster.centroid_world[1]),
            'selected_observation_target_x': float(getattr(
                best.cluster,
                'observation_target_world',
                best.cluster.centroid_world,
            )[0]),
            'selected_observation_target_y': float(getattr(
                best.cluster,
                'observation_target_world',
                best.cluster.centroid_world,
            )[1]),
            'goal_world': (float(goal.pose.position.x), float(goal.pose.position.y)),
            'centroid_world': best.cluster.centroid_world,
            'previous_frontier_cells_in_region': int(region_mask.sum()),
            'previous_clusters': len(clusters),
            'map_update_count_before': self.map_update_count,
            'timestamp': self.now_sec(),
        }
        self.pending_observation_execution = None
        mode = 'RECOVERY_FRONTIER_GOAL' if best is recovery_score else 'FRONTIER_GOAL'
        self.get_logger().info(
            f'[RALC] {mode} selected: '
            f'cluster={best.cluster.cluster_id}, '
            f'centroid=({best.cluster.centroid_world[0]:.2f},'
            f'{best.cluster.centroid_world[1]:.2f}), '
            f'goal=({goal.pose.position.x:.2f},{goal.pose.position.y:.2f}), '
            f'cost={best.total_cost:.3f}, path={best.path_cost:.2f}, '
            f'angle={best.angular_penalty:.2f}, switch={best.switch_cost:.2f}, '
            f'gain={best.information_gain:.0f}'
        )
        self.last_planned_map_update_count = self.map_update_count

    def occupancy_grid_to_masks(self, grid):
        image = np.array(grid.data, dtype=np.int16).reshape(
            (grid.info.height, grid.info.width)
        )
        return image == 0, image == -1

    def detect_frontier_mask(self, free, unknown):
        unknown_neighbor = np.zeros_like(unknown, dtype=bool)
        unknown_neighbor[1:, :] |= unknown[:-1, :]
        unknown_neighbor[:-1, :] |= unknown[1:, :]
        unknown_neighbor[:, 1:] |= unknown[:, :-1]
        unknown_neighbor[:, :-1] |= unknown[:, 1:]
        return free & unknown_neighbor

    def filter_mask_to_region(self, mask, grid):
        height, width = mask.shape
        xs = np.arange(width)
        ys = np.arange(height)
        origin = grid.info.origin.position
        resolution = grid.info.resolution
        world_x = origin.x + (xs.astype(np.float64) + 0.5) * resolution
        world_y = origin.y + (ys.astype(np.float64) + 0.5) * resolution
        x_inside = (
            (world_x >= float(self.current_region['xmin'])) &
            (world_x <= float(self.current_region['xmax']))
        )
        y_inside = (
            (world_y >= float(self.current_region['ymin'])) &
            (world_y <= float(self.current_region['ymax']))
        )
        return mask & (y_inside[:, None] & x_inside[None, :])

    def region_cell_bounds(self, grid):
        """Return clamped map-cell bounds for the current active rectangle."""
        if self.current_region is None:
            return None
        origin = grid.info.origin.position
        resolution = grid.info.resolution
        xmin_cell = int(math.floor(
            (float(self.current_region['xmin']) - origin.x) / resolution
        ))
        xmax_cell = int(math.ceil(
            (float(self.current_region['xmax']) - origin.x) / resolution
        ))
        ymin_cell = int(math.floor(
            (float(self.current_region['ymin']) - origin.y) / resolution
        ))
        ymax_cell = int(math.ceil(
            (float(self.current_region['ymax']) - origin.y) / resolution
        ))
        xmin_cell = max(0, min(grid.info.width, xmin_cell))
        xmax_cell = max(0, min(grid.info.width, xmax_cell))
        ymin_cell = max(0, min(grid.info.height, ymin_cell))
        ymax_cell = max(0, min(grid.info.height, ymax_cell))
        if xmin_cell >= xmax_cell or ymin_cell >= ymax_cell:
            return None
        return xmin_cell, xmax_cell, ymin_cell, ymax_cell

    def reachable_free_component(self, grid, free, robot_pose):
        """Known-free cells reachable from the robot without crossing walls."""
        reachable = np.zeros_like(free, dtype=bool)
        bounds = self.region_cell_bounds(grid)
        if bounds is None:
            return reachable
        start = self.world_to_cell(grid, robot_pose[0], robot_pose[1])
        if start is None:
            return reachable
        xmin, xmax, ymin, ymax = bounds
        if not (xmin <= start[0] < xmax and ymin <= start[1] < ymax):
            return reachable
        image = np.array(grid.data, dtype=np.int16).reshape(
            (grid.info.height, grid.info.width)
        )
        start = self.nearest_free_cell_in_region(image, start, bounds, 12)
        if start is None:
            return reachable

        queue = deque([start])
        reachable[start[1], start[0]] = True
        while queue:
            x, y = queue.popleft()
            for nx, ny in (
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
            ):
                if nx < xmin or nx >= xmax or ny < ymin or ny >= ymax:
                    continue
                if reachable[ny, nx] or not free[ny, nx]:
                    continue
                reachable[ny, nx] = True
                queue.append((nx, ny))
        return reachable

    def nearest_free_cell_in_region(self, image, cell, bounds, radius_cells):
        xmin, xmax, ymin, ymax = bounds
        if xmin <= cell[0] < xmax and ymin <= cell[1] < ymax:
            if image[cell[1], cell[0]] == 0:
                return cell
        best = None
        best_dist = None
        for radius in range(1, radius_cells + 1):
            for y in range(
                max(ymin, cell[1] - radius),
                min(ymax, cell[1] + radius + 1),
            ):
                for x in range(
                    max(xmin, cell[0] - radius),
                    min(xmax, cell[0] + radius + 1),
                ):
                    if image[y, x] != 0:
                        continue
                    dist = math.hypot(x - cell[0], y - cell[1])
                    if best_dist is None or dist < best_dist:
                        best = (x, y)
                        best_dist = dist
            if best is not None:
                return best
        return None

    def unknown_adjacent_to_reachable_free(self, reachable_free, unknown):
        adjacent = np.zeros_like(unknown, dtype=bool)
        adjacent[1:, :] |= reachable_free[:-1, :]
        adjacent[:-1, :] |= reachable_free[1:, :]
        adjacent[:, 1:] |= reachable_free[:, :-1]
        adjacent[:, :-1] |= reachable_free[:, 1:]
        return unknown & adjacent

    def region_occupancy_stats(
        self,
        grid,
        reachable_free,
        unknown,
        reachable_frontier_cells,
    ):
        """Compute map-discovery statistics inside the active region.

        Raw unknown is useful for RViz interpretation, but completion uses only
        unknown cells adjacent to robot-reachable known free space.
        """
        bounds = self.region_cell_bounds(grid)
        if bounds is None:
            return {
                'rectangle_total_cells': 0,
                'raw_unknown_cells_in_region': 0,
                'raw_unknown_ratio_in_region': 0.0,
                'reachable_free_cells_in_region': 0,
                'reachable_frontier_cells_in_region': 0,
                'reachable_unknown_cells_in_region': 0,
                'reachable_unknown_ratio_in_region': 0.0,
                'blocked_unknown_cells_in_rectangle': 0,
                'blocked_or_unreachable_unknown_cells_in_region': 0,
                'blocked_unknown_cells_in_region': 0,
                'unknown_cells_in_region': 0,
                'free_cells_in_region': 0,
                'occupied_cells_in_region': 0,
                'total_cells_in_region': 0,
                'unknown_ratio_in_region': 0.0,
                'robot_coverage_ratio_in_region': 0.0,
            }

        xmin, xmax, ymin, ymax = bounds
        image = np.array(grid.data, dtype=np.int16).reshape(
            (grid.info.height, grid.info.width)
        )
        region = image[ymin:ymax, xmin:xmax]
        total = int(region.size)
        raw_unknown_cells = int(np.count_nonzero(region == -1))
        free_cells = int(np.count_nonzero(region == 0))
        occupied_cells = int(np.count_nonzero(region > 50))
        raw_unknown_ratio = float(raw_unknown_cells / total) if total else 0.0
        reachable_region = reachable_free[ymin:ymax, xmin:xmax]
        reachable_free_cells = int(np.count_nonzero(reachable_region))
        reachable_unknown_mask = self.unknown_adjacent_to_reachable_free(
            reachable_free,
            unknown,
        )
        reachable_unknown_cells = int(np.count_nonzero(
            reachable_unknown_mask[ymin:ymax, xmin:xmax]
        ))
        reachable_unknown_ratio = (
            float(reachable_unknown_cells / total) if total else 0.0
        )
        blocked_unknown_cells = max(
            0,
            raw_unknown_cells - reachable_unknown_cells,
        )
        return {
            'rectangle_total_cells': total,
            'raw_unknown_cells_in_region': raw_unknown_cells,
            'raw_unknown_ratio_in_region': raw_unknown_ratio,
            'reachable_free_cells_in_region': reachable_free_cells,
            'reachable_frontier_cells_in_region': int(reachable_frontier_cells),
            'reachable_unknown_cells_in_region': reachable_unknown_cells,
            'reachable_unknown_ratio_in_region': reachable_unknown_ratio,
            'blocked_unknown_cells_in_rectangle': blocked_unknown_cells,
            'blocked_or_unreachable_unknown_cells_in_region': blocked_unknown_cells,
            'blocked_unknown_cells_in_region': blocked_unknown_cells,
            # Backward-compatible aliases now refer to reachable unknown.
            'unknown_cells_in_region': reachable_unknown_cells,
            'free_cells_in_region': free_cells,
            'occupied_cells_in_region': occupied_cells,
            'total_cells_in_region': total,
            'unknown_ratio_in_region': reachable_unknown_ratio,
            'robot_coverage_ratio_in_region': self.robot_coverage_ratio(
                grid,
                bounds,
                free_cells,
            ),
        }

    def update_robot_coverage(self, robot_pose):
        point = (float(robot_pose[0]), float(robot_pose[1]))
        if not self.robot_trajectory_points:
            self.robot_trajectory_points.append(point)
            return
        last = self.robot_trajectory_points[-1]
        if math.hypot(point[0] - last[0], point[1] - last[1]) >= 0.25:
            self.robot_trajectory_points.append(point)

    def robot_coverage_ratio(self, grid, bounds, free_cells):
        if free_cells <= 0 or not self.robot_trajectory_points:
            return 0.0
        xmin, xmax, ymin, ymax = bounds
        image = np.array(grid.data, dtype=np.int16).reshape(
            (grid.info.height, grid.info.width)
        )
        free_region = image[ymin:ymax, xmin:xmax] == 0
        covered = np.zeros_like(free_region, dtype=bool)
        radius_cells = max(1, int(math.ceil(
            self.robot_coverage_radius / grid.info.resolution
        )))
        for world_x, world_y in self.robot_trajectory_points:
            cell = self.world_to_cell(grid, world_x, world_y)
            if cell is None:
                continue
            cx, cy = cell
            if cx < xmin or cx >= xmax or cy < ymin or cy >= ymax:
                continue
            local_x = cx - xmin
            local_y = cy - ymin
            y0 = max(0, local_y - radius_cells)
            y1 = min(covered.shape[0], local_y + radius_cells + 1)
            x0 = max(0, local_x - radius_cells)
            x1 = min(covered.shape[1], local_x + radius_cells + 1)
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    if math.hypot(xx - local_x, yy - local_y) <= radius_cells:
                        covered[yy, xx] = True
        covered_free = int(np.count_nonzero(covered & free_region))
        return float(covered_free / free_cells)

    def connected_frontier_clusters(self, mask, grid, robot_pose):
        visited = np.zeros_like(mask, dtype=bool)
        clusters = []
        cluster_id = 1
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if visited[y, x] or not mask[y, x]:
                    continue
                cells = self.collect_cluster(x, y, mask, visited)
                if len(cells) < self.min_frontier_cluster_size:
                    continue
                arr = np.array(cells, dtype=np.float64)
                centroid_cell = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
                centroid_world = self.map_cell_to_world(
                    grid, centroid_cell[0], centroid_cell[1]
                )
                goal_world = self.shift_goal_toward_robot(
                    centroid_world, robot_pose[0], robot_pose[1]
                )
                clusters.append(FrontierCluster(
                    cluster_id=cluster_id,
                    cells=np.array(cells, dtype=np.int32),
                    area_cells=len(cells),
                    centroid_world=centroid_world,
                    goal_world=goal_world,
                ))
                cluster_id += 1
        return clusters

    def collect_cluster(self, start_x, start_y, mask, visited):
        queue = deque([(start_x, start_y)])
        visited[start_y, start_x] = True
        cells = []
        height, width = mask.shape
        while queue:
            x, y = queue.popleft()
            cells.append((x, y))
            for ny in range(y - 1, y + 2):
                for nx in range(x - 1, x + 2):
                    if nx == x and ny == y:
                        continue
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        return cells

    def score_frontiers(self, clusters, robot_pose, grid, rejection_stats):
        scores = []
        for cluster in clusters:
            if self.cluster_is_non_actionable(cluster):
                rejection_reason = self.non_actionable_cluster_reasons.get(
                    self.cluster_identity(cluster),
                    'non_actionable',
                )
                if rejection_reason in rejection_stats:
                    rejection_stats[rejection_reason] += 1
                else:
                    rejection_stats['non_actionable'] += 1
                self.remember_non_actionable_marker(cluster)
                continue
            if cluster.area_cells < self.min_actionable_frontier_cluster_size_cells:
                rejection_stats['too_small'] += 1
                self.record_cluster_rejection(cluster, 'too_small')
                continue
            best_candidate = self.best_goal_candidate_for_cluster(
                cluster,
                robot_pose,
                grid,
                rejection_stats,
                self.frontier_goal_candidate_shifts,
            )
            if best_candidate is None:
                continue
            if best_candidate.get('deferred'):
                rejection_stats['deferred_visibility'] += 1
                self.remember_deferred_cluster(cluster)
                continue
            self.deferred_frontier_clusters.pop(self.cluster_identity(cluster), None)
            goal_world = best_candidate['goal']
            path_cost = best_candidate['path_cost']
            cluster.goal_world = goal_world
            cluster.observation_target_world = best_candidate.get(
                'observation_target',
                cluster.centroid_world,
            )
            if path_cost is None:
                rejection_stats['unreachable'] += 1
                self.record_cluster_rejection(cluster, 'unreachable')
                continue
            distance_to_goal = math.hypot(
                goal_world[0] - robot_pose[0],
                goal_world[1] - robot_pose[1],
            )
            if distance_to_goal < self.min_goal_distance_from_robot:
                rejection_stats['too_close'] += 1
                self.record_cluster_rejection(cluster, 'too_close')
                continue
            if not self.goal_is_safe(grid, goal_world):
                rejection_stats['safety'] += 1
                self.record_cluster_rejection(cluster, 'safety')
                continue
            target_heading = math.atan2(
                goal_world[1] - robot_pose[1],
                goal_world[0] - robot_pose[0],
            )
            angular_penalty = abs(self.normalize_angle(target_heading - robot_pose[2]))
            switch_cost = self.switch_cost(cluster.goal_world)
            information_gain = float(cluster.area_cells)
            total = (
                path_cost +
                0.20 * best_candidate['frontier_distance'] -
                0.40 * best_candidate['clearance'] +
                self.beta_a * angular_penalty +
                self.beta_s * switch_cost -
                self.beta_g * information_gain -
                0.01 * float(best_candidate.get('visible_unknown_gain', 0))
            )
            scores.append(FrontierScore(
                cluster=cluster,
                total_cost=total,
                path_cost=path_cost,
                angular_penalty=angular_penalty,
                switch_cost=switch_cost,
                information_gain=information_gain,
            ))
        return scores

    def recovery_frontier_score(self, clusters, robot_pose, grid, rejection_stats):
        """Pick a secondary frontier candidate only when A* still proves reachability.

        The recovery pass may use broader candidate shifts, but it must not bypass
        path validation. Unreachable goals are left for region growth/transition.
        """
        candidates = []
        for cluster in clusters:
            if (
                self.cluster_is_non_actionable(cluster) or
                cluster.area_cells < self.min_actionable_frontier_cluster_size_cells
            ):
                continue
            best_candidate = self.best_goal_candidate_for_cluster(
                cluster,
                robot_pose,
                grid,
                rejection_stats,
                (0.80, 1.10, 1.40, 1.80),
                require_astar=True,
            )
            if best_candidate is None:
                rejection_stats['unreachable'] += 1
                self.record_cluster_rejection(cluster, 'unreachable')
                continue
            if best_candidate.get('deferred'):
                rejection_stats['deferred_visibility'] += 1
                self.remember_deferred_cluster(cluster)
                continue
            goal_world = best_candidate['goal']
            distance = best_candidate['path_cost']
            cluster.observation_target_world = best_candidate.get(
                'observation_target',
                cluster.centroid_world,
            )
            angular_penalty = abs(self.normalize_angle(
                math.atan2(
                    goal_world[1] - robot_pose[1],
                    goal_world[0] - robot_pose[0],
                ) - robot_pose[2]
            ))
            switch_cost = self.switch_cost(goal_world)
            information_gain = float(cluster.area_cells)
            total = (
                distance +
                0.20 * best_candidate['frontier_distance'] -
                0.40 * best_candidate['clearance'] +
                self.beta_a * angular_penalty +
                self.beta_s * switch_cost -
                self.beta_g * information_gain
            )
            candidates.append(FrontierScore(
                cluster=cluster,
                total_cost=total,
                path_cost=distance,
                angular_penalty=angular_penalty,
                switch_cost=switch_cost,
                information_gain=information_gain,
            ))
            cluster.goal_world = goal_world
        best = min(candidates, key=lambda score: score.total_cost, default=None)
        if best is not None:
            self.get_logger().warn(
                '[RALC] Publishing gated recovery frontier goal after A* validation. '
                f'cluster={best.cluster.cluster_id}, '
                f'goal=({best.cluster.goal_world[0]:.2f},'
                f'{best.cluster.goal_world[1]:.2f}), '
                f'distance={best.path_cost:.2f}'
            )
        return best

    def best_goal_candidate_for_cluster(
        self,
        cluster,
        robot_pose,
        grid,
        rejection_stats,
        shifts,
        require_astar=True,
    ):
        candidates = []
        saw_reachable_low_visibility = False
        saw_only_visibility_failures = True
        for shift in shifts:
            candidate_goal, observation_target = (
                self.observation_candidate_for_cluster(
                    grid,
                    cluster,
                    robot_pose,
                    shift,
                )
            )
            blacklist_record = self.blacklist_record_for_candidate(candidate_goal)
            if blacklist_record is not None:
                saw_only_visibility_failures = False
                self.rejected_goal_candidates.append({
                    'goal': candidate_goal,
                    'reason': 'blacklisted',
                    'label': 'OBSERVATION_POSE_REJECTED',
                })
                rejection_stats['blacklisted'] += 1
                if (
                    blacklist_record.get('failure_type') ==
                    'INEFFECTIVE_OBSERVATION_GOAL'
                ):
                    rejection_stats['ineffective_observation'] += 1
                    self.get_logger().warn(
                        '[RALC] rejected_ineffective_observation: '
                        f'cluster={cluster.cluster_id}, '
                        f'candidate=({candidate_goal[0]:.2f},'
                        f'{candidate_goal[1]:.2f})'
                    )
                continue
            reason = self.candidate_rejection_reason(
                grid,
                candidate_goal,
                cluster.centroid_world,
                robot_pose,
                require_astar=require_astar,
            )
            if reason is not None:
                saw_only_visibility_failures = False
                self.rejected_goal_candidates.append({
                    'goal': candidate_goal,
                    'reason': reason,
                    'label': 'OBSERVATION_POSE_REJECTED',
                })
                if reason == 'occupied':
                    rejection_stats['safety'] += 1
                elif reason == 'unknown':
                    rejection_stats['unknown'] += 1
                elif reason == 'costmap':
                    rejection_stats['costmap'] += 1
                elif reason == 'frontier_line_occupied':
                    rejection_stats['safety'] += 1
                elif reason == 'too_close':
                    rejection_stats['too_close'] += 1
                elif reason == 'too_far_from_frontier':
                    rejection_stats['unreachable'] += 1
                elif reason == 'unreachable':
                    rejection_stats['unreachable'] += 1
                continue

            path_cost = self.astar_path_distance(
                grid,
                robot_pose[:2],
                candidate_goal,
            )
            if path_cost is None:
                if require_astar:
                    self.rejected_goal_candidates.append({
                        'goal': candidate_goal,
                        'reason': 'unreachable',
                        'label': 'OBSERVATION_POSE_REJECTED',
                    })
                    rejection_stats['unreachable'] += 1
                    continue
                path_cost = math.hypot(
                    candidate_goal[0] - robot_pose[0],
                    candidate_goal[1] - robot_pose[1],
                )

            visible_unknown_gain = self.visible_unknown_gain(
                grid,
                candidate_goal,
                cluster,
            )
            if visible_unknown_gain < self.min_visible_unknown_gain:
                saw_reachable_low_visibility = True
                rejection_stats['observation_pose_no_visible_unknown'] += 1
                if self.reject_observation_pose_no_visible_unknown:
                    self.rejected_goal_candidates.append({
                        'goal': candidate_goal,
                        'reason': 'OBSERVATION_POSE_NO_VISIBLE_UNKNOWN',
                        'label': 'OBSERVATION_POSE_REJECTED',
                        'visible_unknown_gain': visible_unknown_gain,
                    })
                    continue
            saw_only_visibility_failures = False

            clearance = self.clearance_to_blocked_cells(grid, candidate_goal)
            frontier_distance = math.hypot(
                candidate_goal[0] - cluster.centroid_world[0],
                candidate_goal[1] - cluster.centroid_world[1],
            )
            score = (
                path_cost +
                0.20 * frontier_distance -
                0.40 * clearance -
                0.01 * float(visible_unknown_gain)
            )
            self.valid_visible_goal_candidates.append({
                'goal': candidate_goal,
                'cluster_id': cluster.cluster_id,
                'visible_unknown_gain': visible_unknown_gain,
                'observation_target': observation_target,
            })
            candidates.append({
                'goal': candidate_goal,
                'observation_target': observation_target,
                'path_cost': path_cost,
                'clearance': clearance,
                'frontier_distance': frontier_distance,
                'visible_unknown_gain': visible_unknown_gain,
                'score': score,
            })

        best = min(candidates, key=lambda item: item['score'], default=None)
        if best is not None:
            self.get_logger().info(
                '[RALC_DEBUG] selected candidate goal: '
                f'cluster={cluster.cluster_id}, '
                f'goal=({best["goal"][0]:.2f},{best["goal"][1]:.2f}), '
                f'path={best["path_cost"]:.2f}, '
                f'clearance={best["clearance"]:.2f}, '
                f'frontier_distance={best["frontier_distance"]:.2f}, '
                f'visible_unknown_gain={best["visible_unknown_gain"]}'
            )
        elif (
            saw_reachable_low_visibility and
            saw_only_visibility_failures
        ):
            self.defer_frontier_cluster(cluster, robot_pose)
            return {'deferred': True}
        return best

    def candidate_rejection_reason(
        self,
        grid,
        candidate_goal,
        frontier_world,
        robot_pose,
        require_astar=True,
    ):
        distance_to_robot = math.hypot(
            candidate_goal[0] - robot_pose[0],
            candidate_goal[1] - robot_pose[1],
        )
        if distance_to_robot < self.min_goal_distance_from_robot:
            return 'too_close'
        frontier_distance = math.hypot(
            candidate_goal[0] - frontier_world[0],
            candidate_goal[1] - frontier_world[1],
        )
        if frontier_distance > self.max_frontier_observation_distance:
            return 'too_far_from_frontier'
        cell = self.world_to_cell(grid, candidate_goal[0], candidate_goal[1])
        if cell is None:
            return 'unknown'
        image = np.array(grid.data, dtype=np.int16).reshape(
            (grid.info.height, grid.info.width)
        )
        if image[cell[1], cell[0]] > 50:
            return 'occupied'
        if image[cell[1], cell[0]] == -1:
            return 'unknown'
        if not self.goal_has_margin(grid, candidate_goal, occupied=True):
            return 'occupied'
        if not self.goal_has_margin(grid, candidate_goal, occupied=False):
            return 'unknown'
        if self.line_to_frontier_crosses_occupied(
            grid,
            candidate_goal,
            frontier_world,
        ):
            return 'frontier_line_occupied'
        if self.goal_is_in_inflated_costmap(candidate_goal):
            return 'costmap'
        if require_astar and self.astar_path_distance(
            grid,
            robot_pose[:2],
            candidate_goal,
        ) is None:
            return 'unreachable'
        return None

    def visible_unknown_gain(self, grid, candidate_goal, cluster):
        candidate_cell = self.world_to_cell(grid, candidate_goal[0], candidate_goal[1])
        if candidate_cell is None:
            return 0
        image = np.array(grid.data, dtype=np.int16).reshape(
            (grid.info.height, grid.info.width)
        )
        unknown_targets = self.cluster_unknown_neighbor_cells(image, cluster)
        visible = set()
        max_targets = 250
        if len(unknown_targets) > max_targets:
            step = max(1, len(unknown_targets) // max_targets)
            unknown_targets = unknown_targets[::step]
        for target in unknown_targets:
            if self.has_line_of_sight_to_unknown(image, candidate_cell, target):
                visible.add(target)
        return len(visible)

    def cluster_unknown_neighbor_cells(self, image, cluster):
        targets = set()
        height, width = image.shape
        for cell_x, cell_y in cluster.cells:
            x = int(cell_x)
            y = int(cell_y)
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if image[ny, nx] == -1:
                        targets.add((nx, ny))
        return sorted(targets)

    def has_line_of_sight_to_unknown(self, image, start, target):
        cells = self.bresenham_line(start[0], start[1], target[0], target[1])
        if not cells:
            return False
        for index, (x, y) in enumerate(cells):
            if x < 0 or y < 0 or y >= image.shape[0] or x >= image.shape[1]:
                return False
            value = image[y, x]
            if index == len(cells) - 1:
                return value == -1
            if value > 50:
                return False
            if value == -1:
                return False
        return False

    def line_to_frontier_crosses_occupied(self, grid, candidate_goal, frontier_world):
        start = self.world_to_cell(grid, candidate_goal[0], candidate_goal[1])
        target = self.world_to_cell(grid, frontier_world[0], frontier_world[1])
        if start is None or target is None:
            return True
        image = np.array(grid.data, dtype=np.int16).reshape(
            (grid.info.height, grid.info.width)
        )
        cells = self.bresenham_line(start[0], start[1], target[0], target[1])
        if not cells:
            return True
        for x, y in cells:
            if x < 0 or y < 0 or y >= image.shape[0] or x >= image.shape[1]:
                return True
            if image[y, x] > 50:
                return True
        return False

    def bresenham_line(self, x0, y0, x1, y1):
        x0 = int(x0)
        y0 = int(y0)
        x1 = int(x1)
        y1 = int(y1)
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        x = x0
        y = y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * error
            if e2 >= dy:
                error += dy
                x += sx
            if e2 <= dx:
                error += dx
                y += sy
        return points

    def goal_has_margin(self, grid, goal_world, occupied=True):
        cell = self.world_to_cell(grid, goal_world[0], goal_world[1])
        if cell is None:
            return False
        image = np.array(grid.data, dtype=np.int16).reshape(
            (grid.info.height, grid.info.width)
        )
        margin = (
            self.occupied_safety_margin if occupied
            else self.unknown_safety_margin
        )
        radius_cells = max(1, int(math.ceil(margin / grid.info.resolution)))
        for y in range(max(0, cell[1] - radius_cells), min(grid.info.height, cell[1] + radius_cells + 1)):
            for x in range(max(0, cell[0] - radius_cells), min(grid.info.width, cell[0] + radius_cells + 1)):
                if occupied and image[y, x] > 50:
                    return False
                if not occupied and image[y, x] == -1:
                    return False
        return True

    def goal_is_in_inflated_costmap(self, goal_world):
        if self.latest_global_costmap is None:
            return False
        cell = self.world_to_cell(
            self.latest_global_costmap,
            goal_world[0],
            goal_world[1],
        )
        if cell is None:
            return True
        data = np.array(self.latest_global_costmap.data, dtype=np.int16).reshape(
            (
                self.latest_global_costmap.info.height,
                self.latest_global_costmap.info.width,
            )
        )
        return data[cell[1], cell[0]] != 0

    def clearance_to_blocked_cells(self, grid, goal_world):
        cell = self.world_to_cell(grid, goal_world[0], goal_world[1])
        if cell is None:
            return 0.0
        image = np.array(grid.data, dtype=np.int16).reshape(
            (grid.info.height, grid.info.width)
        )
        max_radius_cells = max(1, int(math.ceil(1.5 / grid.info.resolution)))
        best = max_radius_cells
        for y in range(max(0, cell[1] - max_radius_cells), min(grid.info.height, cell[1] + max_radius_cells + 1)):
            for x in range(max(0, cell[0] - max_radius_cells), min(grid.info.width, cell[0] + max_radius_cells + 1)):
                if image[y, x] <= 50 and image[y, x] != -1:
                    continue
                dist = math.hypot(x - cell[0], y - cell[1])
                if dist < best:
                    best = dist
        return best * grid.info.resolution

    def astar_path_distance(self, grid, start_world, goal_world):
        start = self.world_to_cell(grid, start_world[0], start_world[1])
        goal = self.world_to_cell(grid, goal_world[0], goal_world[1])
        if start is None or goal is None:
            return None
        image = np.array(grid.data, dtype=np.int16).reshape(
            (grid.info.height, grid.info.width)
        )
        start = self.nearest_free_cell(image, start, radius_cells=12)
        goal = self.nearest_free_cell(image, goal, radius_cells=16)
        if start is None or goal is None:
            return None
        queue = [(0.0, start)]
        costs = {start: 0.0}
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
        while queue:
            _priority, current = heapq.heappop(queue)
            if current == goal:
                return costs[current] * grid.info.resolution
            for dx, dy in neighbors:
                nx = current[0] + dx
                ny = current[1] + dy
                if nx < 0 or ny < 0 or nx >= grid.info.width or ny >= grid.info.height:
                    continue
                if image[ny, nx] != 0:
                    continue
                step = math.sqrt(2.0) if dx and dy else 1.0
                new_cost = costs[current] + step
                neighbor = (nx, ny)
                if neighbor in costs and new_cost >= costs[neighbor]:
                    continue
                costs[neighbor] = new_cost
                heuristic = math.hypot(goal[0] - nx, goal[1] - ny)
                heapq.heappush(queue, (new_cost + heuristic, neighbor))
        return None

    def nearest_free_cell(self, image, cell, radius_cells):
        if image[cell[1], cell[0]] == 0:
            return cell
        best = None
        best_dist = None
        for radius in range(1, radius_cells + 1):
            for y in range(max(0, cell[1] - radius), min(image.shape[0], cell[1] + radius + 1)):
                for x in range(max(0, cell[0] - radius), min(image.shape[1], cell[0] + radius + 1)):
                    if image[y, x] != 0:
                        continue
                    dist = math.hypot(x - cell[0], y - cell[1])
                    if best_dist is None or dist < best_dist:
                        best = (x, y)
                        best_dist = dist
            if best is not None:
                return best
        return None

    def goal_is_safe(self, grid, goal_world):
        return (
            self.goal_has_margin(grid, goal_world, occupied=True) and
            self.goal_has_margin(grid, goal_world, occupied=False) and
            not self.goal_is_in_inflated_costmap(goal_world)
        )

    def frontier_status_reason(self, raw_mask, region_mask, clusters, rejection_stats):
        if int(region_mask.sum()) == 0 and not clusters:
            return 'NO_FRONTIER_IN_REGION'
        if (
            int(rejection_stats.get('deferred_visibility', 0)) >=
            self.max_deferred_frontiers_before_region_growth
        ):
            return 'NO_ACTIONABLE_FRONTIER_IN_REGION'
        if self.all_region_clusters_are_non_actionable(clusters, rejection_stats):
            if not self.all_cluster_rejections_are_explicit(
                clusters,
                rejection_stats,
            ):
                return 'PLANNER_CLASSIFICATION_ERROR'
            return 'NO_ACTIONABLE_FRONTIER_IN_REGION'
        if clusters and rejection_stats['safety'] >= len(clusters):
            return 'FRONTIERS_REJECTED_BY_SAFETY'
        if clusters and rejection_stats['too_close'] >= len(clusters):
            return 'FRONTIERS_REJECTED_BY_SAFETY'
        if int(region_mask.sum()) > 0:
            return 'FRONTIERS_UNREACHABLE'
        if int(raw_mask.sum()) == 0:
            return 'NO_GLOBAL_FRONTIER'
        if not clusters:
            return 'NO_FRONTIER_IN_REGION'
        return 'FRONTIERS_UNREACHABLE'

    def publish_no_frontier(
        self,
        reason,
        cluster_count,
        region_frontier_cells,
        occupancy_stats,
        actionable_clusters,
        non_actionable_clusters,
        observation_report=None,
    ):
        deferred_count = self.active_deferred_frontier_count()
        completion_allowed = (
            self.completion_allowed_by_unknown(occupancy_stats) and
            deferred_count == 0
        )
        payload = {
            'reason': reason,
            'region_id': self.current_region.get('region_id'),
            'clusters': cluster_count,
            'frontier_cells_in_region': region_frontier_cells,
            'actionable_clusters': actionable_clusters,
            'non_actionable_clusters': non_actionable_clusters,
            'completion_allowed': completion_allowed,
            'deferred_frontier_clusters': deferred_count,
            'map_update_count': self.map_update_count,
            **occupancy_stats,
        }
        if observation_report:
            payload.update(observation_report)
        msg = String()
        msg.data = json.dumps(payload)
        self.no_frontier_pub.publish(msg)
        self.get_logger().info(f'[RALC] no_frontier_in_region: {msg.data}')

    def publish_frontier_status(
        self,
        reason,
        cluster_count,
        region_frontier_cells,
        rejection_stats,
        occupancy_stats,
        actionable_clusters,
        non_actionable_clusters,
        selected_score=None,
        observation_report=None,
    ):
        repeated_count = self.max_rejection_count_for_current_region()
        deferred_count = self.active_deferred_frontier_count()
        completion_allowed = (
            self.completion_allowed_by_unknown(occupancy_stats) and
            deferred_count == 0
        )
        payload = {
            'reason': reason,
            'region_id': self.current_region.get('region_id'),
            'clusters': cluster_count,
            'frontier_cells_in_region': region_frontier_cells,
            'actionable_clusters': actionable_clusters,
            'non_actionable_clusters': non_actionable_clusters,
            'completion_allowed': completion_allowed,
            'rejections': rejection_stats,
            'rejected_too_small': int(rejection_stats.get('too_small', 0)),
            'rejected_by_safety': int(rejection_stats.get('safety', 0)),
            'rejected_too_close': int(rejection_stats.get('too_close', 0)),
            'rejected_unreachable': int(rejection_stats.get('unreachable', 0)),
            'rejected_blacklisted': int(rejection_stats.get('blacklisted', 0)),
            'rejected_ineffective_observation': int(
                rejection_stats.get('ineffective_observation', 0)
            ),
            'rejected_observation_pose_no_visible_unknown': int(
                rejection_stats.get('observation_pose_no_visible_unknown', 0)
            ),
            'deferred_frontier_clusters': deferred_count,
            'deferred_frontiers_before_region_growth': (
                self.max_deferred_frontiers_before_region_growth
            ),
            'repeated_count': repeated_count,
            'map_update_count': self.map_update_count,
            **occupancy_stats,
        }
        if selected_score is not None:
            observation_target = getattr(
                selected_score.cluster,
                'observation_target_world',
                selected_score.cluster.centroid_world,
            )
            observation_yaw = math.atan2(
                observation_target[1] - selected_score.cluster.goal_world[1],
                observation_target[0] - selected_score.cluster.goal_world[0],
            )
            payload.update({
                'selected_cluster_id': selected_score.cluster.cluster_id,
                'selected_goal_x': selected_score.cluster.goal_world[0],
                'selected_goal_y': selected_score.cluster.goal_world[1],
                'selected_centroid_x': selected_score.cluster.centroid_world[0],
                'selected_centroid_y': selected_score.cluster.centroid_world[1],
                'selected_observation_target_x': observation_target[0],
                'selected_observation_target_y': observation_target[1],
                'selected_observation_yaw': observation_yaw,
            })
        if observation_report:
            payload.update(observation_report)
        msg = String()
        msg.data = json.dumps(payload)
        self.status_pub.publish(msg)
        if reason == 'REGION_STATS':
            self.get_logger().info(f'[RALC] frontier_planner_status: {msg.data}')
        else:
            self.get_logger().warn(f'[RALC] frontier_planner_status: {msg.data}')

    def completion_allowed_by_unknown(self, occupancy_stats):
        return (
            float(
                occupancy_stats.get(
                    'reachable_unknown_ratio_in_region',
                    occupancy_stats.get('unknown_ratio_in_region', 0.0),
                )
            ) <= self.region_reachable_unknown_completion_threshold
        )

    def non_actionable_cluster_count(self, clusters, rejection_stats):
        explicit_rejections = (
            int(rejection_stats.get('too_small', 0)) +
            int(rejection_stats.get('safety', 0)) +
            int(rejection_stats.get('unreachable', 0)) +
            int(rejection_stats.get('too_close', 0)) +
            int(rejection_stats.get('non_actionable', 0)) +
            int(rejection_stats.get('blacklisted', 0)) +
            int(rejection_stats.get('ineffective_observation', 0)) +
            int(rejection_stats.get('costmap', 0)) +
            int(rejection_stats.get('unknown', 0))
        )
        return min(len(clusters), explicit_rejections)

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def nearest_cluster_to_point(self, clusters, point, region_id):
        best = None
        best_distance = None
        for cluster in clusters:
            distance = math.hypot(
                cluster.goal_world[0] - point[0],
                cluster.goal_world[1] - point[1],
            )
            centroid_distance = math.hypot(
                cluster.centroid_world[0] - point[0],
                cluster.centroid_world[1] - point[1],
            )
            distance = min(distance, centroid_distance)
            if best_distance is None or distance < best_distance:
                best = cluster
                best_distance = distance
        if best is None or best_distance is None:
            return None
        if best_distance > self.failed_goal_aliasing_distance:
            return None
        return best, best_distance

    def point_identity_for_region(self, x, y, region_id):
        rounded_x = round(float(x) / 0.25) * 0.25
        rounded_y = round(float(y) / 0.25) * 0.25
        return int(region_id), rounded_x, rounded_y

    def cluster_identity_for_region(self, cluster, region_id):
        return self.point_identity_for_region(
            cluster.centroid_world[0],
            cluster.centroid_world[1],
            region_id,
        )

    def cluster_identity(self, cluster):
        region_id = self.current_region.get('region_id') if self.current_region else 0
        return self.cluster_identity_for_region(cluster, int(region_id))

    def cluster_rejection_key(self, cluster, reason):
        region_id, rounded_x, rounded_y = self.cluster_identity(cluster)
        return region_id, rounded_x, rounded_y, reason

    def record_cluster_rejection(self, cluster, reason):
        key = self.cluster_rejection_key(cluster, reason)
        count = self.non_actionable_rejection_counts.get(key, 0) + 1
        self.non_actionable_rejection_counts[key] = count
        if count >= self.max_non_actionable_rejections:
            cluster_id = self.cluster_identity(cluster)
            self.non_actionable_cluster_ids.add(cluster_id)
            self.non_actionable_cluster_reasons[cluster_id] = reason
            self.remember_non_actionable_marker(cluster)
            self.get_logger().warn(
                '[RALC] Marked frontier cluster as non-actionable artifact: '
                f'region={key[0]}, centroid=({cluster.centroid_world[0]:.2f},'
                f'{cluster.centroid_world[1]:.2f}), reason={reason}, '
                f'rejections={count}'
            )
        elif reason == 'too_small':
            self.remember_non_actionable_marker(cluster)

    def cluster_is_non_actionable(self, cluster):
        return self.cluster_identity(cluster) in self.non_actionable_cluster_ids

    def remember_non_actionable_marker(self, cluster):
        if not any(
            existing.cluster_id == cluster.cluster_id
            for existing in self.last_non_actionable_clusters
        ):
            self.last_non_actionable_clusters.append(cluster)

    def max_rejection_count_for_current_region(self):
        if self.current_region is None:
            return 0
        region_id = int(self.current_region.get('region_id', 0))
        counts = [
            count for key, count in self.non_actionable_rejection_counts.items()
            if key[0] == region_id
        ]
        return max(counts, default=0)

    def failed_goal_count_for_current_region(self):
        if self.current_region is None:
            return 0
        region_id = int(self.current_region.get('region_id', 0))
        self.prune_expired_failed_goal_blacklist()
        return sum(
            1 for record in self.failed_goal_blacklist.values()
            if int(record.get('region_id', -1)) == region_id
        )

    def prune_expired_failed_goal_blacklist(self):
        if not self.failed_goal_blacklist:
            return
        now = self.now_sec()
        expired = [
            key for key, record in self.failed_goal_blacklist.items()
            if now - float(record.get('timestamp', 0.0)) > float(
                record.get('blacklist_seconds', self.failed_goal_blacklist_seconds)
            )
        ]
        for key in expired:
            self.failed_goal_blacklist.pop(key, None)

    def cluster_is_failed_goal_blacklisted(self, cluster):
        return self.blacklist_record_for_cluster(cluster) is not None

    def blacklist_record_for_candidate(self, candidate_goal):
        if self.current_region is None:
            return None
        self.prune_expired_failed_goal_blacklist()
        region_id = int(self.current_region.get('region_id', 0))
        for record in self.failed_goal_blacklist.values():
            if int(record.get('region_id', -1)) != region_id:
                continue
            aliasing_distance = float(
                record.get('aliasing_distance', self.failed_goal_aliasing_distance)
            )
            failed_goal = (
                float(record.get('failed_goal_x', 0.0)),
                float(record.get('failed_goal_y', 0.0)),
            )
            if math.hypot(
                candidate_goal[0] - failed_goal[0],
                candidate_goal[1] - failed_goal[1],
            ) <= aliasing_distance:
                return record
        return None

    def blacklist_record_for_cluster(self, cluster):
        if self.current_region is None:
            return None
        self.prune_expired_failed_goal_blacklist()
        region_id = int(self.current_region.get('region_id', 0))
        cluster_key = self.cluster_identity_for_region(cluster, region_id)
        if cluster_key in self.failed_goal_blacklist:
            return self.failed_goal_blacklist[cluster_key]
        for record in self.failed_goal_blacklist.values():
            if int(record.get('region_id', -1)) != region_id:
                continue
            aliasing_distance = float(
                record.get('aliasing_distance', self.failed_goal_aliasing_distance)
            )
            failed_goal = (
                float(record.get('failed_goal_x', 0.0)),
                float(record.get('failed_goal_y', 0.0)),
            )
            if math.hypot(
                cluster.goal_world[0] - failed_goal[0],
                cluster.goal_world[1] - failed_goal[1],
            ) <= aliasing_distance:
                return record
            if math.hypot(
                cluster.centroid_world[0] - failed_goal[0],
                cluster.centroid_world[1] - failed_goal[1],
            ) <= aliasing_distance:
                return record
        return None

    def evaluate_pending_observation(self, clusters, frontier_cells_after):
        if (
            self.pending_observation_goal is None or
            self.pending_observation_execution is None
        ):
            self.last_observation_report = None
            return None
        pending = self.pending_observation_goal
        execution = self.pending_observation_execution
        if self.current_region is None:
            self.pending_observation_goal = None
            self.pending_observation_execution = None
            self.last_observation_report = None
            return None
        region_id = int(self.current_region.get('region_id'))
        if int(pending.get('region_id', -1)) != region_id:
            self.pending_observation_goal = None
            self.pending_observation_execution = None
            self.last_observation_report = None
            return None

        frontiers_before = int(
            pending.get('previous_frontier_cells_in_region', frontier_cells_after)
        )
        clusters_before = int(pending.get('previous_clusters', len(clusters)))
        cluster, cluster_distance = self.find_cluster_near_centroid(
            clusters,
            (
                float(pending.get('selected_centroid_x')),
                float(pending.get('selected_centroid_y')),
            ),
            self.ineffective_goal_aliasing_distance,
        )
        selected_cluster_still_present = cluster is not None
        map_update_increased = (
            self.map_update_count > int(pending.get('map_update_count_before', -1))
        )
        travel = float(execution.get('robot_travel_distance') or 0.0)
        frontier_reduction = max(0, frontiers_before - int(frontier_cells_after))
        frontier_reduction_ratio = (
            float(frontier_reduction) / float(frontiers_before)
            if frontiers_before > 0 else 0.0
        )
        no_travel = travel < self.min_observation_travel_distance
        no_frontier_reduction = (
            frontier_reduction_ratio < self.min_frontier_reduction_ratio
        )
        ineffective = (
            no_travel and
            no_frontier_reduction and
            selected_cluster_still_present
        )
        reason = 'OBSERVATION_EFFECTIVE'
        if ineffective:
            reason = 'NO_TRAVEL_NO_FRONTIER_REDUCTION'
            cluster_key = self.point_identity_for_region(
                float(pending.get('selected_centroid_x')),
                float(pending.get('selected_centroid_y')),
                region_id,
            )
            attempts = self.ineffective_observation_attempts.get(cluster_key, 0) + 1
            self.ineffective_observation_attempts[cluster_key] = attempts
            self.failed_goal_blacklist[cluster_key] = {
                'region_id': region_id,
                'centroid_x': float(pending.get('selected_centroid_x')),
                'centroid_y': float(pending.get('selected_centroid_y')),
                'failed_goal_x': float(pending.get('selected_goal_x')),
                'failed_goal_y': float(pending.get('selected_goal_y')),
                'failure_type': 'INEFFECTIVE_OBSERVATION_GOAL',
                'message': reason,
                'timestamp': self.now_sec(),
                'blacklist_seconds': self.ineffective_goal_blacklist_seconds,
                'aliasing_distance': self.ineffective_goal_aliasing_distance,
            }
            self.get_logger().warn(
                '[RALC] Observation result: '
                f'cluster={pending.get("selected_cluster_id")}, '
                f'goal=({float(pending.get("selected_goal_x")):.2f},'
                f'{float(pending.get("selected_goal_y")):.2f}), '
                f'travel={travel:.2f}m, '
                f'frontiers_before={frontiers_before}, '
                f'frontiers_after={int(frontier_cells_after)}, '
                f'clusters_before={clusters_before}, '
                f'clusters_after={len(clusters)}, '
                f'map_update_count_before={pending.get("map_update_count_before")}, '
                f'map_update_count_after={self.map_update_count}, '
                f'effective=false, reason={reason}. '
                'Blacklisting observation goal.'
            )
        else:
            if travel >= self.min_observation_travel_distance:
                reason = 'ROBOT_TRAVEL'
            elif frontier_reduction_ratio >= self.min_frontier_reduction_ratio:
                reason = 'FRONTIER_REDUCTION'
            elif not selected_cluster_still_present:
                reason = 'SELECTED_CLUSTER_DISAPPEARED'
            self.get_logger().info(
                '[RALC] Observation result: '
                f'cluster={pending.get("selected_cluster_id")}, '
                f'goal=({float(pending.get("selected_goal_x")):.2f},'
                f'{float(pending.get("selected_goal_y")):.2f}), '
                f'travel={travel:.2f}m, '
                f'frontiers_before={frontiers_before}, '
                f'frontiers_after={int(frontier_cells_after)}, '
                f'clusters_before={clusters_before}, '
                f'clusters_after={len(clusters)}, '
                f'effective=true, reason={reason}.'
            )

        report = {
            'previous_frontier_cells_in_region': frontiers_before,
            'previous_clusters': clusters_before,
            'selected_cluster_id': pending.get('selected_cluster_id'),
            'selected_goal_x': pending.get('selected_goal_x'),
            'selected_goal_y': pending.get('selected_goal_y'),
            'selected_centroid_x': pending.get('selected_centroid_x'),
            'selected_centroid_y': pending.get('selected_centroid_y'),
            'frontier_cells_after': int(frontier_cells_after),
            'clusters_after': len(clusters),
            'selected_cluster_still_present': selected_cluster_still_present,
            'selected_cluster_distance': cluster_distance,
            'map_update_count_before': pending.get('map_update_count_before'),
            'map_update_count_after': self.map_update_count,
            'map_update_count_increased': map_update_increased,
            'robot_travel_distance': travel,
            'frontier_reduction_ratio': frontier_reduction_ratio,
            'observation_effective': not ineffective,
            'observation_reason': reason,
        }
        self.last_observation_report = report
        self.pending_observation_goal = None
        self.pending_observation_execution = None
        return report

    def find_cluster_near_centroid(self, clusters, centroid, max_distance):
        best = None
        best_distance = None
        for cluster in clusters:
            distance = math.hypot(
                cluster.centroid_world[0] - centroid[0],
                cluster.centroid_world[1] - centroid[1],
            )
            if best_distance is None or distance < best_distance:
                best = cluster
                best_distance = distance
        if best is None or best_distance is None or best_distance > max_distance:
            return None, None
        return best, best_distance

    def defer_frontier_cluster(self, cluster, robot_pose):
        key = self.cluster_identity(cluster)
        now = self.now_sec()
        self.deferred_frontier_clusters[key] = {
            'region_id': key[0],
            'centroid_x': cluster.centroid_world[0],
            'centroid_y': cluster.centroid_world[1],
            'cluster_id': cluster.cluster_id,
            'first_seen_sec': self.deferred_frontier_clusters.get(
                key, {}
            ).get('first_seen_sec', now),
            'last_seen_sec': now,
            'robot_x': robot_pose[0],
            'robot_y': robot_pose[1],
            'reason': 'DEFERRED_WALL_BLOCKED_OR_NO_VIEWPOINT',
        }
        self.remember_deferred_cluster(cluster)
        self.get_logger().warn(
            '[RALC] DEFERRED_FRONTIER_CLUSTER: '
            f'cluster={cluster.cluster_id}, '
            f'centroid=({cluster.centroid_world[0]:.2f},'
            f'{cluster.centroid_world[1]:.2f}), '
            'reason=DEFERRED_WALL_BLOCKED_OR_NO_VIEWPOINT'
        )

    def remember_deferred_cluster(self, cluster):
        if not any(existing.cluster_id == cluster.cluster_id for existing in self.last_deferred_clusters):
            self.last_deferred_clusters.append(cluster)

    def should_recheck_deferred_frontiers(self, robot_pose):
        if not self.deferred_frontier_clusters:
            return False
        now = self.now_sec()
        for record in self.deferred_frontier_clusters.values():
            if (
                now - float(record.get('last_seen_sec', 0.0)) >=
                self.deferred_frontier_timeout_sec
            ):
                return True
        if self.last_deferred_recheck_robot_pose is None:
            return True
        return (
            math.hypot(
                robot_pose[0] - self.last_deferred_recheck_robot_pose[0],
                robot_pose[1] - self.last_deferred_recheck_robot_pose[1],
            ) >= self.deferred_frontier_recheck_distance
        )

    def active_deferred_frontier_count(self):
        if self.current_region is None:
            return 0
        region_id = int(self.current_region.get('region_id', 0) or 0)
        now = self.now_sec()
        return sum(
            1 for record in self.deferred_frontier_clusters.values()
            if (
                int(record.get('region_id', -1)) == region_id and
                now - float(record.get('last_seen_sec', 0.0)) <=
                self.deferred_frontier_timeout_sec
            )
        )

    def all_region_clusters_are_non_actionable(self, clusters, rejection_stats):
        if not clusters:
            return False
        rejected = (
            int(rejection_stats.get('too_small', 0)) +
            int(rejection_stats.get('safety', 0)) +
            int(rejection_stats.get('unreachable', 0)) +
            int(rejection_stats.get('too_close', 0)) +
            int(rejection_stats.get('non_actionable', 0)) +
            int(rejection_stats.get('blacklisted', 0)) +
            int(rejection_stats.get('ineffective_observation', 0))
        )
        if rejected < len(clusters):
            return False
        if int(rejection_stats.get('too_small', 0)) == len(clusters):
            return True
        if int(rejection_stats.get('non_actionable', 0)) == len(clusters):
            return True
        return (
            self.max_rejection_count_for_current_region() >=
            self.max_non_actionable_rejections
        )

    def all_cluster_rejections_are_explicit(self, clusters, rejection_stats):
        explicit = (
            int(rejection_stats.get('too_small', 0)) +
            int(rejection_stats.get('safety', 0)) +
            int(rejection_stats.get('unreachable', 0)) +
            int(rejection_stats.get('blacklisted', 0)) +
            int(rejection_stats.get('ineffective_observation', 0)) +
            int(rejection_stats.get('costmap', 0)) +
            int(rejection_stats.get('unknown', 0))
        )
        return bool(clusters) and explicit >= len(clusters)

    def log_disappeared_frontier_clusters(
        self,
        grid,
        raw_mask,
        region_mask,
        clusters,
    ):
        """Explain why previously visible clusters are not in this cycle.

        The labels are diagnostic only. They help separate true map discovery
        from planner filtering and RViz marker churn.
        """
        if self.current_region is None:
            return
        region_id = int(self.current_region.get('region_id', 0))
        previous = self.previous_cluster_snapshots.get(region_id, {})
        current = {
            self.cluster_identity_for_region(cluster, region_id): cluster
            for cluster in clusters
        }

        for key, old in previous.items():
            if key in current:
                continue
            reason = self.frontier_disappearance_reason(
                grid,
                raw_mask,
                region_mask,
                old,
                key,
            )
            self.get_logger().info(
                '[RALC_DEBUG] frontier_cluster_disappeared: '
                f'region={region_id}, centroid=({old["x"]:.2f},'
                f'{old["y"]:.2f}), reason={reason}'
            )

        self.previous_cluster_snapshots[region_id] = {
            key: {
                'x': cluster.centroid_world[0],
                'y': cluster.centroid_world[1],
            }
            for key, cluster in current.items()
        }

    def frontier_disappearance_reason(self, grid, raw_mask, region_mask, old, key):
        if key in self.non_actionable_cluster_ids:
            return 'rejected_non_actionable'
        cell = self.world_to_cell(grid, old['x'], old['y'])
        if cell is None:
            return 'moved_outside_region'
        x, y = cell
        if raw_mask[y, x] and not region_mask[y, x]:
            return 'moved_outside_region'
        if not raw_mask[y, x]:
            image = np.array(grid.data, dtype=np.int16).reshape(
                (grid.info.height, grid.info.width)
            )
            if self.cell_has_unknown_neighbor(image, x, y):
                return 'marker_lifetime_expired'
            if image[y, x] == -1:
                return 'consumed_by_map_update'
            return 'no_longer_adjacent_to_unknown'
        return 'marker_lifetime_expired'

    def cell_has_unknown_neighbor(self, image, x, y):
        for ny in range(max(0, y - 1), min(image.shape[0], y + 2)):
            for nx in range(max(0, x - 1), min(image.shape[1], x + 2)):
                if nx == x and ny == y:
                    continue
                if image[ny, nx] == -1:
                    return True
        return False

    def switch_cost(self, goal_world):
        if self.previous_selected_frontier is None:
            return 0.0
        distance = math.hypot(
            goal_world[0] - self.previous_selected_frontier[0],
            goal_world[1] - self.previous_selected_frontier[1],
        )
        return 0.0 if distance <= self.same_frontier_distance else 1.0

    def shift_goal_toward_robot(
        self,
        frontier_world,
        robot_x,
        robot_y,
        shift_override=None,
    ):
        dx = robot_x - frontier_world[0]
        dy = robot_y - frontier_world[1]
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            return frontier_world
        desired_shift = (
            self.frontier_goal_inward_shift
            if shift_override is None else float(shift_override)
        )
        shift = min(desired_shift, max(0.0, distance - 0.05))
        return (
            frontier_world[0] + shift * dx / distance,
            frontier_world[1] + shift * dy / distance,
        )

    def observation_candidate_for_cluster(self, grid, cluster, robot_pose, shift):
        unknown_centroid = self.cluster_unknown_centroid_world(grid, cluster)
        if unknown_centroid is None:
            return (
                self.shift_goal_toward_robot(
                    cluster.centroid_world,
                    robot_pose[0],
                    robot_pose[1],
                    shift,
                ),
                cluster.centroid_world,
            )
        dx = unknown_centroid[0] - cluster.centroid_world[0]
        dy = unknown_centroid[1] - cluster.centroid_world[1]
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            return (
                self.shift_goal_toward_robot(
                    cluster.centroid_world,
                    robot_pose[0],
                    robot_pose[1],
                    shift,
                ),
                unknown_centroid,
            )
        # Frontier cells sit between free and unknown space. A useful
        # observation pose is on the known/free side, facing the unknown side.
        return (
            (
                cluster.centroid_world[0] - float(shift) * dx / distance,
                cluster.centroid_world[1] - float(shift) * dy / distance,
            ),
            unknown_centroid,
        )

    def cluster_unknown_centroid_world(self, grid, cluster):
        image = np.array(grid.data, dtype=np.int16).reshape(
            (grid.info.height, grid.info.width)
        )
        unknown_cells = self.cluster_unknown_neighbor_cells(image, cluster)
        if not unknown_cells:
            return None
        xs = [cell[0] for cell in unknown_cells]
        ys = [cell[1] for cell in unknown_cells]
        return self.map_cell_to_world(
            grid,
            float(sum(xs)) / float(len(xs)),
            float(sum(ys)) / float(len(ys)),
        )

    def world_to_cell(self, grid, world_x, world_y):
        origin = grid.info.origin.position
        cell_x = int((world_x - origin.x) / grid.info.resolution)
        cell_y = int((world_y - origin.y) / grid.info.resolution)
        if (
            cell_x < 0 or cell_x >= grid.info.width or
            cell_y < 0 or cell_y >= grid.info.height
        ):
            return None
        return cell_x, cell_y

    def map_cell_to_world(self, grid, cell_x, cell_y):
        origin = grid.info.origin.position
        resolution = grid.info.resolution
        return (
            float(origin.x + (cell_x + 0.5) * resolution),
            float(origin.y + (cell_y + 0.5) * resolution),
        )

    def make_goal_msg(self, grid, goal_world, frontier_world=None):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = grid.header.frame_id or self.map_frame
        msg.pose.position.x = goal_world[0]
        msg.pose.position.y = goal_world[1]
        yaw = 0.0
        if frontier_world is not None:
            yaw = math.atan2(
                frontier_world[1] - goal_world[1],
                frontier_world[0] - goal_world[0],
            )
        msg.pose.orientation.z = math.sin(0.5 * yaw)
        msg.pose.orientation.w = math.cos(0.5 * yaw)
        return msg

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def update_region_mask_cache(self, grid, reachable_free):
        if self.current_region is None:
            return
        region_id = int(self.current_region.get('region_id', 0) or 0)
        if region_id <= 0:
            return
        ys, xs = np.nonzero(reachable_free)
        origin = grid.info.origin.position
        resolution = float(grid.info.resolution)
        points = [
            (
                float(origin.x + (float(x) + 0.5) * resolution),
                float(origin.y + (float(y) + 0.5) * resolution),
            )
            for x, y in zip(xs, ys)
        ]
        status = self.current_region.get('status', 'ACTIVE')
        self.region_mask_cache[region_id] = {
            'region_id': region_id,
            'status': status,
            'points': points,
            'center_x': float(self.current_region.get('center_x', 0.0)),
            'center_y': float(self.current_region.get('center_y', 0.0)),
            'resolution': resolution,
            'frame_id': grid.header.frame_id or self.map_frame,
        }

    def region_color(self, region_id):
        index = (int(region_id) - 1) % len(self.region_color_palette)
        return self.region_color_palette[index]

    def region_mask_alpha(self, status):
        if status == 'ACTIVE':
            return 0.45
        if status in ('REFINEMENT', 'REGION_REFINEMENT'):
            return 0.30
        return 0.18

    def publish_region_mask_markers(self, grid):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        fallback_frame = grid.header.frame_id or self.map_frame
        for region_id in sorted(self.region_mask_cache):
            record = self.region_mask_cache[region_id]
            status = record.get('status', 'COMPLETED')
            color = self.region_color(region_id)
            alpha = self.region_mask_alpha(status)
            frame_id = record.get('frame_id', fallback_frame)
            resolution = float(record.get('resolution', grid.info.resolution))

            mask_marker = Marker()
            mask_marker.header.stamp = stamp
            mask_marker.header.frame_id = frame_id
            mask_marker.ns = 'ralc_region_mask_markers'
            mask_marker.id = int(region_id)
            mask_marker.type = Marker.CUBE_LIST
            mask_marker.action = Marker.ADD
            mask_marker.pose.orientation.w = 1.0
            mask_marker.scale.x = resolution
            mask_marker.scale.y = resolution
            mask_marker.scale.z = 0.025
            mask_marker.color.r = color[0]
            mask_marker.color.g = color[1]
            mask_marker.color.b = color[2]
            mask_marker.color.a = alpha
            for x, y in record.get('points', []):
                point = Point()
                point.x = x
                point.y = y
                point.z = 0.055
                mask_marker.points.append(point)
            marker_array.markers.append(mask_marker)

            label = Marker()
            label.header.stamp = stamp
            label.header.frame_id = frame_id
            label.ns = 'ralc_region_mask_labels'
            label.id = 10000 + int(region_id)
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(record.get('center_x', 0.0))
            label.pose.position.y = float(record.get('center_y', 0.0))
            label.pose.position.z = 0.75
            label.pose.orientation.w = 1.0
            label.scale.z = 0.24
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 0.95
            label.text = f'Region {region_id} {status}'
            marker_array.markers.append(label)

        self.region_mask_marker_pub.publish(marker_array)

    def publish_frontier_markers(
        self,
        grid,
        raw_mask,
        region_mask,
        clusters,
        best,
        reachable_free,
        best_is_recovery=False,
    ):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        frame_id = grid.header.frame_id or self.map_frame
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)
        outside_region_mask = raw_mask & ~region_mask
        self.append_points_marker(marker_array, stamp, frame_id, grid, outside_region_mask,
                                  'ralc_raw_frontier_cells', 10000,
                                  (0.75, 0.2, 1.0, 0.85), 0.06)
        self.append_points_marker(marker_array, stamp, frame_id, grid, region_mask,
                                  'ralc_region_frontier_cells', 10001,
                                  (0.1, 0.55, 1.0, 0.95), 0.08)
        best_id = best.cluster.cluster_id if best else None
        for cluster in clusters:
            is_best = cluster.cluster_id == best_id
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = frame_id
            marker.ns = 'ralc_frontier_centroids'
            marker.id = cluster.cluster_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = cluster.centroid_world[0]
            marker.pose.position.y = cluster.centroid_world[1]
            marker.pose.position.z = 0.18
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.25 if is_best else 0.15
            marker.scale.y = 0.25 if is_best else 0.15
            marker.scale.z = 0.25 if is_best else 0.15
            marker.color.r = 1.0 if is_best else 0.2
            marker.color.g = 0.8 if is_best else 0.6
            marker.color.b = 0.1 if is_best else 1.0
            marker.color.a = 0.95
            marker_array.markers.append(marker)
            if is_best:
                arrow = Marker()
                arrow.header.stamp = stamp
                arrow.header.frame_id = frame_id
                arrow.ns = 'ralc_frontier_goal'
                arrow.id = 1
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.pose.position.x = cluster.goal_world[0]
                arrow.pose.position.y = cluster.goal_world[1]
                arrow.pose.position.z = 0.2
                observation_target = getattr(
                    cluster,
                    'observation_target_world',
                    cluster.centroid_world,
                )
                yaw = math.atan2(
                    observation_target[1] - cluster.goal_world[1],
                    observation_target[0] - cluster.goal_world[0],
                )
                arrow.pose.orientation.z = math.sin(0.5 * yaw)
                arrow.pose.orientation.w = math.cos(0.5 * yaw)
                arrow.scale.x = 0.4
                arrow.scale.y = 0.08
                arrow.scale.z = 0.08
                arrow.color.r = 0.05 if not best_is_recovery else 1.0
                arrow.color.g = 0.95 if not best_is_recovery else 0.45
                arrow.color.b = 0.0
                arrow.color.a = 0.95
                marker_array.markers.append(arrow)
        self.append_valid_visible_goal_markers(marker_array, stamp, frame_id)
        self.append_deferred_frontier_markers(marker_array, stamp, frame_id)
        self.append_rejected_candidate_markers(marker_array, stamp, frame_id)
        self.marker_pub.publish(marker_array)
        self.legacy_marker_pub.publish(marker_array)

    def append_valid_visible_goal_markers(self, marker_array, stamp, frame_id):
        for index, candidate in enumerate(self.valid_visible_goal_candidates):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = frame_id
            marker.ns = 'ralc_valid_visible_observation_goals'
            marker.id = 20000 + index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = candidate['goal'][0]
            marker.pose.position.y = candidate['goal'][1]
            marker.pose.position.z = 0.26
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.13
            marker.scale.y = 0.13
            marker.scale.z = 0.13
            marker.color.r = 0.05
            marker.color.g = 1.0
            marker.color.b = 0.15
            marker.color.a = 0.85
            marker_array.markers.append(marker)

            label = Marker()
            label.header.stamp = stamp
            label.header.frame_id = frame_id
            label.ns = 'ralc_valid_visible_observation_goal_labels'
            label.id = 21000 + index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = candidate['goal'][0]
            label.pose.position.y = candidate['goal'][1]
            label.pose.position.z = 0.48
            label.pose.orientation.w = 1.0
            label.scale.z = 0.14
            label.color.r = 0.4
            label.color.g = 1.0
            label.color.b = 0.4
            label.color.a = 0.9
            label.text = 'FRONTIER_DISCOVERED'
            marker_array.markers.append(label)

    def append_deferred_frontier_markers(self, marker_array, stamp, frame_id):
        for index, cluster in enumerate(self.last_deferred_clusters):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = frame_id
            marker.ns = 'ralc_deferred_frontier_clusters'
            marker.id = 22000 + index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = cluster.centroid_world[0]
            marker.pose.position.y = cluster.centroid_world[1]
            marker.pose.position.z = 0.30
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.28
            marker.scale.y = 0.28
            marker.scale.z = 0.12
            marker.color.r = 1.0
            marker.color.g = 0.55
            marker.color.b = 0.0
            marker.color.a = 0.9
            marker_array.markers.append(marker)

            label = Marker()
            label.header.stamp = stamp
            label.header.frame_id = frame_id
            label.ns = 'ralc_deferred_frontier_cluster_labels'
            label.id = 23000 + index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = cluster.centroid_world[0]
            label.pose.position.y = cluster.centroid_world[1]
            label.pose.position.z = 0.58
            label.pose.orientation.w = 1.0
            label.scale.z = 0.16
            label.color.r = 1.0
            label.color.g = 0.65
            label.color.b = 0.1
            label.color.a = 0.95
            label.text = 'DEFERRED_FRONTIER_CLUSTER'
            marker_array.markers.append(label)

    def append_rejected_candidate_markers(self, marker_array, stamp, frame_id):
        for index, candidate in enumerate(self.rejected_goal_candidates):
            visibility_reject = (
                candidate['reason'] == 'OBSERVATION_POSE_NO_VISIBLE_UNKNOWN'
            )
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = frame_id
            marker.ns = 'ralc_rejected_frontier_goal_candidates'
            marker.id = 30000 + index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = candidate['goal'][0]
            marker.pose.position.y = candidate['goal'][1]
            marker.pose.position.z = 0.20
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.12
            marker.scale.y = 0.12
            marker.scale.z = 0.08
            marker.color.r = 1.0 if visibility_reject else 0.55
            marker.color.g = 0.05 if visibility_reject else 0.55
            marker.color.b = 0.0 if visibility_reject else 0.55
            marker.color.a = 0.75
            marker_array.markers.append(marker)

            label = Marker()
            label.header.stamp = stamp
            label.header.frame_id = frame_id
            label.ns = 'ralc_rejected_frontier_goal_candidate_labels'
            label.id = 40000 + index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = candidate['goal'][0]
            label.pose.position.y = candidate['goal'][1]
            label.pose.position.z = 0.42
            label.pose.orientation.w = 1.0
            label.scale.z = 0.16
            label.color.r = 1.0 if visibility_reject else 0.75
            label.color.g = 0.2 if visibility_reject else 0.75
            label.color.b = 0.1 if visibility_reject else 0.75
            label.color.a = 0.9
            label.text = (
                f'OBSERVATION_POSE_REJECTED:{candidate["reason"]}'
            )
            marker_array.markers.append(label)

    def publish_non_actionable_markers(self, grid, clusters):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        frame_id = grid.header.frame_id or self.map_frame

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for cluster in clusters:
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = frame_id
            marker.ns = 'ralc_non_actionable_frontiers'
            marker.id = cluster.cluster_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = cluster.centroid_world[0]
            marker.pose.position.y = cluster.centroid_world[1]
            marker.pose.position.z = 0.24
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.22
            marker.scale.y = 0.22
            marker.scale.z = 0.08
            marker.color.r = 0.55
            marker.color.g = 0.55
            marker.color.b = 0.55
            marker.color.a = 0.85
            marker_array.markers.append(marker)

            cross = Marker()
            cross.header.stamp = stamp
            cross.header.frame_id = frame_id
            cross.ns = 'ralc_non_actionable_frontier_crosses'
            cross.id = 10000 + cluster.cluster_id
            cross.type = Marker.LINE_LIST
            cross.action = Marker.ADD
            cross.pose.orientation.w = 1.0
            cross.scale.x = 0.04
            cross.color.r = 0.85
            cross.color.g = 0.85
            cross.color.b = 0.85
            cross.color.a = 0.95
            cx, cy = cluster.centroid_world
            for x1, y1, x2, y2 in (
                (cx - 0.18, cy - 0.18, cx + 0.18, cy + 0.18),
                (cx - 0.18, cy + 0.18, cx + 0.18, cy - 0.18),
            ):
                p1 = Point()
                p1.x = x1
                p1.y = y1
                p1.z = 0.30
                p2 = Point()
                p2.x = x2
                p2.y = y2
                p2.z = 0.30
                cross.points.extend([p1, p2])
            marker_array.markers.append(cross)

        self.non_actionable_marker_pub.publish(marker_array)

    def publish_failed_goal_markers(self, grid):
        self.prune_expired_failed_goal_blacklist()
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        frame_id = grid.header.frame_id or self.map_frame

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        marker_id = 1
        for record in self.failed_goal_blacklist.values():
            failure_type = str(record.get('failure_type', 'NAV2_FAILED'))
            ineffective = failure_type == 'INEFFECTIVE_OBSERVATION'
            sphere = Marker()
            sphere.header.stamp = stamp
            sphere.header.frame_id = frame_id
            sphere.ns = 'ralc_failed_frontier_goal_markers'
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(record.get('failed_goal_x', 0.0))
            sphere.pose.position.y = float(record.get('failed_goal_y', 0.0))
            sphere.pose.position.z = 0.28
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.28
            sphere.scale.y = 0.28
            sphere.scale.z = 0.12
            sphere.color.r = 0.75 if ineffective else 1.0
            sphere.color.g = 0.25 if ineffective else 0.0
            sphere.color.b = 1.0 if ineffective else 0.0
            sphere.color.a = 0.9
            marker_array.markers.append(sphere)

            cross = Marker()
            cross.header.stamp = stamp
            cross.header.frame_id = frame_id
            cross.ns = 'ralc_failed_frontier_goal_crosses'
            cross.id = marker_id + 10000
            cross.type = Marker.LINE_LIST
            cross.action = Marker.ADD
            cross.pose.orientation.w = 1.0
            cross.scale.x = 0.06
            cross.color.r = 0.75 if ineffective else 1.0
            cross.color.g = 0.25 if ineffective else 0.0
            cross.color.b = 1.0 if ineffective else 0.0
            cross.color.a = 0.95
            cx = float(record.get('failed_goal_x', 0.0))
            cy = float(record.get('failed_goal_y', 0.0))
            for x1, y1, x2, y2 in (
                (cx - 0.22, cy - 0.22, cx + 0.22, cy + 0.22),
                (cx - 0.22, cy + 0.22, cx + 0.22, cy - 0.22),
            ):
                p1 = Point()
                p1.x = x1
                p1.y = y1
                p1.z = 0.36
                p2 = Point()
                p2.x = x2
                p2.y = y2
                p2.z = 0.36
                cross.points.extend([p1, p2])
            marker_array.markers.append(cross)

            label = Marker()
            label.header.stamp = stamp
            label.header.frame_id = frame_id
            label.ns = 'ralc_failed_frontier_goal_labels'
            label.id = marker_id + 20000
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = cx
            label.pose.position.y = cy
            label.pose.position.z = 0.62
            label.pose.orientation.w = 1.0
            label.scale.z = 0.22
            label.color.r = 0.8 if ineffective else 1.0
            label.color.g = 0.35 if ineffective else 0.15
            label.color.b = 1.0 if ineffective else 0.15
            label.color.a = 0.95
            label.text = failure_type
            marker_array.markers.append(label)

            marker_id += 1

        self.failed_goal_marker_pub.publish(marker_array)

    def append_points_marker(self, marker_array, stamp, frame_id, grid, mask, ns, marker_id, color, size):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = size
        marker.scale.y = size
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        ys, xs = np.nonzero(mask)
        origin = grid.info.origin.position
        for x, y in zip(xs, ys):
            point = Point()
            point.x = origin.x + (float(x) + 0.5) * grid.info.resolution
            point.y = origin.y + (float(y) + 0.5) * grid.info.resolution
            point.z = 0.14
            marker.points.append(point)
        marker_array.markers.append(marker)

    def log_debug(self, grid, raw_mask, region_mask, clusters, scores, robot_pose):
        self.get_logger().info(
            '[RALC_DEBUG] frontier_planner: '
            f'raw={int(raw_mask.sum())}, region={int(region_mask.sum())}, '
            f'clusters={len(clusters)}, scored={len(scores)}, '
            f'robot=({robot_pose[0]:.2f},{robot_pose[1]:.2f})'
        )

    def log_unknown_ratio_debug(self, occupancy_stats):
        raw_ratio = float(occupancy_stats.get('raw_unknown_ratio_in_region', 0.0))
        reachable_ratio = float(
            occupancy_stats.get('reachable_unknown_ratio_in_region', 0.0)
        )
        if raw_ratio > self.region_unknown_completion_threshold:
            if reachable_ratio <= self.region_reachable_unknown_completion_threshold:
                self.get_logger().info(
                    '[RALC] Raw unknown ratio is high, but reachable unknown '
                    'ratio is low; allowing region completion if frontier '
                    f'conditions agree. raw={raw_ratio:.4f}, '
                    f'reachable={reachable_ratio:.4f}'
                )
            else:
                self.get_logger().warn(
                    '[RALC] Reachable unknown remains in active region. '
                    f'raw_unknown_ratio={raw_ratio:.4f}, '
                    f'reachable_unknown_ratio={reachable_ratio:.4f}, '
                    f'threshold='
                    f'{self.region_reachable_unknown_completion_threshold:.4f}'
                )


def main(args=None):
    rclpy.init(args=args)
    node = RalcFrontierPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
