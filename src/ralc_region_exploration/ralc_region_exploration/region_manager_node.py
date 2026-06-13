import json
import math
import os
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, Empty, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class RegionManager(Node):
    """Owns rectangular R-ALC regions and their lifecycle.

    The paper's region logic is kept here: regions are created at Wmin/Hmin,
    grow around the robot pose until Wmax/Hmax, and are marked REFINEMENT then
    COMPLETED only when the exploration manager requests those transitions.
    Frontier planning never creates, grows, or completes regions.
    """

    def __init__(self):
        super().__init__('ralc_region_manager')

        self.declare_parameter('region_min_width', 4.0)
        self.declare_parameter('region_min_height', 4.0)
        self.declare_parameter('region_max_width', 10.0)
        self.declare_parameter('region_max_height', 10.0)
        self.declare_parameter('robot_neighborhood_radius', 0.8)
        self.declare_parameter('region_transition_step', 1.0)
        self.declare_parameter('max_new_region_overlap_ratio', 0.45)
        self.declare_parameter('min_new_region_outside_ratio', 0.40)
        self.declare_parameter('max_completed_overlap_ratio_for_next_region', 0.55)
        self.declare_parameter('min_outside_completed_ratio_for_next_region', 0.45)
        self.declare_parameter('min_overlap_ratio_for_connectivity', 0.10)
        self.declare_parameter('max_overlap_ratio_for_connectivity', 0.60)
        self.declare_parameter('next_region_outward_shift_ratio', 0.50)
        self.declare_parameter('next_region_seed_search_min_radius', 0.25)
        self.declare_parameter('next_region_seed_search_max_radius', 1.20)
        self.declare_parameter('next_region_seed_search_step', 0.10)
        self.declare_parameter('next_region_seed_occupied_clearance', 0.25)
        self.declare_parameter('max_next_region_frontier_boundary_distance', 2.0)
        self.declare_parameter('next_region_robot_distance_penalty', 0.35)
        self.declare_parameter('next_region_boundary_distance_penalty', 0.50)
        self.declare_parameter('next_region_seed_adjustment_penalty', 0.25)
        self.declare_parameter('completed_region_margin', 0.5)
        self.declare_parameter('completed_region_inner_margin', 0.5)
        self.declare_parameter('checkpoint_root', 'maps/ralc_checkpoints')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('fallback_base_frame', 'base_link')

        self.region_min_width = float(self.get_parameter('region_min_width').value)
        self.region_min_height = float(self.get_parameter('region_min_height').value)
        self.region_max_width = float(self.get_parameter('region_max_width').value)
        self.region_max_height = float(self.get_parameter('region_max_height').value)
        self.robot_neighborhood_radius = float(
            self.get_parameter('robot_neighborhood_radius').value
        )
        self.region_transition_step = float(
            self.get_parameter('region_transition_step').value
        )
        self.max_new_region_overlap_ratio = float(
            self.get_parameter('max_new_region_overlap_ratio').value
        )
        self.min_new_region_outside_ratio = float(
            self.get_parameter('min_new_region_outside_ratio').value
        )
        self.max_completed_overlap_ratio_for_next_region = float(
            self.get_parameter('max_completed_overlap_ratio_for_next_region').value
        )
        self.min_outside_completed_ratio_for_next_region = float(
            self.get_parameter('min_outside_completed_ratio_for_next_region').value
        )
        self.min_overlap_ratio_for_connectivity = float(
            self.get_parameter('min_overlap_ratio_for_connectivity').value
        )
        self.max_overlap_ratio_for_connectivity = float(
            self.get_parameter('max_overlap_ratio_for_connectivity').value
        )
        self.next_region_outward_shift_ratio = float(
            self.get_parameter('next_region_outward_shift_ratio').value
        )
        self.next_region_seed_search_min_radius = float(
            self.get_parameter('next_region_seed_search_min_radius').value
        )
        self.next_region_seed_search_max_radius = float(
            self.get_parameter('next_region_seed_search_max_radius').value
        )
        self.next_region_seed_search_step = float(
            self.get_parameter('next_region_seed_search_step').value
        )
        self.next_region_seed_occupied_clearance = float(
            self.get_parameter('next_region_seed_occupied_clearance').value
        )
        self.max_next_region_frontier_boundary_distance = float(
            self.get_parameter('max_next_region_frontier_boundary_distance').value
        )
        self.next_region_robot_distance_penalty = float(
            self.get_parameter('next_region_robot_distance_penalty').value
        )
        self.next_region_boundary_distance_penalty = float(
            self.get_parameter('next_region_boundary_distance_penalty').value
        )
        self.next_region_seed_adjustment_penalty = float(
            self.get_parameter('next_region_seed_adjustment_penalty').value
        )
        self.completed_region_margin = float(
            self.get_parameter('completed_region_margin').value
        )
        self.completed_region_inner_margin = float(
            self.get_parameter('completed_region_inner_margin').value
        )
        self.checkpoint_root = self.get_parameter('checkpoint_root').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.fallback_base_frame = self.get_parameter('fallback_base_frame').value

        self.latest_map: Optional[OccupancyGrid] = None
        self.current_region = None
        self.completed_regions: List[dict] = []
        self.next_region_candidate_debug = []
        self.last_global_frontier_cluster_count = 0
        self.next_region_id = 1
        self.initial_region_created = False
        self.pending_next_region_center = None
        self.pending_next_region_seed = None
        self._next_region_reachable_mask = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )
        self.create_next_region_sub = self.create_subscription(
            Empty,
            '/ralc/create_next_region',
            self.create_next_region_callback,
            10,
        )
        self.create_region_at_robot_sub = self.create_subscription(
            Empty,
            '/ralc/create_region_at_robot',
            self.create_region_at_robot_callback,
            10,
        )
        self.mark_refinement_sub = self.create_subscription(
            Empty,
            '/ralc/mark_region_refinement',
            self.mark_region_refinement_callback,
            10,
        )
        self.mark_completed_sub = self.create_subscription(
            Empty,
            '/ralc/mark_region_completed',
            self.mark_region_completed_callback,
            10,
        )
        self.expand_region_sub = self.create_subscription(
            Empty,
            '/ralc/expand_current_region',
            self.expand_current_region_callback,
            10,
        )

        self.region_pub = self.create_publisher(String, '/ralc/current_region', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/ralc/region_markers', 10
        )
        self.all_regions_explored_pub = self.create_publisher(
            Bool, '/ralc/all_regions_explored', 10
        )
        self.next_region_seed_pub = self.create_publisher(
            PoseStamped, '/ralc/next_region_seed', 10
        )
        self.next_region_seed_status_pub = self.create_publisher(
            String, '/ralc/next_region_seed_status', 10
        )
        self.next_region_candidate_marker_pub = self.create_publisher(
            MarkerArray, '/ralc/next_region_candidate_markers', 10
        )

        self.timer = self.create_timer(1.0, self.update_region)

        self.get_logger().info(
            '[RALC_DEBUG] REGION PARAMS: '
            f'Wmin/Hmin={self.region_min_width:.2f}/{self.region_min_height:.2f}, '
            f'Wmax/Hmax={self.region_max_width:.2f}/{self.region_max_height:.2f}, '
            f'robot_neighborhood_radius={self.robot_neighborhood_radius:.2f}, '
            f'max_new_region_overlap_ratio={self.max_new_region_overlap_ratio:.2f}, '
            f'min_new_region_outside_ratio={self.min_new_region_outside_ratio:.2f}, '
            'max_completed_overlap_ratio_for_next_region='
            f'{self.max_completed_overlap_ratio_for_next_region:.2f}, '
            'min_outside_completed_ratio_for_next_region='
            f'{self.min_outside_completed_ratio_for_next_region:.2f}, '
            f'min_overlap_ratio_for_connectivity={self.min_overlap_ratio_for_connectivity:.2f}, '
            f'max_overlap_ratio_for_connectivity={self.max_overlap_ratio_for_connectivity:.2f}, '
            f'next_region_outward_shift_ratio={self.next_region_outward_shift_ratio:.2f}, '
            'next_region_seed_search='
            f'{self.next_region_seed_search_min_radius:.2f}->'
            f'{self.next_region_seed_search_max_radius:.2f}/'
            f'{self.next_region_seed_search_step:.2f}, '
            'next_region_seed_occupied_clearance='
            f'{self.next_region_seed_occupied_clearance:.2f}, '
            'max_next_region_frontier_boundary_distance='
            f'{self.max_next_region_frontier_boundary_distance:.2f}, '
            'next_region_distance_penalties='
            f'robot:{self.next_region_robot_distance_penalty:.2f},'
            f'boundary:{self.next_region_boundary_distance_penalty:.2f},'
            f'adjust:{self.next_region_seed_adjustment_penalty:.2f}, '
            f'completed_region_margin={self.completed_region_margin:.2f}, '
            f'completed_region_inner_margin={self.completed_region_inner_margin:.2f}'
        )

    def map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg

    def next_region_seed_yaw(self, seed, selection) -> float:
        cluster = selection.get('cluster') or {}
        frontier = cluster.get('centroid')
        if frontier is not None:
            dx = float(frontier[0]) - float(seed[0])
            dy = float(frontier[1]) - float(seed[1])
            if math.hypot(dx, dy) > 1e-3:
                return math.atan2(dy, dx)
        center = selection.get('region_center')
        if center is not None:
            dx = float(center[0]) - float(seed[0])
            dy = float(center[1]) - float(seed[1])
            if math.hypot(dx, dy) > 1e-3:
                return math.atan2(dy, dx)
        return 0.0

    def quaternion_z_w_from_yaw(self, yaw: float):
        half_yaw = 0.5 * yaw
        return math.sin(half_yaw), math.cos(half_yaw)

    def create_next_region_callback(self, _msg: Empty):
        robot_xy = self.lookup_robot_xy()
        if robot_xy is None:
            self.get_logger().warn(
                '[RALC] Cannot create next region yet: robot TF unavailable.'
            )
            return
        selection = self.choose_next_region_seed(robot_xy)
        seed = selection['seed'] if selection is not None else None
        if seed is None:
            self.pending_next_region_seed = None
            self.pending_next_region_center = None
            if self.last_global_frontier_cluster_count == 0:
                self.current_region = None
                self.publish_all_regions_explored(True)
                self.publish_next_region_seed_status(
                    'ALL_REGIONS_EXPLORED',
                    'No global frontier clusters remain outside completed regions.',
                )
                self.publish_current_region()
                self.publish_region_markers()
                self.get_logger().info(
                    '[RALC] No global frontiers remain outside completed regions.'
                )
            else:
                dominant_reason, dominant_count = self.dominant_candidate_rejection()
                self.publish_all_regions_explored(False)
                self.publish_next_region_seed_status(
                    'NEXT_REGION_BLOCKED',
                    'Global frontiers exist, but no candidate seed passed '
                    f'filtering. dominant_rejection={dominant_reason} '
                    f'({dominant_count} candidates).',
                )
                self.publish_next_region_candidate_markers()
                self.get_logger().warn(
                    '[RALC] Global frontiers remain, but no acceptable '
                    'next-region seed was found. Not publishing all_regions_explored.'
                )
            return

        self.publish_all_regions_explored(False)
        seed_yaw = self.next_region_seed_yaw(seed, selection)
        self.publish_next_region_seed_status(
            'SEED_SELECTED',
            f'Next region seed selected at ({seed[0]:.2f}, {seed[1]:.2f}); '
            f'planned region center=({selection["region_center"][0]:.2f}, '
            f'{selection["region_center"][1]:.2f}), yaw={seed_yaw:.2f} rad.',
        )
        self.publish_next_region_candidate_markers(selected_seed=seed)
        self.pending_next_region_seed = seed
        self.pending_next_region_center = selection['region_center']
        seed_msg = PoseStamped()
        seed_msg.header.stamp = self.get_clock().now().to_msg()
        seed_msg.header.frame_id = self.map_frame
        seed_msg.pose.position.x = seed[0]
        seed_msg.pose.position.y = seed[1]
        qz, qw = self.quaternion_z_w_from_yaw(seed_yaw)
        seed_msg.pose.orientation.z = qz
        seed_msg.pose.orientation.w = qw
        self.next_region_seed_pub.publish(seed_msg)
        self.get_logger().info(
            f'[RALC] Published next_region_seed=({seed[0]:.2f}, {seed[1]:.2f}); '
            f'yaw={seed_yaw:.2f} rad; active region will be created around robot '
            'after transition.'
        )

    def create_region_at_robot_callback(self, _msg: Empty):
        robot_xy = self.lookup_robot_xy()
        if robot_xy is None:
            self.get_logger().warn(
                '[RALC] Cannot create region at robot: TF unavailable.'
            )
            return
        center = self.pending_next_region_center
        if center is None:
            center = robot_xy
        self.current_region = self.create_region(center[0], center[1])
        self.pending_next_region_center = None
        self.pending_next_region_seed = None
        self.publish_all_regions_explored(False)
        self.publish_current_region()
        self.publish_region_markers()
        self.get_logger().info(
            f'[RALC] Created ACTIVE region {self.current_region["region_id"]} '
            f'with center=({center[0]:.2f}, {center[1]:.2f}); '
            f'robot=({robot_xy[0]:.2f}, {robot_xy[1]:.2f}).'
        )

    def mark_region_refinement_callback(self, _msg: Empty):
        if self.current_region is None:
            return
        if self.current_region.get('status') == 'COMPLETED':
            return
        self.current_region['status'] = 'REFINEMENT'
        self.publish_current_region()
        self.publish_region_markers()
        self.get_logger().info(
            f'[RALC] Region {self.current_region["region_id"]} status=REFINEMENT.'
        )

    def mark_region_completed_callback(self, _msg: Empty):
        if self.current_region is None:
            return
        region = dict(self.current_region)
        region['status'] = 'COMPLETED'
        region['completed_time_sec'] = self.now_sec()
        region['checkpoint_path'] = self.region_checkpoint_path(region['region_id'])
        self.completed_regions.append(region)
        self.current_region = None
        self.publish_current_region()
        self.publish_region_markers()
        self.get_logger().info(
            f'[RALC] Region {region["region_id"]} marked COMPLETED; '
            f'checkpoint_path={region["checkpoint_path"]}'
        )

    def expand_current_region_callback(self, _msg: Empty):
        robot_xy = self.lookup_robot_xy()
        if robot_xy is None:
            self.get_logger().warn(
                '[RALC_DEBUG] expand_current_region ignored: robot TF unavailable.'
            )
            return
        if self.current_region is None or self.current_region.get('status') != 'ACTIVE':
            self.get_logger().warn(
                '[RALC_DEBUG] expand_current_region ignored: no ACTIVE region.'
            )
            return

        self.force_expand_active_region(robot_xy[0], robot_xy[1])
        self.publish_current_region(robot_xy, reason='forced_expand_current_region')
        self.publish_region_markers()

    def lookup_robot_xy(self) -> Optional[Tuple[float, float]]:
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

    def update_region(self):
        if self.latest_map is None:
            return
        robot_xy = self.lookup_robot_xy()
        if robot_xy is None:
            return

        if self.current_region is None and not self.initial_region_created:
            self.current_region = self.create_region(robot_xy[0], robot_xy[1])
            self.initial_region_created = True
            self.publish_all_regions_explored(False)
            self.get_logger().info(
                f'[RALC] Created initial ACTIVE region: {self.current_region}'
            )

        if (
            self.current_region is not None and
            self.current_region.get('status') == 'ACTIVE'
        ):
            self.expand_active_region(robot_xy[0], robot_xy[1])

        self.publish_current_region(robot_xy, reason='timer_update')
        self.publish_region_markers()

    def create_region(self, center_x: float, center_y: float):
        half_width = self.region_min_width * 0.5
        half_height = self.region_min_height * 0.5
        region_id = self.next_region_id
        self.next_region_id += 1
        return {
            'region_id': region_id,
            'status': 'ACTIVE',
            'xmin': center_x - half_width,
            'xmax': center_x + half_width,
            'ymin': center_y - half_height,
            'ymax': center_y + half_height,
            'center_x': center_x,
            'center_y': center_y,
            'width': self.region_min_width,
            'height': self.region_min_height,
            'max_width': self.region_max_width,
            'max_height': self.region_max_height,
            'is_at_max_size': False,
            'created_time_sec': self.now_sec(),
            'completed_time_sec': 0.0,
            'checkpoint_path': self.region_checkpoint_path(region_id),
        }

    def expand_active_region(self, robot_x: float, robot_y: float):
        region = self.current_region
        if region is None:
            return

        required = (
            robot_x - self.robot_neighborhood_radius,
            robot_x + self.robot_neighborhood_radius,
            robot_y - self.robot_neighborhood_radius,
            robot_y + self.robot_neighborhood_radius,
        )
        old = (
            float(region['xmin']), float(region['xmax']),
            float(region['ymin']), float(region['ymax']),
        )
        xmin, xmax = self.expand_axis_without_shrinking(
            old[0], old[1], required[0], required[1], self.region_max_width
        )
        ymin, ymax = self.expand_axis_without_shrinking(
            old[2], old[3], required[2], required[3], self.region_max_height
        )

        region['xmin'] = xmin
        region['xmax'] = xmax
        region['ymin'] = ymin
        region['ymax'] = ymax
        region['width'] = xmax - xmin
        region['height'] = ymax - ymin
        region['center_x'] = 0.5 * (xmin + xmax)
        region['center_y'] = 0.5 * (ymin + ymax)
        region['is_at_max_size'] = (
            region['width'] >= self.region_max_width - 1e-6 and
            region['height'] >= self.region_max_height - 1e-6
        )

        new = (xmin, xmax, ymin, ymax)
        if any(abs(new[i] - old[i]) > 1e-3 for i in range(4)):
            self.get_logger().info(
                f'[RALC] Region {region["region_id"]} grew around robot: '
                f'x=[{xmin:.2f},{xmax:.2f}], y=[{ymin:.2f},{ymax:.2f}], '
                f'size=({region["width"]:.2f},{region["height"]:.2f})'
            )

    def force_expand_active_region(self, robot_x: float, robot_y: float):
        """Grow the current ACTIVE region toward Wmax/Hmax without creating a new one."""
        region = self.current_region
        if region is None or region.get('status') != 'ACTIVE':
            return

        old = (
            float(region['xmin']), float(region['xmax']),
            float(region['ymin']), float(region['ymax']),
        )
        new_width = min(self.region_max_width, float(region['width']) + self.region_min_width)
        new_height = min(self.region_max_height, float(region['height']) + self.region_min_height)

        xmin, xmax = self.centered_axis_with_robot_bias(
            old[0], old[1], robot_x, new_width
        )
        ymin, ymax = self.centered_axis_with_robot_bias(
            old[2], old[3], robot_y, new_height
        )
        region['xmin'] = xmin
        region['xmax'] = xmax
        region['ymin'] = ymin
        region['ymax'] = ymax
        region['width'] = xmax - xmin
        region['height'] = ymax - ymin
        region['center_x'] = 0.5 * (xmin + xmax)
        region['center_y'] = 0.5 * (ymin + ymax)
        region['is_at_max_size'] = (
            region['width'] >= self.region_max_width - 1e-6 and
            region['height'] >= self.region_max_height - 1e-6
        )
        self.get_logger().info(
            '[RALC_DEBUG] expand_current_region: '
            f'region={region["region_id"]}, old_bounds='
            f'[{old[0]:.2f},{old[1]:.2f}]x[{old[2]:.2f},{old[3]:.2f}], '
            f'new_bounds=[{xmin:.2f},{xmax:.2f}]x[{ymin:.2f},{ymax:.2f}], '
            f'size=({region["width"]:.2f},{region["height"]:.2f}), '
            f'is_at_max_size={region["is_at_max_size"]}'
        )

    def centered_axis_with_robot_bias(
        self,
        old_lower: float,
        old_upper: float,
        robot_value: float,
        new_size: float,
    ) -> Tuple[float, float]:
        old_center = 0.5 * (old_lower + old_upper)
        lower = old_center - 0.5 * new_size
        upper = old_center + 0.5 * new_size
        required_lower = robot_value - self.robot_neighborhood_radius
        required_upper = robot_value + self.robot_neighborhood_radius
        if required_lower < lower:
            shift = required_lower - lower
            lower += shift
            upper += shift
        if required_upper > upper:
            shift = required_upper - upper
            lower += shift
            upper += shift
        return lower, upper

    def expand_axis_without_shrinking(
        self,
        old_lower: float,
        old_upper: float,
        required_lower: float,
        required_upper: float,
        max_size: float,
    ) -> Tuple[float, float]:
        lower = old_lower
        upper = old_upper
        if required_lower < lower:
            lower = max(required_lower, upper - max_size)
        if required_upper > upper:
            upper = min(required_upper, lower + max_size)
        return lower, upper

    def choose_next_region_seed(
        self,
        robot_xy: Tuple[float, float],
    ) -> Optional[dict]:
        clusters = self.global_frontier_clusters_outside_completed_regions()
        self.last_global_frontier_cluster_count = len(clusters)
        self.next_region_candidate_debug = []
        if not clusters:
            self._next_region_reachable_mask = None
            return None
        self._next_region_reachable_mask = self.reachable_free_mask_from_robot(robot_xy)

        candidate_regions = []
        for cluster in clusters:
            for seed in self.seed_candidates_for_frontier(cluster, robot_xy):
                evaluation = self.evaluate_new_region_seed(seed, cluster, robot_xy)
                self.next_region_candidate_debug.append(evaluation)
                self.log_next_region_candidate(evaluation)
                if not evaluation['accepted']:
                    continue
                candidate_regions.append((cluster, seed, evaluation))
                break

        if not candidate_regions:
            self._next_region_reachable_mask = None
            self.get_logger().warn(
                '[RALC] No acceptable next-region seed after completed-region '
                'overlap/margin/safety filtering.'
            )
            return None

        def score(candidate):
            cluster, seed, evaluation = candidate
            return float(evaluation.get('score', 0.0))

        for _cluster, _seed, evaluation in candidate_regions:
            evaluation['score'] = score((_cluster, _seed, evaluation))

        best_cluster, best_seed, _best_eval = max(candidate_regions, key=score)
        for evaluation in self.next_region_candidate_debug:
            if math.hypot(
                evaluation['seed'][0] - best_seed[0],
                evaluation['seed'][1] - best_seed[1],
            ) < 1e-6:
                evaluation['selected'] = True
                break
        self._next_region_reachable_mask = None
        self.get_logger().info(
            '[RALC] Selected next-region seed after filtering: '
            f'seed=({best_seed[0]:.2f},{best_seed[1]:.2f}), '
            f'frontier_centroid=({best_cluster["centroid"][0]:.2f},'
            f'{best_cluster["centroid"][1]:.2f}), '
            f'region_center=({_best_eval["region_center_x"]:.2f},'
            f'{_best_eval["region_center_y"]:.2f}), '
            f'overlap={_best_eval["completed_overlap_ratio"]:.2f}, '
            f'outside={_best_eval["outside_completed_ratio"]:.2f}, '
            f'score={_best_eval["score"]:.3f}, size={best_cluster["size"]}'
        )
        return {
            'seed': best_seed,
            'region_center': (
                float(_best_eval['region_center_x']),
                float(_best_eval['region_center_y']),
            ),
            'cluster': best_cluster,
        }

    def seed_candidates_for_frontier(self, cluster, robot_xy):
        """Find safe navigation seeds near a frontier centroid.

        The frontier centroid is the information target and may be unknown-adjacent.
        The returned seed is a robot navigation pose, so it must be known free,
        clear of occupied cells, and connected through known free space.
        """
        centroid = cluster['centroid']
        min_radius = max(0.0, self.next_region_seed_search_min_radius)
        max_radius = max(min_radius, self.next_region_seed_search_max_radius)
        step = max(0.05, self.next_region_seed_search_step)
        angle_count = 32

        radius = min_radius
        while radius <= max_radius + 1e-6:
            ring_candidates = []
            for idx in range(angle_count):
                angle = (2.0 * math.pi * float(idx)) / float(angle_count)
                candidate = (
                    centroid[0] + radius * math.cos(angle),
                    centroid[1] + radius * math.sin(angle),
                )
                if not self.seed_is_frontier_navigation_safe(candidate[0], candidate[1]):
                    continue
                if self.point_deep_inside_completed_region(
                    candidate[0],
                    candidate[1],
                    self.completed_region_inner_margin,
                ):
                    continue
                if not self.seed_reachable_from_robot(candidate, robot_xy):
                    continue
                ring_candidates.append(candidate)

            ring_candidates = self.unique_seed_candidates(ring_candidates)
            if ring_candidates:
                ring_candidates.sort(
                    key=lambda seed: (
                        self.distance_to_completed_boundary(seed[0], seed[1]),
                        math.hypot(seed[0] - centroid[0], seed[1] - centroid[1]),
                    )
                )
                return ring_candidates
            radius += step

        return self.unique_seed_candidates([centroid])

    def outward_region_center_from_frontier(self, frontier):
        completed = self.nearest_completed_region_to_point(frontier[0], frontier[1])
        if completed is None:
            return frontier
        dx = frontier[0] - float(completed['center_x'])
        dy = frontier[1] - float(completed['center_y'])
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return frontier
        outward_x = dx / norm
        outward_y = dy / norm
        return (
            frontier[0] + outward_x * self.region_min_width * self.next_region_outward_shift_ratio,
            frontier[1] + outward_y * self.region_min_height * self.next_region_outward_shift_ratio,
        )

    def adjust_region_center_to_include_points(self, center, points, margin=0.0):
        center_x = float(center[0])
        center_y = float(center[1])
        half_width = self.region_min_width * 0.5
        half_height = self.region_min_height * 0.5
        for point in points:
            px = float(point[0])
            py = float(point[1])
            xmin = center_x - half_width
            xmax = center_x + half_width
            ymin = center_y - half_height
            ymax = center_y + half_height
            if px < xmin + margin:
                center_x += px - (xmin + margin)
            elif px > xmax - margin:
                center_x += px - (xmax - margin)
            if py < ymin + margin:
                center_y += py - (ymin + margin)
            elif py > ymax - margin:
                center_y += py - (ymax - margin)
        return center_x, center_y

    def nearest_completed_region_to_point(self, x, y):
        best = None
        best_distance = None
        for region in self.completed_regions:
            distance = math.hypot(
                x - float(region['center_x']),
                y - float(region['center_y']),
            )
            if best_distance is None or distance < best_distance:
                best = region
                best_distance = distance
        return best

    def unique_seed_candidates(self, candidates):
        unique = []
        for candidate in candidates:
            if any(
                math.hypot(candidate[0] - existing[0], candidate[1] - existing[1]) < 0.05
                for existing in unique
            ):
                continue
            unique.append(candidate)
        return unique

    def evaluate_new_region_seed(self, seed, cluster, robot_xy):
        frontier = cluster['centroid']
        region_center = self.outward_region_center_from_frontier(frontier)
        region_center = self.adjust_region_center_to_include_points(
            region_center,
            (frontier, seed),
            margin=0.05,
        )
        candidate = self.region_rect_from_center(region_center[0], region_center[1])
        overlap_ratio = self.completed_overlap_ratio(candidate)
        outside_ratio = max(0.0, 1.0 - overlap_ratio)
        inside_completed = self.point_in_completed_region(seed[0], seed[1], 0.0)
        deep_inside_completed = self.point_deep_inside_completed_region(
            seed[0],
            seed[1],
            self.completed_region_inner_margin,
        )
        boundary_distance = self.distance_to_completed_boundary(seed[0], seed[1])
        frontier_distance = math.hypot(
            seed[0] - frontier[0],
            seed[1] - frontier[1],
        )
        frontier_boundary_distance = self.distance_to_completed_boundary(
            frontier[0],
            frontier[1],
        )
        robot_seed_distance = math.hypot(seed[0] - robot_xy[0], seed[1] - robot_xy[1])
        reachable_free_ratio, unknown_ratio = self.region_map_ratios(candidate)
        close_to_boundary = boundary_distance <= max(
            self.region_transition_step,
            self.completed_region_margin,
        )
        frontier_close_to_completed_boundary = (
            frontier_boundary_distance <= self.max_next_region_frontier_boundary_distance
        )
        seed_reachable = self.seed_reachable_from_robot(seed, robot_xy)
        has_connectivity_overlap = (
            self.min_overlap_ratio_for_connectivity <= overlap_ratio <=
            self.max_overlap_ratio_for_connectivity
        )
        connected_or_adjacent = seed_reachable
        scoring_boundary_distance = (
            0.0 if math.isinf(frontier_boundary_distance)
            else frontier_boundary_distance
        )
        seed_known_free = self.seed_is_frontier_navigation_safe(seed[0], seed[1])
        region_contains_seed = self.rect_contains_point(candidate, seed[0], seed[1])
        region_contains_frontier = self.rect_contains_point(
            candidate,
            frontier[0],
            frontier[1],
        )
        seed_cell_value = self.map_value_at_world(seed[0], seed[1])
        score = (
            2.0 * outside_ratio +
            1.0 * unknown_ratio +
            1.0 * reachable_free_ratio -
            2.0 * abs(overlap_ratio - 0.30) -
            self.next_region_robot_distance_penalty * robot_seed_distance -
            self.next_region_boundary_distance_penalty * scoring_boundary_distance -
            self.next_region_seed_adjustment_penalty * frontier_distance
        )

        accepted = True
        reason = 'accepted'
        marker_class = 'accepted'

        if (
            overlap_ratio > self.max_completed_overlap_ratio_for_next_region or
            outside_ratio < self.min_outside_completed_ratio_for_next_region
        ):
            accepted = False
            reason = 'completed_overlap_too_high'
            marker_class = 'inside_completed'
        elif deep_inside_completed:
            accepted = False
            reason = 'frontier_seed_inside_completed_region'
            marker_class = 'inside_completed'
        elif not seed_known_free:
            accepted = False
            reason = 'unsafe_or_unknown'
            marker_class = 'unsafe_or_unknown'
        elif not region_contains_seed:
            accepted = False
            reason = 'new_region_does_not_contain_navigation_seed'
            marker_class = 'unsafe_or_unknown'
        elif not region_contains_frontier:
            accepted = False
            reason = 'new_region_does_not_contain_frontier'
            marker_class = 'unsafe_or_unknown'
        elif not connected_or_adjacent:
            accepted = False
            reason = 'not_connected_to_reachable_free_or_completed_boundary'
            marker_class = 'unsafe_or_unknown'

        return {
            'seed': seed,
            'frontier_x': float(frontier[0]),
            'frontier_y': float(frontier[1]),
            'navigation_seed_x': float(seed[0]),
            'navigation_seed_y': float(seed[1]),
            'shifted_seed_x': float(seed[0]),
            'shifted_seed_y': float(seed[1]),
            'region_center_x': float(region_center[0]),
            'region_center_y': float(region_center[1]),
            'frontier_centroid': frontier,
            'frontier_size': int(cluster['size']),
            'overlap_ratio': overlap_ratio,
            'outside_ratio': outside_ratio,
            'completed_overlap_ratio': overlap_ratio,
            'outside_completed_ratio': outside_ratio,
            'reachable_free_ratio': reachable_free_ratio,
            'unknown_ratio': unknown_ratio,
            'inside_completed': inside_completed,
            'deep_inside_completed': deep_inside_completed,
            'seed_known_free': seed_known_free,
            'seed_cell_value': seed_cell_value,
            'seed_reachable_from_robot': seed_reachable,
            'region_contains_seed': region_contains_seed,
            'region_contains_frontier': region_contains_frontier,
            'frontier_close_to_completed_boundary': frontier_close_to_completed_boundary,
            'has_connectivity_overlap': has_connectivity_overlap,
            'close_to_boundary': close_to_boundary,
            'connected_or_adjacent': connected_or_adjacent,
            'robot_seed_distance': robot_seed_distance,
            'distance_to_completed_boundary': boundary_distance,
            'frontier_distance_to_completed_boundary': frontier_boundary_distance,
            'distance_to_nearest_global_frontier': frontier_distance,
            'seed_adjustment_distance': frontier_distance,
            'score': score,
            'accepted': accepted,
            'reason': reason,
            'rejection_reason': None if accepted else reason,
            'marker_class': marker_class,
            'selected': False,
        }

    def log_next_region_candidate(self, evaluation):
        seed = evaluation['seed']
        accepted_text = 'accepted' if evaluation['accepted'] else f'rejected:{evaluation["reason"]}'
        self.get_logger().info(
            '[RALC] next-region candidate: '
            f'seed=({seed[0]:.2f}, {seed[1]:.2f}), '
            f'frontier=({evaluation["frontier_x"]:.2f},'
            f'{evaluation["frontier_y"]:.2f}), '
            f'planned_region_center=({evaluation["region_center_x"]:.2f},'
            f'{evaluation["region_center_y"]:.2f}), '
            f'completed_overlap_ratio='
            f'{evaluation["completed_overlap_ratio"]:.2f}, '
            f'outside_completed_ratio='
            f'{evaluation["outside_completed_ratio"]:.2f}, '
            f'reachable_free_ratio={evaluation["reachable_free_ratio"]:.3f}, '
            f'unknown_ratio={evaluation["unknown_ratio"]:.3f}, '
            f'inside_completed={evaluation["inside_completed"]}, '
            f'deep_inside_completed={evaluation["deep_inside_completed"]}, '
            f'seed_known_free={evaluation["seed_known_free"]}, '
            f'seed_cell_value={evaluation["seed_cell_value"]}, '
            f'seed_reachable_from_robot='
            f'{evaluation["seed_reachable_from_robot"]}, '
            f'region_contains_seed={evaluation["region_contains_seed"]}, '
            f'region_contains_frontier={evaluation["region_contains_frontier"]}, '
            f'frontier_close_to_completed_boundary='
            f'{evaluation["frontier_close_to_completed_boundary"]}, '
            f'has_connectivity_overlap={evaluation["has_connectivity_overlap"]}, '
            f'close_to_boundary={evaluation["close_to_boundary"]}, '
            f'connected_or_adjacent={evaluation["connected_or_adjacent"]}, '
            f'robot_seed_distance={evaluation["robot_seed_distance"]:.2f}, '
            f'distance_to_completed_boundary='
            f'{evaluation["distance_to_completed_boundary"]:.2f}, '
            f'frontier_distance_to_completed_boundary='
            f'{evaluation["frontier_distance_to_completed_boundary"]:.2f}, '
            f'distance_to_nearest_global_frontier='
            f'{evaluation["distance_to_nearest_global_frontier"]:.2f}, '
            f'seed_adjustment_distance='
            f'{evaluation["seed_adjustment_distance"]:.2f}, '
            f'score={evaluation["score"]:.3f}, '
            f'{accepted_text}'
        )

    def global_frontier_clusters_outside_completed_regions(self) -> List[dict]:
        if self.latest_map is None:
            return []
        grid = self.latest_map
        data = np.array(grid.data, dtype=np.int16).reshape(
            (grid.info.height, grid.info.width)
        )
        free = data == 0
        unknown = data == -1
        frontier = self.detect_frontier_mask(free, unknown)
        visited = np.zeros_like(frontier, dtype=bool)
        clusters = []
        cluster_id = 1
        for y in range(frontier.shape[0]):
            for x in range(frontier.shape[1]):
                if visited[y, x] or not frontier[y, x]:
                    continue
                cells = self.collect_cluster(x, y, frontier, visited)
                if len(cells) < 3:
                    continue
                arr = np.array(cells, dtype=np.float64)
                centroid_cell = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
                centroid_world = self.map_cell_to_world(
                    grid, centroid_cell[0], centroid_cell[1]
                )
                if self.point_deep_inside_completed_region(
                    centroid_world[0],
                    centroid_world[1],
                    self.completed_region_inner_margin,
                ):
                    continue
                clusters.append({
                    'id': cluster_id,
                    'size': len(cells),
                    'centroid': centroid_world,
                })
                cluster_id += 1
        return clusters

    def detect_frontier_mask(self, free: np.ndarray, unknown: np.ndarray):
        unknown_neighbor = np.zeros_like(unknown, dtype=bool)
        unknown_neighbor[1:, :] |= unknown[:-1, :]
        unknown_neighbor[:-1, :] |= unknown[1:, :]
        unknown_neighbor[:, 1:] |= unknown[:, :-1]
        unknown_neighbor[:, :-1] |= unknown[:, 1:]
        return free & unknown_neighbor

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

    def map_cell_to_world(self, grid, cell_x, cell_y):
        origin = grid.info.origin.position
        resolution = grid.info.resolution
        return (
            float(origin.x + (cell_x + 0.5) * resolution),
            float(origin.y + (cell_y + 0.5) * resolution),
        )

    def world_to_map_cell(self, x, y):
        if self.latest_map is None:
            return None
        grid = self.latest_map
        origin = grid.info.origin.position
        cell_x = int((x - origin.x) / grid.info.resolution)
        cell_y = int((y - origin.y) / grid.info.resolution)
        if (
            cell_x < 0 or cell_x >= grid.info.width or
            cell_y < 0 or cell_y >= grid.info.height
        ):
            return None
        return cell_x, cell_y

    def map_value_at_world(self, x, y):
        cell = self.world_to_map_cell(x, y)
        if cell is None or self.latest_map is None:
            return None
        data = np.array(self.latest_map.data, dtype=np.int16).reshape(
            (self.latest_map.info.height, self.latest_map.info.width)
        )
        return int(data[cell[1], cell[0]])

    def point_in_completed_region(self, x: float, y: float, margin: float = 0.0) -> bool:
        for region in self.completed_regions:
            if (
                float(region['xmin']) - margin <= x <= float(region['xmax']) + margin and
                float(region['ymin']) - margin <= y <= float(region['ymax']) + margin
            ):
                return True
        return False

    def point_deep_inside_completed_region(self, x: float, y: float, inner_margin: float) -> bool:
        for region in self.completed_regions:
            xmin = float(region['xmin']) + inner_margin
            xmax = float(region['xmax']) - inner_margin
            ymin = float(region['ymin']) + inner_margin
            ymax = float(region['ymax']) - inner_margin
            if xmin > xmax or ymin > ymax:
                continue
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True
        return False

    def distance_to_completed_boundary(self, x: float, y: float) -> float:
        if not self.completed_regions:
            return float('inf')
        best = float('inf')
        for region in self.completed_regions:
            xmin = float(region['xmin'])
            xmax = float(region['xmax'])
            ymin = float(region['ymin'])
            ymax = float(region['ymax'])
            inside_x = xmin <= x <= xmax
            inside_y = ymin <= y <= ymax
            if inside_x and inside_y:
                distance = min(x - xmin, xmax - x, y - ymin, ymax - y)
            else:
                dx = max(xmin - x, 0.0, x - xmax)
                dy = max(ymin - y, 0.0, y - ymax)
                distance = math.hypot(dx, dy)
            best = min(best, distance)
        return best

    def region_rect_from_center(self, center_x, center_y):
        half_width = self.region_min_width * 0.5
        half_height = self.region_min_height * 0.5
        return {
            'xmin': center_x - half_width,
            'xmax': center_x + half_width,
            'ymin': center_y - half_height,
            'ymax': center_y + half_height,
        }

    def rect_contains_point(self, rect, x, y):
        return (
            float(rect['xmin']) <= x <= float(rect['xmax']) and
            float(rect['ymin']) <= y <= float(rect['ymax'])
        )

    def rect_overlap_ratio(self, candidate, completed):
        x_overlap = max(
            0.0,
            min(float(candidate['xmax']), float(completed['xmax'])) -
            max(float(candidate['xmin']), float(completed['xmin'])),
        )
        y_overlap = max(
            0.0,
            min(float(candidate['ymax']), float(completed['ymax'])) -
            max(float(candidate['ymin']), float(completed['ymin'])),
        )
        candidate_area = max(
            1e-6,
            (float(candidate['xmax']) - float(candidate['xmin'])) *
            (float(candidate['ymax']) - float(candidate['ymin'])),
        )
        return (x_overlap * y_overlap) / candidate_area

    def completed_overlap_ratio(self, candidate):
        candidate_area = max(
            1e-6,
            (float(candidate['xmax']) - float(candidate['xmin'])) *
            (float(candidate['ymax']) - float(candidate['ymin'])),
        )
        overlap_area = 0.0
        for region in self.completed_regions:
            x_overlap = max(
                0.0,
                min(float(candidate['xmax']), float(region['xmax'])) -
                max(float(candidate['xmin']), float(region['xmin'])),
            )
            y_overlap = max(
                0.0,
                min(float(candidate['ymax']), float(region['ymax'])) -
                max(float(candidate['ymin']), float(region['ymin'])),
            )
            overlap_area += x_overlap * y_overlap
        return min(1.0, overlap_area / candidate_area)

    def region_map_ratios(self, candidate):
        if self.latest_map is None:
            return 0.0, 0.0
        grid = self.latest_map
        origin = grid.info.origin.position
        resolution = grid.info.resolution
        xmin = max(0, int(math.floor((float(candidate['xmin']) - origin.x) / resolution)))
        xmax = min(
            grid.info.width,
            int(math.ceil((float(candidate['xmax']) - origin.x) / resolution)),
        )
        ymin = max(0, int(math.floor((float(candidate['ymin']) - origin.y) / resolution)))
        ymax = min(
            grid.info.height,
            int(math.ceil((float(candidate['ymax']) - origin.y) / resolution)),
        )
        if xmin >= xmax or ymin >= ymax:
            return 0.0, 0.0
        data = np.array(grid.data, dtype=np.int16).reshape(
            (grid.info.height, grid.info.width)
        )
        region = data[ymin:ymax, xmin:xmax]
        total = max(1, int(region.size))
        free = int(np.count_nonzero(region == 0))
        unknown = int(np.count_nonzero(region == -1))
        return float(free / total), float(unknown / total)

    def seed_is_known_free(self, x, y):
        if self.latest_map is None:
            return False
        grid = self.latest_map
        origin = grid.info.origin.position
        cell_x = int((x - origin.x) / grid.info.resolution)
        cell_y = int((y - origin.y) / grid.info.resolution)
        if (
            cell_x < 0 or cell_x >= grid.info.width or
            cell_y < 0 or cell_y >= grid.info.height
        ):
            return False
        data = np.array(grid.data, dtype=np.int16).reshape(
            (grid.info.height, grid.info.width)
        )
        radius_cells = max(1, int(math.ceil(0.25 / grid.info.resolution)))
        for yy in range(max(0, cell_y - radius_cells), min(grid.info.height, cell_y + radius_cells + 1)):
            for xx in range(max(0, cell_x - radius_cells), min(grid.info.width, cell_x + radius_cells + 1)):
                if data[yy, xx] != 0:
                    return False
        return True

    def seed_is_frontier_navigation_safe(self, x, y):
        if self.latest_map is None:
            return False
        grid = self.latest_map
        cell = self.world_to_map_cell(x, y)
        if cell is None:
            return False
        cell_x, cell_y = cell
        data = np.array(grid.data, dtype=np.int16).reshape(
            (grid.info.height, grid.info.width)
        )
        if data[cell_y, cell_x] != 0:
            return False
        radius_cells = max(
            1,
            int(math.ceil(
                self.next_region_seed_occupied_clearance / grid.info.resolution
            )),
        )
        for yy in range(max(0, cell_y - radius_cells), min(grid.info.height, cell_y + radius_cells + 1)):
            for xx in range(max(0, cell_x - radius_cells), min(grid.info.width, cell_x + radius_cells + 1)):
                if data[yy, xx] > 50:
                    return False
        return True

    def seed_reachable_from_robot(self, seed, robot_xy):
        if self.latest_map is None:
            return False
        goal = self.world_to_map_cell(seed[0], seed[1])
        if goal is None:
            return False
        if self._next_region_reachable_mask is not None:
            return bool(self._next_region_reachable_mask[goal[1], goal[0]])
        mask = self.reachable_free_mask_from_robot(robot_xy)
        if mask is None:
            return False
        return bool(mask[goal[1], goal[0]])

    def reachable_free_mask_from_robot(self, robot_xy):
        if self.latest_map is None:
            return None
        start = self.world_to_map_cell(robot_xy[0], robot_xy[1])
        if start is None:
            return None
        grid = self.latest_map
        data = np.array(grid.data, dtype=np.int16).reshape(
            (grid.info.height, grid.info.width)
        )
        start = self.nearest_free_cell(start[0], start[1], data)
        if start is None:
            return None

        queue = deque([start])
        visited = np.zeros(data.shape, dtype=bool)
        visited[start[1], start[0]] = True
        while queue:
            x, y = queue.popleft()
            for ny in range(y - 1, y + 2):
                for nx in range(x - 1, x + 2):
                    if nx == x and ny == y:
                        continue
                    if nx < 0 or ny < 0 or nx >= grid.info.width or ny >= grid.info.height:
                        continue
                    if visited[ny, nx] or data[ny, nx] != 0:
                        continue
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        return visited

    def nearest_free_cell(self, cell_x, cell_y, data):
        height, width = data.shape
        if 0 <= cell_x < width and 0 <= cell_y < height and data[cell_y, cell_x] == 0:
            return cell_x, cell_y
        max_radius = max(1, int(math.ceil(0.5 / self.latest_map.info.resolution)))
        for radius in range(1, max_radius + 1):
            for yy in range(max(0, cell_y - radius), min(height, cell_y + radius + 1)):
                for xx in range(max(0, cell_x - radius), min(width, cell_x + radius + 1)):
                    if data[yy, xx] == 0:
                        return xx, yy
        return None

    def region_checkpoint_path(self, region_id):
        return os.path.abspath(os.path.join(
            self.checkpoint_root, f'region_{int(region_id)}'
        ))

    def publish_all_regions_explored(self, explored: bool):
        msg = Bool()
        msg.data = explored
        self.all_regions_explored_pub.publish(msg)

    def publish_next_region_seed_status(self, reason: str, message: str):
        rejection_counts = self.next_region_rejection_counts()
        dominant_reason, dominant_count = self.dominant_candidate_rejection()
        msg = String()
        msg.data = json.dumps({
            'reason': reason,
            'message': message,
            'global_frontier_clusters_outside_completed': (
                self.last_global_frontier_cluster_count
            ),
            'candidate_count': len(self.next_region_candidate_debug),
            'candidate_rejection_counts': rejection_counts,
            'dominant_rejection_reason': dominant_reason,
            'dominant_rejection_count': dominant_count,
        })
        self.next_region_seed_status_pub.publish(msg)
        self.get_logger().warn(f'[RALC] next_region_seed_status: {msg.data}')

    def next_region_rejection_counts(self):
        counts = {}
        for evaluation in self.next_region_candidate_debug:
            if evaluation.get('accepted'):
                continue
            reason = str(evaluation.get('reason', 'unknown'))
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def dominant_candidate_rejection(self):
        counts = self.next_region_rejection_counts()
        if not counts:
            return 'none', 0
        reason, count = max(counts.items(), key=lambda item: item[1])
        return reason, int(count)

    def publish_next_region_candidate_markers(self, selected_seed=None):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for idx, evaluation in enumerate(self.next_region_candidate_debug):
            marker_class = evaluation['marker_class']
            if selected_seed is not None and math.hypot(
                evaluation['seed'][0] - selected_seed[0],
                evaluation['seed'][1] - selected_seed[1],
            ) < 1e-6:
                marker_class = 'selected'

            color = self.next_region_candidate_color(marker_class)
            seed = evaluation['seed']

            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.map_frame
            marker.ns = 'ralc_next_region_candidate_seeds'
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = seed[0]
            marker.pose.position.y = seed[1]
            marker.pose.position.z = 0.35
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.28 if marker_class == 'selected' else 0.18
            marker.scale.y = 0.28 if marker_class == 'selected' else 0.18
            marker.scale.z = 0.16
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
            marker_array.markers.append(marker)

            label = Marker()
            label.header.stamp = stamp
            label.header.frame_id = self.map_frame
            label.ns = 'ralc_next_region_candidate_labels'
            label.id = 10000 + idx
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = seed[0]
            label.pose.position.y = seed[1]
            label.pose.position.z = 0.62
            label.pose.orientation.w = 1.0
            label.scale.z = 0.16
            label.color.r, label.color.g, label.color.b, label.color.a = color
            label.text = (
                'selected' if marker_class == 'selected'
                else evaluation['reason']
            )
            marker_array.markers.append(label)

        self.next_region_candidate_marker_pub.publish(marker_array)

    def next_region_candidate_color(self, marker_class):
        if marker_class == 'selected':
            return (1.0, 0.9, 0.05, 0.95)
        if marker_class == 'accepted':
            return (0.0, 0.9, 0.1, 0.85)
        if marker_class == 'inside_completed':
            return (1.0, 0.0, 0.0, 0.85)
        if marker_class == 'overlap_too_high':
            return (1.0, 0.45, 0.0, 0.85)
        return (0.55, 0.55, 0.55, 0.85)

    def publish_current_region(self, robot_xy=None, reason='publish_current_region'):
        msg = String()
        msg.data = json.dumps(self.current_region or {})
        self.region_pub.publish(msg)
        if self.current_region is None:
            return
        region = self.current_region
        robot_text = 'unavailable'
        if robot_xy is not None:
            robot_text = f'({robot_xy[0]:.2f},{robot_xy[1]:.2f})'
        self.get_logger().info(
            '[RALC_DEBUG] publish_current_region: '
            f'reason={reason}, region_id={region["region_id"]}, '
            f'status={region["status"]}, '
            f'width={float(region["width"]):.2f}, '
            f'height={float(region["height"]):.2f}, '
            f'max_width={float(region["max_width"]):.2f}, '
            f'max_height={float(region["max_height"]):.2f}, '
            f'is_at_max_size={bool(region["is_at_max_size"])}, '
            f'center=({float(region["center_x"]):.2f},'
            f'{float(region["center_y"]):.2f}), robot={robot_text}'
        )

    def publish_region_markers(self):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)
        for region in self.completed_regions:
            self.append_region_markers(marker_array, stamp, region)
        if self.current_region is not None:
            self.append_region_markers(marker_array, stamp, self.current_region)
        self.marker_pub.publish(marker_array)

    def append_region_markers(self, marker_array, stamp, region):
        status = region.get('status', 'ACTIVE')
        xmin = float(region['xmin'])
        xmax = float(region['xmax'])
        ymin = float(region['ymin'])
        ymax = float(region['ymax'])
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        region_id = int(region['region_id'])

        if status == 'ACTIVE':
            fill_color = (0.1, 0.8, 0.2, 0.20)
            line_color = (0.1, 1.0, 0.2, 0.95)
        elif status == 'REFINEMENT':
            fill_color = (1.0, 0.55, 0.05, 0.18)
            line_color = (1.0, 0.55, 0.05, 0.95)
        else:
            fill_color = (0.2, 0.35, 0.2, 0.16)
            line_color = (0.45, 0.65, 0.45, 0.75)

        box = Marker()
        box.header.stamp = stamp
        box.header.frame_id = self.map_frame
        box.ns = 'ralc_region_window_markers'
        box.id = region_id
        box.type = Marker.CUBE
        box.action = Marker.ADD
        box.pose.position.x = center_x
        box.pose.position.y = center_y
        box.pose.position.z = 0.03
        box.pose.orientation.w = 1.0
        box.scale.x = xmax - xmin
        box.scale.y = ymax - ymin
        box.scale.z = 0.04
        box.color.r, box.color.g, box.color.b, box.color.a = fill_color
        marker_array.markers.append(box)

        outline = Marker()
        outline.header.stamp = stamp
        outline.header.frame_id = self.map_frame
        outline.ns = 'ralc_region_window_outlines'
        outline.id = region_id
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.pose.orientation.w = 1.0
        outline.scale.x = 0.06
        outline.color.r, outline.color.g, outline.color.b, outline.color.a = line_color
        for x, y in (
            (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin),
        ):
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.08
            outline.points.append(point)
        marker_array.markers.append(outline)

        label = Marker()
        label.header.stamp = stamp
        label.header.frame_id = self.map_frame
        label.ns = 'ralc_region_window_labels'
        label.id = region_id
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = center_x
        label.pose.position.y = center_y
        label.pose.position.z = 0.45
        label.pose.orientation.w = 1.0
        label.scale.z = 0.23
        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.color.a = 1.0
        label.text = f"Region {region_id}\n{status}"
        marker_array.markers.append(label)

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = RegionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
