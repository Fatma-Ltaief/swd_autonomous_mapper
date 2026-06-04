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
            f'completed_region_margin={self.completed_region_margin:.2f}, '
            f'completed_region_inner_margin={self.completed_region_inner_margin:.2f}'
        )

    def map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg

    def create_next_region_callback(self, _msg: Empty):
        robot_xy = self.lookup_robot_xy()
        if robot_xy is None:
            self.get_logger().warn(
                '[RALC] Cannot create next region yet: robot TF unavailable.'
            )
            return
        seed = self.choose_next_region_seed(robot_xy)
        if seed is None:
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
                self.publish_all_regions_explored(False)
                self.publish_next_region_seed_status(
                    'NO_ACCEPTABLE_SEED',
                    'Global frontiers exist, but no candidate seed passed filtering.',
                )
                self.publish_next_region_candidate_markers()
                self.get_logger().warn(
                    '[RALC] Global frontiers remain, but no acceptable '
                    'next-region seed was found. Not publishing all_regions_explored.'
                )
            return

        self.publish_all_regions_explored(False)
        self.publish_next_region_seed_status(
            'SEED_SELECTED',
            f'Next region seed selected at ({seed[0]:.2f}, {seed[1]:.2f}).',
        )
        self.publish_next_region_candidate_markers(selected_seed=seed)
        seed_msg = PoseStamped()
        seed_msg.header.stamp = self.get_clock().now().to_msg()
        seed_msg.header.frame_id = self.map_frame
        seed_msg.pose.position.x = seed[0]
        seed_msg.pose.position.y = seed[1]
        seed_msg.pose.orientation.w = 1.0
        self.next_region_seed_pub.publish(seed_msg)
        self.get_logger().info(
            f'[RALC] Published next_region_seed=({seed[0]:.2f}, {seed[1]:.2f}); '
            'active region will be created around robot after transition.'
        )

    def create_region_at_robot_callback(self, _msg: Empty):
        robot_xy = self.lookup_robot_xy()
        if robot_xy is None:
            self.get_logger().warn(
                '[RALC] Cannot create region at robot: TF unavailable.'
            )
            return
        self.current_region = self.create_region(robot_xy[0], robot_xy[1])
        self.publish_all_regions_explored(False)
        self.publish_current_region()
        self.publish_region_markers()
        self.get_logger().info(
            f'[RALC] Created ACTIVE region {self.current_region["region_id"]} '
            f'around robot at ({robot_xy[0]:.2f}, {robot_xy[1]:.2f}).'
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
    ) -> Optional[Tuple[float, float]]:
        clusters = self.global_frontier_clusters_outside_completed_regions()
        self.last_global_frontier_cluster_count = len(clusters)
        self.next_region_candidate_debug = []
        if not clusters:
            return None

        candidate_regions = []
        for cluster in clusters:
            for seed in self.seed_candidates_toward_frontier(robot_xy, cluster['centroid']):
                evaluation = self.evaluate_new_region_seed(seed, cluster)
                self.next_region_candidate_debug.append(evaluation)
                self.log_next_region_candidate(evaluation)
                if not evaluation['accepted']:
                    continue
                candidate_regions.append((cluster, seed))
                break

        if not candidate_regions:
            self.get_logger().warn(
                '[RALC] No acceptable next-region seed after completed-region '
                'overlap/margin/safety filtering.'
            )
            return None

        def score(candidate):
            cluster, seed = candidate
            distance = math.hypot(seed[0] - robot_xy[0], seed[1] - robot_xy[1])
            information_gain = float(cluster['size'])
            return distance - 0.04 * information_gain

        best_cluster, best_seed = min(candidate_regions, key=score)
        for evaluation in self.next_region_candidate_debug:
            if math.hypot(
                evaluation['seed'][0] - best_seed[0],
                evaluation['seed'][1] - best_seed[1],
            ) < 1e-6:
                evaluation['selected'] = True
                break
        self.get_logger().info(
            '[RALC] Selected next-region seed after filtering: '
            f'seed=({best_seed[0]:.2f},{best_seed[1]:.2f}), '
            f'frontier_centroid=({best_cluster["centroid"][0]:.2f},'
            f'{best_cluster["centroid"][1]:.2f}), size={best_cluster["size"]}'
        )
        return best_seed

    def seed_candidates_toward_frontier(self, robot_xy, centroid):
        direction_x = centroid[0] - robot_xy[0]
        direction_y = centroid[1] - robot_xy[1]
        norm = math.hypot(direction_x, direction_y)
        if norm < 1e-6:
            return [centroid]
        unit_x = direction_x / norm
        unit_y = direction_y / norm
        candidates = []
        step = max(0.5, self.region_transition_step)
        distance = step
        while distance < norm:
            candidates.append((
                robot_xy[0] + distance * unit_x,
                robot_xy[1] + distance * unit_y,
            ))
            distance += step
        candidates.append(centroid)
        return candidates

    def evaluate_new_region_seed(self, seed, cluster):
        candidate = self.region_rect_from_center(seed[0], seed[1])
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
            seed[0] - cluster['centroid'][0],
            seed[1] - cluster['centroid'][1],
        )

        accepted = True
        reason = 'accepted'
        marker_class = 'accepted'

        if deep_inside_completed:
            accepted = False
            reason = 'inside_completed'
            marker_class = 'inside_completed'
        elif not self.seed_is_known_free(seed[0], seed[1]):
            accepted = False
            reason = 'unsafe_or_unknown'
            marker_class = 'unsafe_or_unknown'
        elif overlap_ratio > self.max_new_region_overlap_ratio:
            accepted = False
            reason = 'overlap_too_high'
            marker_class = 'overlap_too_high'
        elif outside_ratio < self.min_new_region_outside_ratio:
            accepted = False
            reason = 'outside_area_too_small'
            marker_class = 'overlap_too_high'

        return {
            'seed': seed,
            'frontier_centroid': cluster['centroid'],
            'frontier_size': int(cluster['size']),
            'overlap_ratio': overlap_ratio,
            'outside_ratio': outside_ratio,
            'inside_completed': inside_completed,
            'deep_inside_completed': deep_inside_completed,
            'distance_to_completed_boundary': boundary_distance,
            'distance_to_nearest_global_frontier': frontier_distance,
            'accepted': accepted,
            'reason': reason,
            'marker_class': marker_class,
            'selected': False,
        }

    def log_next_region_candidate(self, evaluation):
        seed = evaluation['seed']
        accepted_text = 'accepted' if evaluation['accepted'] else f'rejected:{evaluation["reason"]}'
        self.get_logger().info(
            '[RALC] next-region candidate: '
            f'x={seed[0]:.2f}, y={seed[1]:.2f}, '
            f'overlap_ratio={evaluation["overlap_ratio"]:.2f}, '
            f'outside_ratio={evaluation["outside_ratio"]:.2f}, '
            f'inside_completed={evaluation["inside_completed"]}, '
            f'deep_inside_completed={evaluation["deep_inside_completed"]}, '
            f'distance_to_completed_boundary='
            f'{evaluation["distance_to_completed_boundary"]:.2f}, '
            f'distance_to_nearest_global_frontier='
            f'{evaluation["distance_to_nearest_global_frontier"]:.2f}, '
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

    def region_checkpoint_path(self, region_id):
        return os.path.abspath(os.path.join(
            self.checkpoint_root, f'region_{int(region_id)}'
        ))

    def publish_all_regions_explored(self, explored: bool):
        msg = Bool()
        msg.data = explored
        self.all_regions_explored_pub.publish(msg)

    def publish_next_region_seed_status(self, reason: str, message: str):
        msg = String()
        msg.data = json.dumps({
            'reason': reason,
            'message': message,
            'global_frontier_clusters_outside_completed': (
                self.last_global_frontier_cluster_count
            ),
            'candidate_count': len(self.next_region_candidate_debug),
        })
        self.next_region_seed_status_pub.publish(msg)
        self.get_logger().warn(f'[RALC] next_region_seed_status: {msg.data}')

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
