import json
import math
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class RalcPgsPlanner(Node):
    """Pose Graph Stabilizing planner using SLAM backend keyframes.

    R-ALC regional refinement stabilizes the pose graph by computing the
    convex hull of keyframes inside the current region and traversing the hull
    in both directions. A trajectory proxy remains available only as an
    explicit fallback for backend-missing development runs.
    """

    def __init__(self):
        super().__init__('ralc_pgs_planner')

        self.declare_parameter('use_trajectory_proxy_if_backend_missing', False)
        self.declare_parameter('min_pgs_regional_keyframes', 3)
        self.declare_parameter('pgs_keyframe_dedup_distance', 0.10)
        self.declare_parameter('close_hull_loop', True)
        self.declare_parameter('trajectory_proxy_min_spacing', 0.75)
        self.declare_parameter('pgs_use_region_owned_keyframes', True)
        self.declare_parameter('pgs_filter_keyframes_by_region_entry', True)
        self.declare_parameter('pgs_region_entry_anchor_node_count', 0)
        self.declare_parameter('pgs_min_waypoint_spacing', 0.45)
        self.declare_parameter('pgs_use_costmap_waypoint_filter', True)
        self.declare_parameter('pgs_costmap_topic', '/global_costmap/costmap')
        self.declare_parameter('pgs_waypoint_cost_threshold', 35)
        self.declare_parameter('pgs_waypoint_clearance', 0.25)
        self.declare_parameter('pgs_waypoint_snap_radius', 0.80)
        self.declare_parameter('pgs_min_safe_waypoints', 2)
        self.declare_parameter('pgs_require_costmap_path_connectivity', False)
        self.declare_parameter('pgs_max_waypoint_path_length', 8.0)

        self.use_trajectory_proxy_if_backend_missing = bool(
            self.get_parameter('use_trajectory_proxy_if_backend_missing').value
        )
        self.min_pgs_regional_keyframes = int(
            self.get_parameter('min_pgs_regional_keyframes').value
        )
        self.pgs_keyframe_dedup_distance = float(
            self.get_parameter('pgs_keyframe_dedup_distance').value
        )
        self.close_hull_loop = bool(self.get_parameter('close_hull_loop').value)
        self.trajectory_proxy_min_spacing = float(
            self.get_parameter('trajectory_proxy_min_spacing').value
        )
        self.pgs_use_region_owned_keyframes = bool(
            self.get_parameter('pgs_use_region_owned_keyframes').value
        )
        self.pgs_filter_keyframes_by_region_entry = bool(
            self.get_parameter('pgs_filter_keyframes_by_region_entry').value
        )
        self.pgs_region_entry_anchor_node_count = max(
            0,
            int(self.get_parameter('pgs_region_entry_anchor_node_count').value),
        )
        self.pgs_min_waypoint_spacing = float(
            self.get_parameter('pgs_min_waypoint_spacing').value
        )
        self.pgs_use_costmap_waypoint_filter = bool(
            self.get_parameter('pgs_use_costmap_waypoint_filter').value
        )
        self.pgs_costmap_topic = self.get_parameter('pgs_costmap_topic').value
        self.pgs_waypoint_cost_threshold = int(
            self.get_parameter('pgs_waypoint_cost_threshold').value
        )
        self.pgs_waypoint_clearance = float(
            self.get_parameter('pgs_waypoint_clearance').value
        )
        self.pgs_waypoint_snap_radius = float(
            self.get_parameter('pgs_waypoint_snap_radius').value
        )
        self.pgs_min_safe_waypoints = int(
            self.get_parameter('pgs_min_safe_waypoints').value
        )
        self.pgs_require_costmap_path_connectivity = bool(
            self.get_parameter('pgs_require_costmap_path_connectivity').value
        )
        self.pgs_max_waypoint_path_length = float(
            self.get_parameter('pgs_max_waypoint_path_length').value
        )

        self.current_region = None
        self.current_region_id = None
        self.region_entry_min_node_id = None
        self.pose_graph_nodes = []
        self.pose_graph_frame_id = 'map'
        self.region_owned_keyframes = []
        self.region_keyframes_frame_id = 'map'
        self.region_keyframes_region_id = None
        self.region_keyframes_available = False
        self.latest_costmap: Optional[OccupancyGrid] = None
        self.last_waypoint_safety_debug = {}
        self.trajectory: List[Tuple[float, float]] = []
        self.last_pose: Optional[Tuple[float, float]] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            String, '/ralc/current_region', self.region_callback, 10
        )
        self.create_subscription(
            String, '/ralc/pose_graph_nodes', self.pose_graph_nodes_callback, 10
        )
        self.create_subscription(
            String, '/ralc/region_keyframes', self.region_keyframes_callback, 10
        )
        self.create_subscription(
            OccupancyGrid,
            self.pgs_costmap_topic,
            self.costmap_callback,
            10,
        )
        self.create_subscription(String, '/ralc/request_pgs', self.request_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 50)

        self.waypoints_pub = self.create_publisher(PoseArray, '/ralc/pgs_waypoints', 10)
        self.status_pub = self.create_publisher(String, '/ralc/pgs_status', 10)
        self.markers_pub = self.create_publisher(MarkerArray, '/ralc/pgs_markers', 10)
        self.get_logger().info(
            '[RALC] PGS planner ready: source=/ralc/pose_graph_nodes, '
            f'trajectory_proxy_fallback={self.use_trajectory_proxy_if_backend_missing}.'
        )

    def region_callback(self, msg: String):
        try:
            region = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.current_region = region if region else None
        region_id = self.current_region.get('region_id') if self.current_region else None
        if region_id != self.current_region_id:
            self.current_region_id = region_id
            self.region_entry_min_node_id = self.compute_region_entry_min_node_id()
            self.get_logger().info(
                '[RALC] PGS region entry: '
                f'region={region_id}, entry_min_node_id={self.region_entry_min_node_id}, '
                f'total_backend_nodes={len(self.pose_graph_nodes)}'
            )

    def pose_graph_nodes_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('[RALC] Ignoring malformed /ralc/pose_graph_nodes.')
            return
        nodes = data.get('nodes', [])
        parsed = []
        for node in nodes:
            try:
                parsed.append({
                    'id': int(node.get('id', len(parsed))),
                    'x': float(node['x']),
                    'y': float(node['y']),
                    'z': float(node.get('z', 0.0)),
                    'theta': float(node.get('theta', 0.0)),
                    'region_id': node.get('region_id'),
                    'region_assignment_reason': node.get(
                        'region_assignment_reason'
                    ),
                    'inside_current_region': node.get('inside_current_region'),
                })
            except (KeyError, TypeError, ValueError):
                continue
        self.pose_graph_nodes = parsed
        self.pose_graph_frame_id = data.get('frame_id', 'map') or 'map'
        if self.current_region is not None and self.region_entry_min_node_id is None:
            self.region_entry_min_node_id = self.compute_region_entry_min_node_id()
            self.get_logger().info(
                '[RALC] PGS initialized region entry after backend update: '
                f'region={self.current_region_id}, '
                f'entry_min_node_id={self.region_entry_min_node_id}, '
                f'total_backend_nodes={len(self.pose_graph_nodes)}'
            )

    def region_keyframes_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('[RALC] Ignoring malformed /ralc/region_keyframes.')
            return
        nodes = data.get('keyframes', [])
        parsed = []
        for node in nodes:
            try:
                parsed.append({
                    'id': int(node.get('id', len(parsed))),
                    'x': float(node['x']),
                    'y': float(node['y']),
                    'z': float(node.get('z', 0.0)),
                    'theta': float(node.get('theta', 0.0)),
                    'region_id': node.get('region_id'),
                    'region_assignment_reason': node.get(
                        'region_assignment_reason'
                    ),
                    'inside_current_region': node.get('inside_current_region'),
                })
            except (KeyError, TypeError, ValueError):
                continue
        self.region_owned_keyframes = parsed
        self.region_keyframes_frame_id = data.get('frame_id', 'map') or 'map'
        self.region_keyframes_region_id = data.get('current_region_id')
        self.region_keyframes_available = True

    def costmap_callback(self, msg: OccupancyGrid):
        self.latest_costmap = msg

    def odom_callback(self, msg: Odometry):
        if not self.use_trajectory_proxy_if_backend_missing:
            return
        xy = self.lookup_robot_xy()
        if xy is None:
            x = float(msg.pose.pose.position.x)
            y = float(msg.pose.pose.position.y)
        else:
            x, y = xy
        if self.last_pose is not None and math.hypot(
            x - self.last_pose[0], y - self.last_pose[1]
        ) < 0.10:
            return
        self.last_pose = (x, y)
        self.trajectory.append((x, y))

    def lookup_robot_xy(self) -> Optional[Tuple[float, float]]:
        for source_frame in ('base_footprint', 'base_link'):
            try:
                transform = self.tf_buffer.lookup_transform(
                    'map',
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.02),
                )
                t = transform.transform.translation
                return float(t.x), float(t.y)
            except TransformException:
                continue
        return None

    def request_callback(self, msg: String):
        mode = msg.data or 'regional'
        region_id = self.current_region.get('region_id') if self.current_region else None
        total_backend_nodes = len(self.pose_graph_nodes)

        regional_nodes, filter_debug = self.nodes_for_mode(mode)
        source = filter_debug.get('pgs_keyframe_source', 'pose_graph_nodes')
        if not regional_nodes and total_backend_nodes == 0:
            if not self.use_trajectory_proxy_if_backend_missing:
                self.publish_unavailable(
                    reason='PGS_UNAVAILABLE_BACKEND_MISSING',
                    mode=mode,
                    total_backend_nodes=0,
                    regional_keyframes=0,
                    region_id=region_id,
                    source=source,
                    filter_debug=filter_debug,
                )
                self.publish_markers([], [], [], 'map', region_id, source)
                return
            regional_nodes = self.proxy_nodes_for_mode(mode)
            source = 'trajectory_proxy'
            filter_debug = self.proxy_filter_debug(mode, regional_nodes)

        keyframe_points = [(node['x'], node['y']) for node in regional_nodes]
        deduped_points = self.deduplicate_points(
            keyframe_points,
            self.pgs_keyframe_dedup_distance,
        )
        regional_keyframes = len(deduped_points)
        if regional_keyframes < self.min_pgs_regional_keyframes:
            self.get_logger().warn(
                '[RALC] PGS unavailable: not enough regional keyframes. '
                f'region={region_id}, total_backend_nodes={total_backend_nodes}, '
                f'regional_keyframes={regional_keyframes}, source={source}, '
                f'filter={filter_debug}'
            )
            self.publish_unavailable(
                reason='PGS_UNAVAILABLE_NOT_ENOUGH_REGIONAL_KEYFRAMES',
                mode=mode,
                total_backend_nodes=total_backend_nodes,
                regional_keyframes=regional_keyframes,
                region_id=region_id,
                source=source,
                filter_debug=filter_debug,
            )
            self.publish_markers(deduped_points, [], [], 'map', region_id, source)
            return

        hull = self.monotonic_chain(deduped_points)
        if len(hull) < 3:
            self.get_logger().warn(
                '[RALC] PGS unavailable: degenerate regional keyframe hull. '
                f'region={region_id}, regional_keyframes={regional_keyframes}, '
                f'hull_vertices={len(hull)}, source={source}'
            )
            self.publish_unavailable(
                reason='PGS_UNAVAILABLE_DEGENERATE_HULL',
                mode=mode,
                total_backend_nodes=total_backend_nodes,
                regional_keyframes=regional_keyframes,
                region_id=region_id,
                source=source,
                hull_vertices=len(hull),
                filter_debug=filter_debug,
            )
            self.publish_markers(deduped_points, hull, [], 'map', region_id, source)
            return

        waypoints = self.hull_to_waypoints(hull)
        waypoint_safety = dict(self.last_waypoint_safety_debug)
        if (
            self.pgs_use_costmap_waypoint_filter and
            len(waypoints.poses) < self.pgs_min_safe_waypoints
        ):
            filter_debug.update(waypoint_safety)
            self.get_logger().warn(
                '[RALC] PGS unavailable: no safe costmap waypoints. '
                f'region={region_id}, raw_hull_vertices={len(hull)}, '
                f'safe_waypoints={len(waypoints.poses)}, source={source}, '
                f'filter={filter_debug}'
            )
            self.publish_unavailable(
                reason='PGS_UNAVAILABLE_NO_SAFE_WAYPOINTS',
                mode=mode,
                total_backend_nodes=total_backend_nodes,
                regional_keyframes=regional_keyframes,
                region_id=region_id,
                source=source,
                hull_vertices=len(hull),
                filter_debug=filter_debug,
            )
            self.publish_markers(deduped_points, hull, [], 'map', region_id, source)
            return
        filter_debug.update(waypoint_safety)
        self.waypoints_pub.publish(waypoints)
        self.publish_markers(
            deduped_points,
            hull,
            [(pose.position.x, pose.position.y) for pose in waypoints.poses],
            waypoints.header.frame_id,
            region_id,
            source,
        )
        self.publish_ready(
            mode=mode,
            region_id=region_id,
            total_backend_nodes=total_backend_nodes,
            regional_keyframes=regional_keyframes,
            hull_vertices=len(hull),
            waypoint_count=len(waypoints.poses),
            source=source,
            filter_debug=filter_debug,
        )
        self.get_logger().info(
            '[RALC] PGS_WAYPOINTS_READY: '
            f'region={region_id}, total_backend_nodes={total_backend_nodes}, '
            f'regional_keyframes={regional_keyframes}, hull_vertices={len(hull)}, '
            f'waypoints={len(waypoints.poses)}, source={source}, '
            f'filter={filter_debug}'
        )

    def nodes_for_mode(self, mode: str):
        debug = {
            'pgs_keyframe_filter': 'global' if mode == 'global' else 'region_entry',
            'pgs_keyframe_source': 'pose_graph_nodes',
            'region_status': (
                self.current_region.get('status')
                if self.current_region else None
            ),
            'region_owned_keyframes_available': self.region_keyframes_available,
            'region_owned_keyframes_region_id': self.region_keyframes_region_id,
            'region_owned_keyframes': len(self.region_owned_keyframes),
            'spatial_regional_keyframes': 0,
            'entry_filtered_prior_keyframes': 0,
            'region_entry_min_node_id': self.region_entry_min_node_id,
            'use_region_owned_keyframes': self.pgs_use_region_owned_keyframes,
            'region_entry_anchor_node_count': self.pgs_region_entry_anchor_node_count,
            'filter_keyframes_by_region_entry': self.pgs_filter_keyframes_by_region_entry,
            'selected_keyframe_min_id': None,
            'selected_keyframe_max_id': None,
        }
        if mode == 'global':
            debug['spatial_regional_keyframes'] = len(self.pose_graph_nodes)
            self.update_selected_keyframe_id_debug(debug, self.pose_graph_nodes)
            return list(self.pose_graph_nodes), debug
        if self.current_region is None:
            return [], debug
        region_id = self.current_region.get('region_id')
        if (
            self.pgs_use_region_owned_keyframes and
            self.region_keyframes_available and
            self.region_keyframes_region_id == region_id
        ):
            regional_nodes = list(self.region_owned_keyframes)
            debug['pgs_keyframe_filter'] = 'region_owned_keyframes'
            debug['pgs_keyframe_source'] = 'region_keyframes'
            debug['spatial_regional_keyframes'] = len(regional_nodes)
            self.update_selected_keyframe_id_debug(debug, regional_nodes)
            return regional_nodes, debug
        spatial_nodes = [
            node for node in self.pose_graph_nodes
            if self.point_inside_region(node['x'], node['y'], self.current_region)
        ]
        debug['pgs_keyframe_filter'] = 'spatial_region_entry_fallback'
        debug['spatial_regional_keyframes'] = len(spatial_nodes)
        if (
            self.pgs_filter_keyframes_by_region_entry and
            self.region_entry_min_node_id is not None
        ):
            regional_nodes = [
                node for node in spatial_nodes
                if node.get('id', -1) >= self.region_entry_min_node_id
            ]
            debug['entry_filtered_prior_keyframes'] = (
                len(spatial_nodes) - len(regional_nodes)
            )
            self.update_selected_keyframe_id_debug(debug, regional_nodes)
            return regional_nodes, debug
        self.update_selected_keyframe_id_debug(debug, spatial_nodes)
        return spatial_nodes, debug

    def update_selected_keyframe_id_debug(self, debug, nodes):
        if not nodes:
            return
        ids = [int(node.get('id', 0)) for node in nodes]
        debug['selected_keyframe_min_id'] = min(ids)
        debug['selected_keyframe_max_id'] = max(ids)

    def compute_region_entry_min_node_id(self):
        if not self.pose_graph_nodes:
            return None
        max_node_id = max(node.get('id', 0) for node in self.pose_graph_nodes)
        anchor_count = self.pgs_region_entry_anchor_node_count
        if anchor_count <= 0:
            return max_node_id + 1
        return max(0, max_node_id - anchor_count + 1)

    def proxy_filter_debug(self, mode: str, regional_nodes):
        return {
            'pgs_keyframe_filter': 'trajectory_proxy',
            'spatial_regional_keyframes': len(regional_nodes),
            'entry_filtered_prior_keyframes': 0,
            'region_entry_min_node_id': None,
            'region_entry_anchor_node_count': 0,
            'filter_keyframes_by_region_entry': False,
            'mode': mode,
        }

    def proxy_nodes_for_mode(self, mode: str):
        points = self.trajectory
        if mode != 'global' and self.current_region is not None:
            points = [
                point for point in points
                if self.point_inside_region(point[0], point[1], self.current_region)
            ]
        points = self.deduplicate_points(points, self.trajectory_proxy_min_spacing)
        return [
            {'id': index, 'x': point[0], 'y': point[1], 'z': 0.0, 'theta': 0.0}
            for index, point in enumerate(points)
        ]

    def point_inside_region(self, x, y, region):
        return (
            float(region['xmin']) <= x <= float(region['xmax']) and
            float(region['ymin']) <= y <= float(region['ymax'])
        )

    def deduplicate_points(self, points, min_distance):
        deduped = []
        for point in points:
            if any(
                math.hypot(point[0] - existing[0], point[1] - existing[1]) <
                min_distance
                for existing in deduped
            ):
                continue
            deduped.append((float(point[0]), float(point[1])))
        return deduped

    def monotonic_chain(self, points):
        unique = sorted(set(points))
        if len(unique) <= 1:
            return unique

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in unique:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(unique):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    def hull_to_waypoints(self, hull):
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        clockwise = list(reversed(hull))
        counter_clockwise = list(hull)
        sequence = clockwise + counter_clockwise[1:]
        if self.close_hull_loop and sequence:
            sequence.append(sequence[0])
        sequence = self.prune_waypoint_sequence(sequence)
        sequence = self.costmap_safe_waypoint_sequence(sequence)

        for index, (x, y) in enumerate(sequence):
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            if index + 1 < len(sequence):
                nx, ny = sequence[index + 1]
                yaw = math.atan2(ny - y, nx - x)
            else:
                yaw = 0.0
            pose.orientation.z = math.sin(0.5 * yaw)
            pose.orientation.w = math.cos(0.5 * yaw)
            msg.poses.append(pose)
        return msg

    def costmap_safe_waypoint_sequence(self, sequence):
        self.last_waypoint_safety_debug = {
            'pgs_costmap_filter_enabled': self.pgs_use_costmap_waypoint_filter,
            'pgs_costmap_available': self.latest_costmap is not None,
            'pgs_raw_waypoints': len(sequence),
            'pgs_safe_waypoints': len(sequence),
            'pgs_snapped_waypoints': 0,
            'pgs_dropped_unsafe_waypoints': 0,
            'pgs_waypoint_cost_threshold': self.pgs_waypoint_cost_threshold,
            'pgs_waypoint_clearance': self.pgs_waypoint_clearance,
            'pgs_waypoint_snap_radius': self.pgs_waypoint_snap_radius,
            'pgs_path_connectivity_required': (
                self.pgs_require_costmap_path_connectivity
            ),
            'pgs_path_reachable_waypoints': 0,
            'pgs_dropped_unreachable_waypoints': 0,
        }
        if not self.pgs_use_costmap_waypoint_filter:
            return sequence
        if self.latest_costmap is None:
            self.last_waypoint_safety_debug['pgs_safety_reason'] = (
                'COSTMAP_UNAVAILABLE_RAW_WAYPOINTS'
            )
            return sequence
        adjusted = []
        snapped = 0
        dropped = 0
        for point in sequence:
            safe_point = self.safe_or_snapped_waypoint(point)
            if safe_point is None:
                dropped += 1
                continue
            if math.hypot(safe_point[0] - point[0], safe_point[1] - point[1]) > 1e-6:
                snapped += 1
            adjusted.append(safe_point)
        adjusted = self.prune_duplicate_waypoints(adjusted)
        adjusted = self.prune_waypoint_sequence(adjusted)
        adjusted, path_debug = self.reachable_waypoint_sequence(adjusted)
        self.last_waypoint_safety_debug.update({
            'pgs_safe_waypoints': len(adjusted),
            'pgs_snapped_waypoints': snapped,
            'pgs_dropped_unsafe_waypoints': dropped,
        })
        self.last_waypoint_safety_debug.update(path_debug)
        return adjusted

    def reachable_waypoint_sequence(self, sequence):
        debug = {
            'pgs_path_connectivity_required': (
                self.pgs_require_costmap_path_connectivity
            ),
            'pgs_path_reachable_waypoints': len(sequence),
            'pgs_dropped_unreachable_waypoints': 0,
            'pgs_robot_pose_for_path': None,
        }
        if not self.pgs_require_costmap_path_connectivity:
            return sequence, debug
        if self.latest_costmap is None:
            debug['pgs_path_reason'] = 'COSTMAP_UNAVAILABLE'
            return sequence, debug
        robot_xy = self.lookup_robot_xy()
        if robot_xy is None:
            debug['pgs_path_reason'] = 'ROBOT_POSE_UNAVAILABLE'
            return sequence, debug
        debug['pgs_robot_pose_for_path'] = [robot_xy[0], robot_xy[1]]
        previous = self.safe_or_snapped_waypoint(robot_xy)
        if previous is None:
            debug['pgs_path_reason'] = 'ROBOT_POSE_NOT_IN_SAFE_COSTMAP'
            debug['pgs_path_reachable_waypoints'] = 0
            debug['pgs_dropped_unreachable_waypoints'] = len(sequence)
            return [], debug
        reachable = []
        dropped = 0
        for point in sequence:
            path_length = self.costmap_path_length(previous, point)
            if path_length is None:
                dropped += 1
                continue
            if (
                self.pgs_max_waypoint_path_length > 0.0 and
                path_length > self.pgs_max_waypoint_path_length
            ):
                dropped += 1
                continue
            reachable.append(point)
            previous = point
        debug['pgs_path_reachable_waypoints'] = len(reachable)
        debug['pgs_dropped_unreachable_waypoints'] = dropped
        return reachable, debug

    def costmap_path_length(self, start_world, goal_world):
        grid = self.latest_costmap
        start = self.world_to_cell(grid, start_world[0], start_world[1])
        goal = self.world_to_cell(grid, goal_world[0], goal_world[1])
        if start is None or goal is None:
            return None
        if not self.is_safe_costmap_cell(grid, start[0], start[1]):
            return None
        if not self.is_safe_costmap_cell(grid, goal[0], goal[1]):
            return None
        if start == goal:
            return 0.0
        resolution = float(grid.info.resolution)
        max_cost = None
        if self.pgs_max_waypoint_path_length > 0.0:
            max_cost = self.pgs_max_waypoint_path_length / resolution
        frontier = [(0.0, start)]
        best = {start: 0.0}
        visited = set()
        while frontier:
            frontier.sort(key=lambda item: item[0])
            cost, cell = frontier.pop(0)
            if cell in visited:
                continue
            visited.add(cell)
            if cell == goal:
                return cost * resolution
            if max_cost is not None and cost > max_cost:
                continue
            for nx, ny, step_cost in self.safe_neighbors(grid, cell[0], cell[1]):
                next_cell = (nx, ny)
                next_cost = cost + step_cost
                if max_cost is not None and next_cost > max_cost:
                    continue
                if next_cost >= best.get(next_cell, float('inf')):
                    continue
                best[next_cell] = next_cost
                frontier.append((next_cost, next_cell))
        return None

    def safe_neighbors(self, grid, cell_x, cell_y):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = cell_x + dx
                ny = cell_y + dy
                if not self.is_safe_costmap_cell(grid, nx, ny):
                    continue
                if dx != 0 and dy != 0:
                    if (
                        not self.is_safe_costmap_cell(grid, cell_x + dx, cell_y) or
                        not self.is_safe_costmap_cell(grid, cell_x, cell_y + dy)
                    ):
                        continue
                    yield nx, ny, math.sqrt(2.0)
                else:
                    yield nx, ny, 1.0

    def safe_or_snapped_waypoint(self, point):
        grid = self.latest_costmap
        cell = self.world_to_cell(grid, point[0], point[1])
        if cell is not None and self.is_safe_costmap_cell(grid, cell[0], cell[1]):
            return point
        if cell is None:
            return None
        resolution = float(grid.info.resolution)
        max_radius_cells = max(1, int(math.ceil(
            self.pgs_waypoint_snap_radius / resolution
        )))
        candidates = []
        for radius_cells in range(1, max_radius_cells + 1):
            candidates.clear()
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    if max(abs(dx), abs(dy)) != radius_cells:
                        continue
                    cx = cell[0] + dx
                    cy = cell[1] + dy
                    if not self.is_safe_costmap_cell(grid, cx, cy):
                        continue
                    world = self.cell_to_world(grid, cx, cy)
                    distance = math.hypot(world[0] - point[0], world[1] - point[1])
                    candidates.append((distance, world))
            if candidates:
                candidates.sort(key=lambda item: item[0])
                return candidates[0][1]
        return None

    def is_safe_costmap_cell(self, grid, cell_x, cell_y):
        if (
            cell_x < 0 or cell_x >= grid.info.width or
            cell_y < 0 or cell_y >= grid.info.height
        ):
            return False
        resolution = float(grid.info.resolution)
        clearance_cells = max(0, int(math.ceil(
            self.pgs_waypoint_clearance / resolution
        )))
        for dx in range(-clearance_cells, clearance_cells + 1):
            for dy in range(-clearance_cells, clearance_cells + 1):
                if dx * dx + dy * dy > clearance_cells * clearance_cells:
                    continue
                cx = cell_x + dx
                cy = cell_y + dy
                if (
                    cx < 0 or cx >= grid.info.width or
                    cy < 0 or cy >= grid.info.height
                ):
                    return False
                value = grid.data[cy * grid.info.width + cx]
                if value < 0 or value > self.pgs_waypoint_cost_threshold:
                    return False
        return True

    def world_to_cell(self, grid, world_x, world_y):
        origin = grid.info.origin.position
        resolution = float(grid.info.resolution)
        cell_x = int((world_x - origin.x) / resolution)
        cell_y = int((world_y - origin.y) / resolution)
        if (
            cell_x < 0 or cell_x >= grid.info.width or
            cell_y < 0 or cell_y >= grid.info.height
        ):
            return None
        return cell_x, cell_y

    def cell_to_world(self, grid, cell_x, cell_y):
        origin = grid.info.origin.position
        resolution = float(grid.info.resolution)
        return (
            float(origin.x + (cell_x + 0.5) * resolution),
            float(origin.y + (cell_y + 0.5) * resolution),
        )

    def prune_duplicate_waypoints(self, sequence):
        deduped = []
        for point in sequence:
            if deduped and math.hypot(
                point[0] - deduped[-1][0],
                point[1] - deduped[-1][1],
            ) < 0.05:
                continue
            deduped.append(point)
        return deduped

    def prune_waypoint_sequence(self, sequence):
        if not sequence:
            return []
        pruned = [sequence[0]]
        for point in sequence[1:]:
            if math.hypot(
                point[0] - pruned[-1][0],
                point[1] - pruned[-1][1],
            ) < self.pgs_min_waypoint_spacing:
                continue
            pruned.append(point)
        if (
            self.close_hull_loop and len(pruned) > 2 and
            math.hypot(
                pruned[0][0] - pruned[-1][0],
                pruned[0][1] - pruned[-1][1],
            ) < self.pgs_min_waypoint_spacing
        ):
            pruned[-1] = pruned[0]
        return pruned

    def publish_unavailable(
        self,
        reason,
        mode,
        total_backend_nodes,
        regional_keyframes,
        region_id,
        source,
        hull_vertices=0,
        filter_debug=None,
    ):
        payload = {
            'mode': mode,
            'status': 'UNAVAILABLE',
            'reason': reason,
            'message': reason,
            'success': False,
            'region_id': region_id,
            'total_backend_nodes': total_backend_nodes,
            'regional_keyframes': regional_keyframes,
            'hull_vertices': hull_vertices,
            'waypoint_count': 0,
            'waypoints': 0,
            'source': source,
        }
        if filter_debug:
            payload.update(filter_debug)
        msg = String()
        msg.data = json.dumps(payload)
        self.status_pub.publish(msg)

    def publish_ready(
        self,
        mode,
        region_id,
        total_backend_nodes,
        regional_keyframes,
        hull_vertices,
        waypoint_count,
        source,
        filter_debug=None,
    ):
        payload = {
            'mode': mode,
            'status': 'PGS_WAYPOINTS_READY',
            'reason': 'PGS_WAYPOINTS_READY',
            'message': 'PGS_WAYPOINTS_READY',
            'success': True,
            'region_id': region_id,
            'total_backend_nodes': total_backend_nodes,
            'regional_keyframes': regional_keyframes,
            'hull_vertices': hull_vertices,
            'waypoint_count': waypoint_count,
            'waypoints': waypoint_count,
            'source': source,
        }
        if filter_debug:
            payload.update(filter_debug)
        msg = String()
        msg.data = json.dumps(payload)
        self.status_pub.publish(msg)

    def publish_markers(self, keyframes, hull, waypoints, frame_id, region_id, source):
        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)
        stamp = self.get_clock().now().to_msg()

        if keyframes:
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = frame_id
            marker.ns = 'ralc_pgs_regional_keyframes'
            marker.id = 1
            marker.type = Marker.SPHERE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.12
            marker.scale.y = 0.12
            marker.scale.z = 0.12
            marker.color.g = 0.9
            marker.color.b = 1.0
            marker.color.a = 0.9
            for x, y in keyframes:
                point = Point()
                point.x = x
                point.y = y
                point.z = 0.18
                marker.points.append(point)
            marker_array.markers.append(marker)

        if hull:
            line = Marker()
            line.header.stamp = stamp
            line.header.frame_id = frame_id
            line.ns = 'ralc_pgs_hull'
            line.id = 2
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.pose.orientation.w = 1.0
            line.scale.x = 0.07
            line.color.r = 1.0
            line.color.g = 0.45
            line.color.a = 0.95
            closed = hull + [hull[0]] if len(hull) > 2 else hull
            for x, y in closed:
                point = Point()
                point.x = x
                point.y = y
                point.z = 0.32
                line.points.append(point)
            marker_array.markers.append(line)

        if waypoints:
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = frame_id
            marker.ns = 'ralc_pgs_waypoints'
            marker.id = 3
            marker.type = Marker.SPHERE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.18
            marker.scale.y = 0.18
            marker.scale.z = 0.18
            marker.color.r = 0.25
            marker.color.g = 1.0
            marker.color.a = 0.95
            for x, y in waypoints:
                point = Point()
                point.x = x
                point.y = y
                point.z = 0.45
                marker.points.append(point)
            marker_array.markers.append(marker)

        label = Marker()
        label.header.stamp = stamp
        label.header.frame_id = frame_id
        label.ns = 'ralc_pgs_label'
        label.id = 4
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.orientation.w = 1.0
        label.pose.position.z = 0.85
        label.scale.z = 0.28
        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.color.a = 0.95
        label.text = (
            f'PGS region={region_id} keyframes={len(keyframes)} '
            f'hull={len(hull)} source={source}'
        )
        anchor_points = hull or keyframes or waypoints
        if anchor_points:
            label.pose.position.x = sum(point[0] for point in anchor_points) / len(
                anchor_points
            )
            label.pose.position.y = sum(point[1] for point in anchor_points) / len(
                anchor_points
            )
        marker_array.markers.append(label)

        self.markers_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = RalcPgsPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
