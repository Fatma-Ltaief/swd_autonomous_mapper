import json
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class RalcSlamBackend(Node):
    """
    Adapter between SLAM Toolbox graph_visualization and R-ALC.

    SLAM Toolbox currently exposes pose graph information as RViz markers on:
        /slam_toolbox/graph_visualization

    This node converts that visualization-only topic into clean R-ALC backend topics:

        /ralc/pose_graph_nodes
        /ralc/pose_graph_edges
        /ralc/region_keyframes
        /ralc/slam_backend_status
        /ralc/pose_uncertainty
        /ralc/keyframe_information_matrix
        /ralc/expected_loop_closure_gain
        /ralc/alc_candidates

    First purpose:
        provide real SLAM Toolbox pose-graph nodes/keyframes to the R-ALC PGS planner.

    This does NOT yet provide covariance/uncertainty for ALC.
    """

    def __init__(self):
        super().__init__('ralc_slam_backend')

        self.declare_parameter(
            'graph_visualization_topic',
            '/slam_toolbox/graph_visualization',
        )
        self.declare_parameter('node_marker_namespace', 'slam_toolbox')
        self.declare_parameter('edge_marker_namespace', 'slam_toolbox_edges')
        self.declare_parameter('publish_period_sec', 1.0)
        self.declare_parameter('deduplicate_position_epsilon', 1e-4)
        self.declare_parameter('assign_keyframes_to_refinement_regions', True)

        self.graph_topic = str(
            self.get_parameter('graph_visualization_topic').value
        )
        self.node_ns = str(
            self.get_parameter('node_marker_namespace').value
        )
        self.edge_ns = str(
            self.get_parameter('edge_marker_namespace').value
        )
        self.publish_period = float(
            self.get_parameter('publish_period_sec').value
        )
        self.dedup_eps = float(
            self.get_parameter('deduplicate_position_epsilon').value
        )
        self.assign_keyframes_to_refinement_regions = bool(
            self.get_parameter('assign_keyframes_to_refinement_regions').value
        )

        self.current_region: Optional[Dict] = None
        self.latest_frame_id = 'map'
        self.latest_stamp = None
        self.latest_nodes: List[Dict] = []
        self.latest_edges: List[Dict] = []
        self.latest_marker_count = 0
        self.graph_msg_count = 0
        self.keyframe_ownership: Dict[int, Dict] = {}

        self.region_sub = self.create_subscription(
            String,
            '/ralc/current_region',
            self.region_callback,
            10,
        )
        self.graph_sub = self.create_subscription(
            MarkerArray,
            self.graph_topic,
            self.graph_callback,
            10,
        )

        self.nodes_pub = self.create_publisher(
            String,
            '/ralc/pose_graph_nodes',
            10,
        )
        self.edges_pub = self.create_publisher(
            String,
            '/ralc/pose_graph_edges',
            10,
        )
        self.region_keyframes_pub = self.create_publisher(
            String,
            '/ralc/region_keyframes',
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            '/ralc/slam_backend_status',
            10,
        )
        self.pose_uncertainty_pub = self.create_publisher(
            String,
            '/ralc/pose_uncertainty',
            10,
        )
        self.information_matrix_pub = self.create_publisher(
            String,
            '/ralc/keyframe_information_matrix',
            10,
        )
        self.expected_loop_closure_gain_pub = self.create_publisher(
            String,
            '/ralc/expected_loop_closure_gain',
            10,
        )
        self.alc_candidates_pub = self.create_publisher(
            String,
            '/ralc/alc_candidates',
            10,
        )

        self.timer = self.create_timer(
            self.publish_period,
            self.publish_backend_snapshot,
        )

        self.get_logger().info(
            '[RALC_BACKEND] SLAM backend adapter ready. '
            f'Subscribing to {self.graph_topic}; publishing '
            '/ralc/pose_graph_nodes, /ralc/pose_graph_edges, '
            '/ralc/region_keyframes, /ralc/slam_backend_status, and '
            'honest ALC backend availability topics.'
        )

    def region_callback(self, msg: String):
        try:
            region = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.current_region = region if region else None

    def graph_callback(self, msg: MarkerArray):
        self.graph_msg_count += 1
        self.latest_marker_count = len(msg.markers)

        nodes = []
        edges = []

        for marker in msg.markers:
            if marker.action == Marker.DELETEALL:
                continue

            if marker.header.frame_id:
                self.latest_frame_id = marker.header.frame_id
                self.latest_stamp = marker.header.stamp

            if marker.ns == self.node_ns and marker.type in (
                Marker.SPHERE,
                Marker.CUBE,
                Marker.CYLINDER,
            ):
                node = self.extract_node(marker)
                if node is not None:
                    nodes.append(node)

            elif marker.ns == self.edge_ns and marker.type == Marker.LINE_LIST:
                edges.extend(self.extract_edges(marker))

        self.latest_nodes = self.annotate_region_ownership(
            self.deduplicate_nodes(nodes)
        )
        self.latest_edges = edges

    def extract_node(self, marker: Marker):
        x = float(marker.pose.position.x)
        y = float(marker.pose.position.y)
        z = float(marker.pose.position.z)

        # SLAM Toolbox node markers appear as SPHERE markers whose ids are graph node ids.
        return {
            'id': int(marker.id),
            'x': x,
            'y': y,
            'z': z,
            'theta': 0.0,
            'source': 'slam_toolbox_graph_visualization',
        }

    def extract_edges(self, marker: Marker):
        edges = []
        points = marker.points

        # LINE_LIST stores edges as pairs:
        # points[0] -> points[1], points[2] -> points[3], ...
        for i in range(0, len(points) - 1, 2):
            p0 = points[i]
            p1 = points[i + 1]
            edges.append({
                'id': len(edges),
                'from_x': float(p0.x),
                'from_y': float(p0.y),
                'from_z': float(p0.z),
                'to_x': float(p1.x),
                'to_y': float(p1.y),
                'to_z': float(p1.z),
                'source': 'slam_toolbox_graph_visualization',
            })

        return edges

    def deduplicate_nodes(self, nodes: List[Dict]):
        """
        RViz marker streams can occasionally contain repeated/overlapping markers.
        Keep a stable list of unique graph nodes by position.
        """
        unique = []
        seen: List[Tuple[float, float]] = []

        for node in sorted(nodes, key=lambda n: n['id']):
            x = node['x']
            y = node['y']

            duplicate = False
            for sx, sy in seen:
                if abs(x - sx) <= self.dedup_eps and abs(y - sy) <= self.dedup_eps:
                    duplicate = True
                    break

            if duplicate:
                continue

            node = dict(node)
            node['backend_index'] = len(unique)
            unique.append(node)
            seen.append((x, y))

        return unique

    def annotate_region_ownership(self, nodes: List[Dict]):
        annotated = []
        active_region = self.current_region
        for node in nodes:
            node = dict(node)
            node_id = int(node['id'])
            owner = self.keyframe_ownership.get(node_id)
            if owner is None:
                owner = self.assign_owner_for_node(node)
                if owner is not None:
                    self.keyframe_ownership[node_id] = owner
            spatial_region_id = self.spatial_region_id_for_node(node)
            if owner is not None:
                node.update({
                    'region_id': owner.get('region_id'),
                    'region_status_at_assignment': owner.get(
                        'region_status_at_assignment'
                    ),
                    'region_assignment_reason': owner.get('assignment_reason'),
                    'assigned_region_bounds': owner.get('region_bounds'),
                    'region_assignment_node_x': owner.get('node_x'),
                    'region_assignment_node_y': owner.get('node_y'),
                })
            else:
                node.update({
                    'region_id': None,
                    'region_status_at_assignment': None,
                    'region_assignment_reason': 'UNASSIGNED_NO_ACTIVE_REGION_MATCH',
                    'assigned_region_bounds': None,
                    'region_assignment_node_x': None,
                    'region_assignment_node_y': None,
                })
            node['spatial_current_region_id'] = spatial_region_id
            node['inside_current_region'] = (
                active_region is not None and
                spatial_region_id == active_region.get('region_id')
            )
            annotated.append(node)
        return annotated

    def assign_owner_for_node(self, node: Dict):
        region = self.current_region
        if not region:
            return None
        status = str(region.get('status', ''))
        if status not in ('ACTIVE', 'REFINEMENT'):
            return None
        if status == 'REFINEMENT' and not self.assign_keyframes_to_refinement_regions:
            return None
        if not self.point_inside_region(node['x'], node['y'], region):
            return None
        return {
            'region_id': region.get('region_id'),
            'region_status_at_assignment': status,
            'assignment_reason': 'FIRST_SEEN_INSIDE_ACTIVE_REGION',
            'region_bounds': {
                'xmin': region.get('xmin'),
                'xmax': region.get('xmax'),
                'ymin': region.get('ymin'),
                'ymax': region.get('ymax'),
            },
            'node_x': float(node['x']),
            'node_y': float(node['y']),
        }

    def spatial_region_id_for_node(self, node: Dict):
        if self.current_region and self.point_inside_region(
            node['x'],
            node['y'],
            self.current_region,
        ):
            return self.current_region.get('region_id')
        return None

    def point_inside_region(self, x, y, region):
        return (
            float(region['xmin']) <= float(x) <= float(region['xmax']) and
            float(region['ymin']) <= float(y) <= float(region['ymax'])
        )

    def region_keyframe_nodes(self, region_id):
        if region_id is None:
            return []
        return [
            node for node in self.latest_nodes
            if node.get('region_id') == region_id
        ]

    def ownership_counts(self):
        counts = {}
        for node in self.latest_nodes:
            region_id = node.get('region_id')
            key = 'unassigned' if region_id is None else str(region_id)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def publish_backend_snapshot(self):
        stamp_sec = 0
        stamp_nanosec = 0
        if self.latest_stamp is not None:
            stamp_sec = int(self.latest_stamp.sec)
            stamp_nanosec = int(self.latest_stamp.nanosec)

        nodes_msg = String()
        nodes_msg.data = json.dumps({
            'backend': 'slam_toolbox_graph_visualization',
            'frame_id': self.latest_frame_id,
            'stamp_sec': stamp_sec,
            'stamp_nanosec': stamp_nanosec,
            'node_count': len(self.latest_nodes),
            'nodes': self.latest_nodes,
        })
        self.nodes_pub.publish(nodes_msg)

        edges_msg = String()
        edges_msg.data = json.dumps({
            'backend': 'slam_toolbox_graph_visualization',
            'frame_id': self.latest_frame_id,
            'stamp_sec': stamp_sec,
            'stamp_nanosec': stamp_nanosec,
            'edge_count': len(self.latest_edges),
            'edges': self.latest_edges,
        })
        self.edges_pub.publish(edges_msg)

        current_region_id = (
            self.current_region.get('region_id')
            if self.current_region else None
        )
        current_region_status = (
            self.current_region.get('status')
            if self.current_region else None
        )
        region_keyframes = self.region_keyframe_nodes(current_region_id)
        region_msg = String()
        region_msg.data = json.dumps({
            'backend': 'slam_toolbox_graph_visualization',
            'frame_id': self.latest_frame_id,
            'stamp_sec': stamp_sec,
            'stamp_nanosec': stamp_nanosec,
            'current_region_id': current_region_id,
            'current_region_status': current_region_status,
            'region_keyframe_count': len(region_keyframes),
            'total_backend_nodes': len(self.latest_nodes),
            'ownership_counts': self.ownership_counts(),
            'keyframes': region_keyframes,
            'note': (
                'Region ownership is R-ALC bookkeeping assigned when a '
                'SLAM Toolbox keyframe is first observed inside an active '
                'or refinement region. It is not SLAM Toolbox marginalization.'
            ),
        })
        self.region_keyframes_pub.publish(region_msg)
        self.publish_unavailable_alc_interfaces(
            stamp_sec,
            stamp_nanosec,
            current_region_id,
            current_region_status,
        )

        status_msg = String()
        status_msg.data = json.dumps({
            'backend': 'slam_toolbox_graph_visualization',
            'status': 'READY' if len(self.latest_nodes) > 0 else 'WAITING_FOR_GRAPH',
            'graph_topic': self.graph_topic,
            'graph_messages_received': self.graph_msg_count,
            'marker_count': self.latest_marker_count,
            'node_count': len(self.latest_nodes),
            'edge_count': len(self.latest_edges),
            'frame_id': self.latest_frame_id,
            'has_keyframes_for_pgs': len(self.latest_nodes) >= 3,
            'has_covariance_for_alc': False,
            'has_information_matrix_for_alc': False,
            'has_expected_loop_closure_gain': False,
            'has_alc_candidates': False,
            'has_region_keyframe_ownership': True,
            'current_region_id': current_region_id,
            'current_region_status': current_region_status,
            'current_region_keyframe_count': len(region_keyframes),
            'ownership_counts': self.ownership_counts(),
            'note': (
                'This adapter exposes pose graph node/edge geometry from '
                'SLAM Toolbox graph_visualization and R-ALC region ownership '
                'bookkeeping. It does not expose pose covariance, information '
                'matrices, or expected loop-closure gain yet.'
            ),
        })
        self.status_pub.publish(status_msg)

        self.get_logger().debug(
            '[RALC_BACKEND] Published backend snapshot: '
            f'nodes={len(self.latest_nodes)}, edges={len(self.latest_edges)}, '
            f'markers={self.latest_marker_count}.'
        )

    def publish_unavailable_alc_interfaces(
        self,
        stamp_sec,
        stamp_nanosec,
        current_region_id,
        current_region_status,
    ):
        base_payload = {
            'backend': 'slam_toolbox_graph_visualization',
            'status': 'UNAVAILABLE',
            'frame_id': self.latest_frame_id,
            'stamp_sec': stamp_sec,
            'stamp_nanosec': stamp_nanosec,
            'current_region_id': current_region_id,
            'current_region_status': current_region_status,
            'reason': 'SLAM_TOOLBOX_MARKERS_DO_NOT_EXPOSE_ALC_UNCERTAINTY',
            'message': (
                'The current backend reads SLAM Toolbox RViz graph markers, '
                'which contain geometry but not covariance, information '
                'matrices, or expected loop-closure gain.'
            ),
        }
        topics = (
            (
                self.pose_uncertainty_pub,
                'pose_uncertainty',
                'has_covariance_for_alc',
            ),
            (
                self.information_matrix_pub,
                'keyframe_information_matrix',
                'has_information_matrix_for_alc',
            ),
            (
                self.expected_loop_closure_gain_pub,
                'expected_loop_closure_gain',
                'has_expected_loop_closure_gain',
            ),
            (
                self.alc_candidates_pub,
                'alc_candidates',
                'has_alc_candidates',
            ),
        )
        for publisher, interface_name, capability_flag in topics:
            msg = String()
            payload = dict(base_payload)
            payload.update({
                'interface': interface_name,
                capability_flag: False,
            })
            if interface_name == 'alc_candidates':
                payload['candidates'] = []
            msg.data = json.dumps(payload)
            publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RalcSlamBackend()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
