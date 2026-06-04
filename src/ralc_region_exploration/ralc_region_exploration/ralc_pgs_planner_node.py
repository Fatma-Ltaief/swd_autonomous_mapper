import json
import math
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class RalcPgsPlanner(Node):
    """Pose-graph-surrogate refinement planner.

    True R-ALC PGS uses regional keyframes. Until the SLAM backend exposes
    keyframes cleanly, this node uses robot trajectory poses inside the region
    as a clearly logged proxy and publishes hull waypoints.
    """

    def __init__(self):
        super().__init__('ralc_pgs_planner')

        self.declare_parameter('trajectory_proxy_min_spacing', 0.75)
        self.declare_parameter('min_waypoint_spacing', 0.4)
        self.trajectory_proxy_min_spacing = float(
            self.get_parameter('trajectory_proxy_min_spacing').value
        )
        self.min_waypoint_spacing = float(
            self.get_parameter('min_waypoint_spacing').value
        )

        self.current_region = None
        self.trajectory: List[Tuple[float, float]] = []
        self.last_pose: Optional[Tuple[float, float]] = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(String, '/ralc/current_region', self.region_callback, 10)
        self.create_subscription(String, '/ralc/request_pgs', self.request_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 50)
        self.waypoints_pub = self.create_publisher(PoseArray, '/ralc/pgs_waypoints', 10)
        self.status_pub = self.create_publisher(String, '/ralc/pgs_status', 10)
        self.markers_pub = self.create_publisher(MarkerArray, '/ralc/pgs_markers', 10)
        self.get_logger().info('[RALC] PGS planner ready.')

    def region_callback(self, msg: String):
        try:
            region = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.current_region = region if region else None

    def odom_callback(self, msg: Odometry):
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
        raw_points = self.proxy_points_for_mode(mode)
        points = self.downsample_points(raw_points, self.trajectory_proxy_min_spacing)
        self.publish_status(
            mode=mode,
            status_text='READY',
            success=False,
            reason='PGS request received.',
            raw_points=len(raw_points),
            downsampled_points=len(points),
            hull_vertices=0,
            waypoints=0,
            region_id=region_id,
        )
        self.get_logger().warn(
            '[RALC] Backend keyframes unavailable; using downsampled '
            'trajectory proxy for PGS.'
        )
        if len(points) < 3:
            self.get_logger().warn(
                f'[RALC] PGS unavailable: no regional keyframe/trajectory '
                f'points. mode={mode}, raw_points={len(raw_points)}, '
                f'downsampled_points={len(points)}'
            )
            self.publish_status(
                mode=mode,
                status_text='UNAVAILABLE',
                success=False,
                reason='PGS_UNAVAILABLE: fewer than 3 downsampled trajectory proxy points.',
                raw_points=len(raw_points),
                downsampled_points=len(points),
                hull_vertices=0,
                waypoints=0,
                region_id=region_id,
            )
            self.publish_markers([], 'map')
            return
        hull = self.monotonic_chain(points)
        if len(hull) < 3:
            self.get_logger().warn(
                f'[RALC] PGS unavailable: convex hull has fewer than 3 '
                f'vertices. mode={mode}, hull_vertices={len(hull)}'
            )
            self.publish_status(
                mode=mode,
                status_text='UNAVAILABLE',
                success=False,
                reason='PGS_UNAVAILABLE: convex hull has fewer than 3 vertices.',
                raw_points=len(raw_points),
                downsampled_points=len(points),
                hull_vertices=len(hull),
                waypoints=0,
                region_id=region_id,
            )
            self.publish_markers(hull, 'map')
            return
        waypoints = self.hull_to_waypoints(hull)
        if len(waypoints.poses) == 0:
            self.publish_status(
                mode=mode,
                status_text='UNAVAILABLE',
                success=False,
                reason='PGS_UNAVAILABLE: all hull waypoints were too close to robot/previous waypoint.',
                raw_points=len(raw_points),
                downsampled_points=len(points),
                hull_vertices=len(hull),
                waypoints=0,
                region_id=region_id,
            )
            self.publish_markers(hull, 'map')
            return
        self.waypoints_pub.publish(waypoints)
        self.publish_markers(hull, waypoints.header.frame_id)
        self.publish_status(
            mode=mode,
            status_text='USING_TRAJECTORY_PROXY',
            success=True,
            reason='Backend keyframes unavailable; using sparse convex hull of trajectory proxy.',
            raw_points=len(raw_points),
            downsampled_points=len(points),
            hull_vertices=len(hull),
            waypoints=len(waypoints.poses),
            region_id=region_id,
        )
        self.publish_status(
            mode=mode,
            status_text='DONE',
            success=True,
            reason='Sparse PGS waypoint sequence published.',
            raw_points=len(raw_points),
            downsampled_points=len(points),
            hull_vertices=len(hull),
            waypoints=len(waypoints.poses),
            region_id=region_id,
        )
        self.get_logger().info(
            '[RALC] Using trajectory proxy for PGS because SLAM keyframes are '
            f'unavailable. mode={mode}, hull_vertices={len(hull)}, '
            f'waypoints={len(waypoints.poses)}'
        )

    def publish_status(
        self,
        mode: str,
        status_text: str,
        success: bool,
        reason: str,
        raw_points: int,
        downsampled_points: int,
        hull_vertices: int,
        waypoints: int,
        region_id=None,
    ):
        msg = String()
        msg.data = json.dumps({
            'mode': mode,
            'status': status_text,
            'region_id': region_id,
            'success': success,
            'reason': reason,
            'message': reason,
            'raw_points': raw_points,
            'downsampled_points': downsampled_points,
            'hull_vertices': hull_vertices,
            'waypoint_count': waypoints,
            'waypoints': waypoints,
        })
        self.status_pub.publish(msg)

    def proxy_points_for_mode(self, mode: str):
        if mode == 'global' or self.current_region is None:
            return list(self.trajectory)
        region = self.current_region
        return [
            point for point in self.trajectory
            if (
                float(region['xmin']) <= point[0] <= float(region['xmax']) and
                float(region['ymin']) <= point[1] <= float(region['ymax'])
            )
        ]

    def downsample_points(self, points, min_spacing):
        downsampled = []
        seen = set()
        for point in points:
            key = (round(point[0], 2), round(point[1], 2))
            if key in seen:
                continue
            if downsampled and math.hypot(
                point[0] - downsampled[-1][0],
                point[1] - downsampled[-1][1],
            ) < min_spacing:
                continue
            seen.add(key)
            downsampled.append(point)
        return downsampled

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
        sequence = list(hull) + list(reversed(hull))
        robot_xy = self.lookup_robot_xy()
        previous = None
        for x, y in sequence:
            if robot_xy is not None and math.hypot(x - robot_xy[0], y - robot_xy[1]) < self.min_waypoint_spacing:
                continue
            if previous is not None and math.hypot(x - previous[0], y - previous[1]) < self.min_waypoint_spacing:
                continue
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            yaw = 0.0
            if previous is not None:
                yaw = math.atan2(y - previous[1], x - previous[0])
            pose.orientation.z = math.sin(0.5 * yaw)
            pose.orientation.w = math.cos(0.5 * yaw)
            msg.poses.append(pose)
            previous = (x, y)
        return msg

    def publish_markers(self, hull, frame_id):
        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)
        if hull:
            line = Marker()
            line.header.stamp = self.get_clock().now().to_msg()
            line.header.frame_id = frame_id
            line.ns = 'ralc_pgs_hull'
            line.id = 1
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
