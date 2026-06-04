from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class FreeSpaceRegion:
    """A simple geometric summary of one connected free-space component."""

    region_id: int
    cells: np.ndarray
    area_cells: int
    centroid_cell: Tuple[float, float]
    bbox_cells: Tuple[int, int, int, int]
    centroid_world: Tuple[float, float]


class RoomMapAnalyzer(Node):
    """First prototype room/region analyzer for active SLAM.

    This node intentionally does not command Nav2 yet. It only looks at the
    latest SLAM occupancy grid, groups connected free cells into regions, marks
    them in RViz, and publishes a candidate next goal for later nodes to use.
    """

    def __init__(self):
        super().__init__('room_map_analyzer')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('fallback_base_frame', 'base_link')
        self.declare_parameter('min_region_area', 30)
        self.declare_parameter('morphology_kernel_size', 3)
        self.declare_parameter('analysis_period_sec', 1.0)

        self.map_topic = self.get_parameter('map_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.fallback_base_frame = (
            self.get_parameter('fallback_base_frame').value
        )
        self.min_region_area = int(
            self.get_parameter('min_region_area').value
        )
        self.morphology_kernel_size = int(
            self.get_parameter('morphology_kernel_size').value
        )
        analysis_period_sec = float(
            self.get_parameter('analysis_period_sec').value
        )

        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_odom: Optional[Odometry] = None
        self.last_region_count = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/room_active_slam/debug_markers',
            10,
        )
        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/room_active_slam/next_goal',
            10,
        )

        self.analysis_timer = self.create_timer(
            analysis_period_sec,
            self.analyze_latest_map,
        )

        self.get_logger().info(
            f'Room map analyzer started. map_topic={self.map_topic}, '
            f'odom_topic={self.odom_topic}, min_region_area='
            f'{self.min_region_area}, morphology_kernel_size='
            f'{self.morphology_kernel_size}'
        )
        self.get_logger().info(
            'Publishing markers on /room_active_slam/debug_markers and '
            'prototype goals on /room_active_slam/next_goal.'
        )

    def map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def occupancy_grid_to_image(self, grid: OccupancyGrid) -> np.ndarray:
        """Convert OccupancyGrid data into a compact uint8 image.

        OccupancyGrid values follow the common convention:
        - -1 means unknown
        - 0 means free
        - values above 50 are treated as occupied

        For image processing we use:
        - free cells: 255
        - occupied cells: 0
        - unknown cells: 127
        """
        width = grid.info.width
        height = grid.info.height
        occupancy = np.array(
            grid.data,
            dtype=np.int16,
        ).reshape((height, width))

        image = np.full((height, width), 127, dtype=np.uint8)
        image[occupancy == 0] = 255
        image[occupancy > 50] = 0
        return image

    def create_free_space_mask(self, image: np.ndarray) -> np.ndarray:
        """Build and clean a binary free-space mask for OpenCV.

        OpenCV expects uint8 binary images for connected component analysis.
        We keep only exact free cells and remove small pixel noise with
        morphology. Opening removes isolated specks; closing reconnects tiny
        cracks that can appear in SLAM maps.
        """
        free_mask = np.zeros_like(image, dtype=np.uint8)
        free_mask[image == 255] = 255

        kernel_size = max(1, self.morphology_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        if kernel_size <= 1:
            return free_mask

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        opened = cv2.morphologyEx(free_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return cleaned

    def find_free_space_regions(
        self,
        free_mask: np.ndarray,
        grid: OccupancyGrid,
    ) -> List[FreeSpaceRegion]:
        """Find connected free-space regions with OpenCV.

        cv2.connectedComponentsWithStats gives us the region label image,
        per-component area, bounding boxes, and centroids in one robust pass.
        Label 0 is the background, so real free-space regions start at 1.
        """
        (
            label_count,
            labels,
            stats,
            centroids,
        ) = cv2.connectedComponentsWithStats(
            free_mask,
            connectivity=4,
            ltype=cv2.CV_32S,
        )
        regions: List[FreeSpaceRegion] = []
        next_region_id = 1

        for label in range(1, label_count):
            area_cells = int(stats[label, cv2.CC_STAT_AREA])
            if area_cells < self.min_region_area:
                continue

            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            centroid_x = float(centroids[label][0])
            centroid_y = float(centroids[label][1])
            centroid_world = self.map_cell_to_world(
                grid,
                centroid_x,
                centroid_y,
            )

            # Store the member cells for now so robot-region lookup remains
            # simple and exact. Later room logic can switch to using the label
            # image directly if this becomes too heavy for large maps.
            ys, xs = np.where(labels == label)
            cell_array = np.column_stack((xs, ys)).astype(np.int32)

            regions.append(
                FreeSpaceRegion(
                    region_id=next_region_id,
                    cells=cell_array,
                    area_cells=area_cells,
                    centroid_cell=(centroid_x, centroid_y),
                    bbox_cells=(
                        left,
                        top,
                        left + width - 1,
                        top + height - 1,
                    ),
                    centroid_world=centroid_world,
                )
            )
            next_region_id += 1

        return regions

    def get_robot_pose_in_map(self) -> Optional[Tuple[float, float]]:
        """Prefer TF map->base_footprint, then base_link, then /odom."""
        for source_frame in (self.base_frame, self.fallback_base_frame):
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.05),
                )
                translation = transform.transform.translation
                return float(translation.x), float(translation.y)
            except TransformException:
                continue

        if self.latest_odom is None:
            return None

        pose = self.latest_odom.pose.pose.position
        return float(pose.x), float(pose.y)

    def world_to_map_cell(
        self,
        grid: OccupancyGrid,
        world_x: float,
        world_y: float,
    ) -> Tuple[int, int]:
        resolution = grid.info.resolution
        origin = grid.info.origin.position
        mx = int((world_x - origin.x) / resolution)
        my = int((world_y - origin.y) / resolution)
        return mx, my

    def map_cell_to_world(
        self,
        grid: OccupancyGrid,
        cell_x: float,
        cell_y: float,
    ) -> Tuple[float, float]:
        resolution = grid.info.resolution
        origin = grid.info.origin.position
        # Add 0.5 so the marker/goal lands in the center of the map cell.
        world_x = origin.x + (cell_x + 0.5) * resolution
        world_y = origin.y + (cell_y + 0.5) * resolution
        return float(world_x), float(world_y)

    def find_robot_region(
        self,
        regions: List[FreeSpaceRegion],
        grid: OccupancyGrid,
        robot_xy: Optional[Tuple[float, float]],
    ) -> Optional[FreeSpaceRegion]:
        if robot_xy is None:
            return None

        robot_mx, robot_my = self.world_to_map_cell(
            grid,
            robot_xy[0],
            robot_xy[1],
        )

        for region in regions:
            min_x, min_y, max_x, max_y = region.bbox_cells
            if not (min_x <= robot_mx <= max_x and min_y <= robot_my <= max_y):
                continue

            # The bounding box is a cheap first check. This exact cell test
            # confirms the robot is actually inside the component.
            region_cells = region.cells
            matches = (
                (region_cells[:, 0] == robot_mx) &
                (region_cells[:, 1] == robot_my)
            )
            if bool(matches.any()):
                return region

        return None

    def analyze_latest_map(self):
        if self.latest_map is None:
            return

        grid = self.latest_map
        image = self.occupancy_grid_to_image(grid)
        free_mask = self.create_free_space_mask(image)
        regions = self.find_free_space_regions(free_mask, grid)
        robot_xy = self.get_robot_pose_in_map()
        robot_region = self.find_robot_region(regions, grid, robot_xy)

        self.publish_debug_markers(grid, regions, robot_region)

        if robot_region is not None:
            self.publish_next_goal(grid, robot_region)

        if len(regions) != self.last_region_count:
            self.get_logger().info(
                f'Detected {len(regions)} free-space regions; '
                f'robot_region='
                f'{robot_region.region_id if robot_region else "unknown"}'
            )
            self.last_region_count = len(regions)

    def publish_next_goal(
        self,
        grid: OccupancyGrid,
        region: FreeSpaceRegion,
    ):
        """Publish the current region centroid as a prototype goal only."""
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = grid.header.frame_id or self.map_frame
        goal.pose.position.x = region.centroid_world[0]
        goal.pose.position.y = region.centroid_world[1]
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)

    def publish_debug_markers(
        self,
        grid: OccupancyGrid,
        regions: List[FreeSpaceRegion],
        robot_region: Optional[FreeSpaceRegion],
    ):
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        frame_id = grid.header.frame_id or self.map_frame

        # Clear stale markers when the number of regions shrinks.
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)

        for region in regions:
            is_current = (
                robot_region is not None and
                region.region_id == robot_region.region_id
            )
            x, y = region.centroid_world

            centroid_marker = Marker()
            centroid_marker.header.stamp = stamp
            centroid_marker.header.frame_id = frame_id
            centroid_marker.ns = 'region_centroids'
            centroid_marker.id = region.region_id
            centroid_marker.type = Marker.SPHERE
            centroid_marker.action = Marker.ADD
            centroid_marker.pose.position.x = x
            centroid_marker.pose.position.y = y
            centroid_marker.pose.position.z = 0.15
            centroid_marker.pose.orientation.w = 1.0
            centroid_marker.scale.x = 0.25 if is_current else 0.16
            centroid_marker.scale.y = 0.25 if is_current else 0.16
            centroid_marker.scale.z = 0.25 if is_current else 0.16
            centroid_marker.color.r = 1.0 if is_current else 0.1
            centroid_marker.color.g = 0.2 if is_current else 0.7
            centroid_marker.color.b = 0.1 if is_current else 1.0
            centroid_marker.color.a = 0.9
            markers.markers.append(centroid_marker)

            text_marker = Marker()
            text_marker.header.stamp = stamp
            text_marker.header.frame_id = frame_id
            text_marker.ns = 'region_labels'
            text_marker.id = region.region_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.pose.position.z = 0.45
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.25
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = (
                f'R{region.region_id}\\n'
                f'{region.area_cells} cells'
            )
            markers.markers.append(text_marker)

        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = RoomMapAnalyzer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
