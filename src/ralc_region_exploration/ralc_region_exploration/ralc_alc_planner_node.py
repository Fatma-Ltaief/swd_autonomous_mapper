import json

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray


class RalcAlcPlanner(Node):
    """ALC planner interface.

    The paper's ALC requires pose graph covariance / uncertainty estimates.
    SLAM Toolbox does not expose that through the topics currently available in
    this baseline, so this node explicitly reports ALC as unavailable instead
    of replacing it with a spin or goal-count heuristic.
    """

    def __init__(self):
        super().__init__('ralc_alc_planner')

        self.current_region = None
        self.create_subscription(String, '/ralc/current_region', self.region_callback, 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/ralc/alc_goal', 10)
        self.unavailable_pub = self.create_publisher(Bool, '/ralc/alc_unavailable', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/ralc/alc_markers', 10)
        self.timer = self.create_timer(1.0, self.tick)
        self.get_logger().info('[RALC] ALC planner ready; waiting for covariance backend.')

    def region_callback(self, msg: String):
        try:
            region = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.current_region = region if region else None

    def tick(self):
        unavailable = Bool()
        unavailable.data = True
        self.unavailable_pub.publish(unavailable)
        self.publish_empty_marker()
        if self.current_region and self.current_region.get('status') == 'ACTIVE':
            self.get_logger().debug(
                '[RALC] ALC unavailable: exact paper ALC cannot be computed '
                'without pose graph covariance from the SLAM backend.'
            )

    def publish_empty_marker(self):
        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = RalcAlcPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
