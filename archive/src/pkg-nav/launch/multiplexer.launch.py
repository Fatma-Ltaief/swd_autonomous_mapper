import os

from ament_index_python.packages import get_package_share_directory

import launch
import launch_ros.actions
from launch_ros.actions import Node


def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('pkg-nav'),
        'config',
        'multiplexer.yaml'
    )

    return LaunchDescription([
        Node(
            package='pkg-nav',
            executable='velocity_multiplexer',
            name='multiplexer',
            parameters=[config]
        )
    ])