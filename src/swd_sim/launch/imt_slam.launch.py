import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    slam_toolbox_share = get_package_share_directory('slam_toolbox')
    swd_sim_share = get_package_share_directory('swd_sim')

    online_async_launch = os.path.join(
        slam_toolbox_share,
        'launch',
        'online_async_launch.py'
    )
    slam_params_file = os.path.join(
        swd_sim_share,
        'config',
        'swd_slam_toolbox.yaml'
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(online_async_launch),
            launch_arguments={
                'use_sim_time': 'true',
                'slam_params_file': slam_params_file,
            }.items()
        ),
    ])
