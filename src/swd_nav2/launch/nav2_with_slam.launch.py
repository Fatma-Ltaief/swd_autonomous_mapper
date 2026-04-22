from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    swd_nav2_dir = get_package_share_directory('swd_nav2')

    nav2_params = os.path.join(swd_nav2_dir, 'config', 'nav2_params.yaml')

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            # real robot → no sim time
            'use_sim_time': 'false',
            # our custom Nav2 params
            'params_file': nav2_params,
            # tell bringup to use SLAM instead of localization+map_server
            'slam': 'True',
            # still required as a launch argument, but not used when slam=True
            'map': ''
        }.items()
    )

    return LaunchDescription([
        nav2_launch
    ])
