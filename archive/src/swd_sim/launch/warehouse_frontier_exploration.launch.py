from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    swd_sim_share = Path(get_package_share_directory('swd_sim'))
    swd_nav2_share = Path(get_package_share_directory('swd_nav2'))
    nav2_bringup_share = Path(get_package_share_directory('nav2_bringup'))

    world_path = str(swd_sim_share / 'worlds' / 'simple_colored_warehouse.sdf')
    robot_launch = swd_sim_share / 'launch' / 'gazebo_robot.launch.py'
    slam_launch = swd_sim_share / 'launch' / 'imt_slam.launch.py'
    nav2_launch = nav2_bringup_share / 'launch' / 'navigation_launch.py'
    nav2_params = swd_nav2_share / 'config' / 'nav2_params.yaml'

    # Stage 1: start exactly one Gazebo instance, publish robot_description,
    # and spawn the robot into the warehouse world.
    robot_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(robot_launch)),
        launch_arguments={
            'use_sim_time': 'true',
            'world': world_path,
            'x_pose': '0.0',
            'y_pose': '-6.0',
            'z_pose': '0.05',
        }.items(),
    )

    # Stage 2: start SLAM after Gazebo has had time to spawn the robot and
    # begin publishing /scan, /odom, and /tf.
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(slam_launch)),
    )

    # Stage 3: start Nav2 after SLAM starts publishing the map frame and /map.
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(nav2_launch)),
        launch_arguments={
            'use_sim_time': 'true',
            'params_file': str(nav2_params),
        }.items(),
    )

    # Stage 4: start frontier exploration last, once Nav2 action servers and
    # costmaps are expected to be available.
    frontier_explorer = Node(
        package='nav2_wfd',
        executable='explore',
        name='wavefront_frontier_explorer',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'min_frontier_size': 25,
            'obstacle_padding_cells': 3,
            'planner_frequency': 0.25,
            'progress_timeout': 60.0,
            'goal_aliasing_distance': 0.6,
            'unreachable_goal_radius': 0.8,
            'planner_timeout': 5.0,
            'max_planner_candidates': 8,
            'max_goal_cost': 252,
            'stop_publish_count': 20,
            'stop_publish_period': 0.1,
        }],
    )

    return LaunchDescription([
        LogInfo(msg=[
            'Starting one Gazebo warehouse simulation with frontier ',
            'exploration. Required topics: /map, /scan, /odom, /tf, /cmd_vel',
        ]),
        robot_simulation,
        TimerAction(period=5.0, actions=[slam_toolbox]),
        TimerAction(period=10.0, actions=[nav2]),
        TimerAction(period=18.0, actions=[frontier_explorer]),
    ])
