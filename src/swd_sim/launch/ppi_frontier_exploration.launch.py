from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


# PPI office initial robot pose.
# Adjust these values if the robot starts inside geometry or outside the open area.
SPAWN_X = '0.0'
SPAWN_Y = '0.0'
SPAWN_Z = '0.3'
SPAWN_ROLL = '0.0'
SPAWN_PITCH = '0.0'
SPAWN_YAW = '0.0'


def generate_launch_description():
    swd_sim_share = Path(get_package_share_directory('swd_sim'))
    swd_nav2_share = Path(get_package_share_directory('swd_nav2'))
    nav2_bringup_share = Path(get_package_share_directory('nav2_bringup'))

    # PPI office Gazebo Classic world. This launch intentionally starts the
    # same frontier SLAM pipeline as the warehouse launch, but in ppi_office.world.
    world_path = str(swd_sim_share / 'worlds' / 'ppi_office.world')
    robot_launch = swd_sim_share / 'launch' / 'gazebo_robot.launch.py'
    slam_launch = swd_sim_share / 'launch' / 'imt_slam.launch.py'
    nav2_launch = nav2_bringup_share / 'launch' / 'navigation_launch.py'
    nav2_params = swd_nav2_share / 'config' / 'nav2_params.yaml'

    # Stage 1: start exactly one Gazebo instance for the PPI office, publish
    # robot_description, and spawn the SWD robot at the editable pose above.
    robot_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(robot_launch)),
        launch_arguments={
            'use_sim_time': 'true',
            'world': world_path,
            'x_pose': SPAWN_X,
            'y_pose': SPAWN_Y,
            'z_pose': SPAWN_Z,
            'roll': SPAWN_ROLL,
            'pitch': SPAWN_PITCH,
            'yaw': SPAWN_YAW,
        }.items(),
    )

    # Stage 2: start SLAM after Gazebo has spawned the robot and sensor data
    # is expected to be available.
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(slam_launch)),
    )

    # Stage 3: start Nav2 after SLAM begins publishing map data and the map frame.
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
            'Starting one Gazebo PPI office simulation with frontier ',
            'exploration. Required topics: /map, /scan, /odom, /tf, /cmd_vel',
        ]),
        robot_simulation,
        TimerAction(period=5.0, actions=[slam_toolbox]),
        TimerAction(period=10.0, actions=[nav2]),
        TimerAction(period=18.0, actions=[frontier_explorer]),
    ])
