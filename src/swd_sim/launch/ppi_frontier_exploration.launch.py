from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


# PPI office initial robot pose.
# Adjust these values if the robot starts inside geometry or outside the open area.
SPAWN_X = '-1.10'
SPAWN_Y = '-1.67'
SPAWN_Z = '0.3'
SPAWN_ROLL = '0.0'
SPAWN_PITCH = '0.0'
SPAWN_YAW = '60.0'


def generate_launch_description():
    swd_sim_share = Path(get_package_share_directory('swd_sim'))
    swd_nav2_share = Path(get_package_share_directory('swd_nav2'))

    # PPI office Gazebo Classic world. This launch intentionally starts the
    # same frontier SLAM pipeline as the warehouse launch, but in ppi_office.world.
    world_path = str(swd_sim_share / 'worlds' / 'ppi_office.world')
    robot_launch = swd_sim_share / 'launch' / 'gazebo_robot.launch.py'
    slam_launch = swd_sim_share / 'launch' / 'imt_slam.launch.py'
    # PPI office simulation uses a larger fixed global costmap than the shared
    # default so frontier goals remain inside Nav2 planning bounds.
    nav2_params = swd_nav2_share / 'config' / 'nav2_params_ppi.yaml'
    nav2_lifecycle_nodes = [
        'controller_server',
        'planner_server',
        'smoother_server',
        'behavior_server',
        'bt_navigator',
        'waypoint_follower',
    ]
    nav2_remappings = [
        ('/tf', 'tf'),
        ('/tf_static', 'tf_static'),
    ]

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

    # Stage 3: start the PPI Nav2 lifecycle stack explicitly. This keeps the
    # required servers visible in the launch file and lets lifecycle_manager
    # configure/activate all navigation nodes before frontier exploration starts.
    nav2_nodes = [
        LogInfo(msg=[
            'Stage 3/4: Nav2 starting with lifecycle nodes: ',
            ', '.join(nav2_lifecycle_nodes),
        ]),
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[str(nav2_params)],
            remappings=nav2_remappings,
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[str(nav2_params)],
            remappings=nav2_remappings,
        ),
        Node(
            package='nav2_smoother',
            executable='smoother_server',
            name='smoother_server',
            output='screen',
            parameters=[str(nav2_params)],
            remappings=nav2_remappings,
        ),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[str(nav2_params)],
            remappings=nav2_remappings,
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[str(nav2_params)],
            remappings=nav2_remappings,
        ),
        Node(
            package='nav2_waypoint_follower',
            executable='waypoint_follower',
            name='waypoint_follower',
            output='screen',
            parameters=[str(nav2_params)],
            remappings=nav2_remappings,
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'autostart': True,
                'node_names': nav2_lifecycle_nodes,
            }],
        ),
    ]

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
            'safety_margin': 0.5,
            'frontier_blacklist_duration': 45.0,
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
            'PPI launch order: Gazebo -> robot -> SLAM -> Nav2 -> frontier',
        ]),
        LogInfo(msg='Stage 1/4: Gazebo + robot starting for PPI office'),
        robot_simulation,
        TimerAction(period=5.0, actions=[
            LogInfo(msg='Stage 2/4: SLAM starting'),
            slam_toolbox,
        ]),
        TimerAction(period=10.0, actions=nav2_nodes),
        TimerAction(period=30.0, actions=[
            LogInfo(msg=[
                'Stage 4/4: frontier starting after 20 s Nav2 delay. ',
                'Expected active nodes: controller_server, planner_server, ',
                'smoother_server, behavior_server, bt_navigator, ',
                'waypoint_follower',
            ]),
            frontier_explorer,
        ]),
    ])
