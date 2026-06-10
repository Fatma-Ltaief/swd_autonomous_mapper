from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


# PPI office initial robot pose. These match the tuned PPI frontier launch.
SPAWN_X = '7.839656829833984'
SPAWN_Y = '-2.6261680126190186'
SPAWN_Z = '0.23'
SPAWN_ROLL = '0.0'
SPAWN_PITCH = '0.0'
SPAWN_YAW = '0'#3.14159


def generate_launch_description():
    use_sim_time = True
    swd_sim_share = Path(get_package_share_directory('swd_sim'))
    swd_nav2_share = Path(get_package_share_directory('swd_nav2'))
    ralc_share = Path(get_package_share_directory('ralc_region_exploration'))

    world_path = str(swd_sim_share / 'worlds' / 'ppi_office.world')
    robot_launch = swd_sim_share / 'launch' / 'gazebo_robot.launch.py'
    slam_launch = swd_sim_share / 'launch' / 'imt_slam.launch.py'
    ralc_launch = ralc_share / 'launch' / 'ralc_region_debug.launch.py'

    # PPI office simulation uses the PPI-specific Nav2 params with the larger
    # fixed global costmap. This launch starts Nav2 but intentionally does not
    # start nav2_wfd, so R-ALC is the only exploration decision source.
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

    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(slam_launch)),
    )

    scan_min_range_filter = Node(
        package='ralc_region_exploration',
        executable='scan_min_range_filter_node',
        name='scan_min_range_filter',
        output='screen',
        emulate_tty=True,
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'use_sim_time': use_sim_time,
            'input_scan_topic': '/scan',
            'output_scan_topic': '/scan_filtered',
            'min_valid_range': 0.25,
        }],
    )

    nav2_nodes = [
        LogInfo(msg=[
            'Stage 3/4: Nav2 starting for R-ALC with lifecycle nodes: ',
            ', '.join(nav2_lifecycle_nodes),
        ]),
        LogInfo(msg=['Nav2 params file: ', str(nav2_params)]),
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

    ralc_debug = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(ralc_launch)),
        launch_arguments={
            'use_sim_time': 'true',
            'enable_motion': 'true',
        }.items(),
    )

    return LaunchDescription([
        LogInfo(msg=[
            'PPI R-ALC launch order: Gazebo -> robot -> scan filter -> '
            'SLAM -> Nav2 -> R-ALC',
        ]),
        LogInfo(msg='Stage 1/4: Gazebo + robot starting for PPI office'),
        robot_simulation,
        LogInfo(msg='Stage 1b/4: /scan -> /scan_filtered filter starting'),
        scan_min_range_filter,
        TimerAction(period=5.0, actions=[
            LogInfo(msg='Stage 2/4: SLAM starting'),
            slam_toolbox,
        ]),
        TimerAction(period=10.0, actions=nav2_nodes),
        TimerAction(period=30.0, actions=[
            LogInfo(msg=[
                'Stage 4/4: R-ALC starting after 20 s Nav2 delay. ',
                'Expected action server: /navigate_to_pose',
            ]),
            ralc_debug,
        ]),
    ])
