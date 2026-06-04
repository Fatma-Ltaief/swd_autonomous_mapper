from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_topic = LaunchConfiguration('map_topic')
    odom_topic = LaunchConfiguration('odom_topic')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('map_topic', default_value='/map'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),

        LogInfo(
            msg=[
                'room_active_slam: analyzing free-space regions from ',
                map_topic,
                ' and publishing debug markers + next_goal only.',
            ]
        ),

        Node(
            package='room_active_slam',
            executable='room_map_analyzer_node',
            name='room_map_analyzer',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'map_topic': map_topic,
                'odom_topic': odom_topic,
                'map_frame': 'map',
                'base_frame': 'base_footprint',
                'fallback_base_frame': 'base_link',
                'min_region_area': 30,
                'morphology_kernel_size': 3,
                'analysis_period_sec': 1.0,
            }],
        ),
    ])
