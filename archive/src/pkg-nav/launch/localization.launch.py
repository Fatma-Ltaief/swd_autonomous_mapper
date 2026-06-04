import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction
)
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    params_file = os.path.join(
        get_package_share_directory('pkg-nav'),  # Remplacez par votre nom de package
        'config',
        'amcl_params.yaml'
    )
    
    return LaunchDescription([
        # Argument pour spécifier le chemin vers la carte
        DeclareLaunchArgument(
            'map',
            default_value='/home/fatma/ws-ros2/src/rem_mapping/scripts/test_mimo2.yaml',
            description='Chemin vers le fichier de carte YAML'
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_base_footprint_to_base_link',
            arguments=[
                '0', '0', '0',    # translation
                '0', '0', '0',    # rotation (RPY)
                'base_link',
                'base_footprint'
            ]
        ),

        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'use_sim_time': True}, 
                        {'yaml_filename': LaunchConfiguration('map')}]
        ),
            
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[params_file],
        ),

        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{'use_sim_time': True},
                        {'autostart': True},
                        {'node_names': ['map_server', 'amcl']}]
        )
    ])
