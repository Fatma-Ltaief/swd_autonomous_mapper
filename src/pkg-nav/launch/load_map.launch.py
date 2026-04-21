from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction
)
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Argument pour spécifier le chemin vers la carte
        DeclareLaunchArgument(
            'map',
            default_value='/home/fatma/ws-ros2/src/rem_mapping/scripts/test_mimo2.yaml',
            description='Chemin vers le fichier de carte YAML'
        ),
        
        # Noeud map_server avec gestion du cycle de vie
        LifecycleNode(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': LaunchConfiguration('map')}],
            namespace='',
        ),

        # Temporisation avant de configurer le map_server
        TimerAction(
            period=1.0,  # Temps en secondes avant la configuration
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'lifecycle', 'set', '/map_server', 'configure'],
                    output='screen'
                )
            ]
        ),

        # Temporisation avant d'activer le map_server
        TimerAction(
            period=2.0,  # Temps en secondes avant l'activation
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'lifecycle', 'set', '/map_server', 'activate'],
                    output='screen'
                )
            ]
        ),
    ])
