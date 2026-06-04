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
        # Noeud map_server avec gestion du cycle de vie
        LifecycleNode(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            namespace='',
        ),

        # Temporisation avant de configurer le map_server
        TimerAction(
            period=1.0,  # Temps en secondes avant la configuration
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'lifecycle', 'set', '/amcl', 'configure'],
                    output='screen'
                )
            ]
        ),

        # Temporisation avant d'activer le map_server
        TimerAction(
            period=2.0,  # Temps en secondes avant l'activation
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'lifecycle', 'set', '/amcl', 'activate'],
                    output='screen'
                )
            ]
        ),
    ])
