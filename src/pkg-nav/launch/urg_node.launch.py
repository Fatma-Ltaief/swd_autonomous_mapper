from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction
)
from launch.substitutions import LaunchConfiguration

import os
import launch
import yaml
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration
from launch.actions import (DeclareLaunchArgument, EmitEvent, RegisterEventHandler)
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition

def generate_launch_description():

    config_file_path = os.path.join(
        get_package_share_directory('urg_node2'),
        'config',
        # 'params_ether.yaml' ethernet
        'params_serial.yaml'
    )

    # パラメータファイルのロード
    with open(config_file_path, 'r') as file:
        config_params = yaml.safe_load(file)['urg_node2']['ros__parameters']

    return LaunchDescription([        
        # Noeud map_server avec gestion du cycle de vie
        LifecycleNode(
            package='urg_node2',
            executable='urg_node2_node',
            name="urg_node2",
            parameters=[config_params],
            namespace='',
            output='screen',
        ),

        # Temporisation avant de configurer le map_server
        TimerAction(
            period=2.0,  # Temps en secondes avant la configuration
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'lifecycle', 'set', '/urg_node2', 'configure'],
                    output='screen'
                )
            ]
        ),

        # Temporisation avant d'activer le map_server
        TimerAction(
            period=4.0,  # Temps en secondes avant l'activation
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'lifecycle', 'set', '/urg_node2', 'activate'],
                    output='screen'
                )
            ]
        ),
    ])
