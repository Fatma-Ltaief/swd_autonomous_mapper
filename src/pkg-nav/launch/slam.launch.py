from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    params_file = os.path.join(
        get_package_share_directory('pkg-nav'),  # Remplacez par votre nom de package
        'config',
        'slam_params.yaml'
    )

    return LaunchDescription([
        # Static transform: base_footprint -> base_link
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
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[params_file],  # Charge le fichier YAML
            # remappings=[
            #     ('scan', '/base_scan'),
            #     ('odom', '/odom_combined'),
            #     # ajoute ici d’autres remaps si besoin
            # ]
        ),

        # Node(
        #     package='pkg-nav',
        #     executable='sim_clock',
        #     name='sim_clock',
        #     output='screen',
        #     parameters=[params_file],  # Charge le fichier YAML
        #     # remappings=[
        #     #     ('scan', '/base_scan'),
        #     #     ('odom', '/odom_combined'),
        #     #     # ajoute ici d’autres remaps si besoin
        #     # ]
        # )
    ])
