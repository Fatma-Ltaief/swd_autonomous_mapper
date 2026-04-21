from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # AMCL node
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[
            {'use_sim_time': False},
            # adapt these if your topics differ
            {'scan_topic': 'scan'},
        ]
    )

    # Lifecycle manager ONLY for AMCL
    lifecycle_mgr_localization = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[
            {'use_sim_time': False},
            {'autostart': True},
            {'node_names': ['amcl']},
        ]
    )

    return LaunchDescription([
        amcl,
        lifecycle_mgr_localization
    ])

