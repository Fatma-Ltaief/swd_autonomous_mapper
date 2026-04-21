from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    declare_params = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            get_package_share_directory('my_robot_nav'),
            'config', 'nav2_params.yaml'
        ),
        description='Full path to the Nav2 params YAML'
    )

    # ---- Static TFs (since we have no URDF) ----
    # 1) base_link -> laser_frame (edit xyz/rpy to your robot’s lidar mount)
    static_base_to_laser = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='static_base_to_laser',
        arguments=['0.20', '0.0', '0.25',  '0', '0', '0',  'base_link', 'laser_frame']
        #                 x     y     z     R   P   Y       parent      child
    )

    # If your odom is produced by your motor driver/ekf, keep this out.
    # If you have no odom source at all, you can TEMPORARILY fake odom->base_link
    # (robot won’t localize well). Prefer real wheel odom/IMU via robot_localization.
    # static_odom_to_base = Node(
    #     package='tf2_ros', executable='static_transform_publisher',
    #     name='static_odom_to_base',
    #     arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link']
    # )

    # ---- SLAM: online mapping (you can also run it separately as you do now) ----
    slam = Node(
        package='slam_toolbox',
        executable='sync_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
        # For custom slam params, pass a YAML here.
    )

    # ---- Nav2 Bringup ----
    nav2 = Node(
        package='nav2_bringup',
        executable='bringup_launch.py',
        output='screen',
        arguments=['slam:=False',  # we already launched slam_toolbox above
                   f'params_file:={ParameterValue(params_file, value_type=str)}',
                   'use_sim_time:=false']
    )

    # Optional: RViz2 with Nav2 panel
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2'
    )

    return LaunchDescription([
        declare_params,
        static_base_to_laser,
        # static_odom_to_base,
        slam,
        nav2,
        rviz
    ])

