from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    z_pose = LaunchConfiguration('z_pose')
    world = LaunchConfiguration('world')

    # Prefer the description package installed beside this swd_sim package.
    # This avoids accidentally spawning a same-named URDF from another overlay.
    swd_sim_prefix = Path(get_package_prefix('swd_sim'))
    same_workspace_urdf = (
        swd_sim_prefix.parent
        / 'swd_starter_kit_description'
        / 'share'
        / 'swd_starter_kit_description'
        / 'urdf'
        / 'swd_starter_kit.urdf'
    )
    if same_workspace_urdf.exists():
        urdf_path = str(same_workspace_urdf)
    else:
        urdf_path = str(
            Path(get_package_share_directory('swd_starter_kit_description'))
            / 'urdf'
            / 'swd_starter_kit.urdf'
        )

    gazebo_launch_file = str(
        Path(get_package_share_directory('gazebo_ros'))
        / 'launch'
        / 'gazebo.launch.py'
    )

    # robot_description must be forced as a string
    robot_description = ParameterValue(
        Command(['cat', ' ', urdf_path]),
        value_type=str
    )

    # Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={
            'world': world,
            'verbose': 'true',
        }.items()
    )

    # Publish robot_description + TF
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description,
        }]
    )

    # Spawn robot into Gazebo using /robot_description
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_swd_robot',
        output='screen',
        arguments=[
            '-entity', 'swd_starter_kit',
            '-topic', 'robot_description',
            '-x', x_pose,
            '-y', y_pose,
            '-z', z_pose,
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock'
        ),
        DeclareLaunchArgument(
            'x_pose',
            default_value='0.0',
            description='Initial robot x position'
        ),
        DeclareLaunchArgument(
            'y_pose',
            default_value='0.0',
            description='Initial robot y position'
        ),
        DeclareLaunchArgument(
            'z_pose',
            default_value='0.05',
            description='Initial robot z position'
        ),
        DeclareLaunchArgument(
            'world',
            default_value='',
            description='Full path to a Gazebo world file. Leave empty to use Gazebo default world.'
        ),

        gazebo,
        robot_state_publisher,
        spawn_robot,
    ])
