import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _swd_urdf_path():
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
        return str(same_workspace_urdf)

    return str(
        Path(get_package_share_directory('swd_starter_kit_description'))
        / 'urdf'
        / 'swd_starter_kit.urdf'
    )


def generate_launch_description():
    swd_sim_share = Path(get_package_share_directory('swd_sim'))
    world_path = str(swd_sim_share / 'worlds' / 'simple_colored_warehouse.sdf')
    print(f'Warehouse world path: {world_path}')
    models_path = str(swd_sim_share / 'models')
    gazebo_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    gazebo_model_path = (
        models_path
        if not gazebo_model_path
        else models_path + os.pathsep + gazebo_model_path
    )

    gazebo_launch_file = str(
        Path(get_package_share_directory('gazebo_ros'))
        / 'launch'
        / 'gazebo.launch.py'
    )

    robot_description = ParameterValue(
        Command(['cat', ' ', _swd_urdf_path()]),
        value_type=str,
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={
            'world': world_path,
            'verbose': 'true',
        }.items(),
    )

    gazebo_log = LogInfo(msg=['Launching Gazebo with world: ', world_path])

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': robot_description,
        }],
    )

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_swd_robot',
        output='screen',
        arguments=[
            '-entity', 'swd_starter_kit',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '-6.0',
            '-z', '0.05',
            '-R', '0.0',
            '-P', '0.0',
            '-Y', '0.0',
        ],
    )

    return LaunchDescription([
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path),
        gazebo,
        gazebo_log,
        robot_state_publisher,
        spawn_robot,
    ])
