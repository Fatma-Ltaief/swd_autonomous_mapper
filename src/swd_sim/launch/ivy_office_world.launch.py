import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    swd_sim_share = Path(get_package_share_directory('swd_sim'))
    world_path = str(swd_sim_share / 'worlds' / 'ivy_office.world')
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

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={
            'world': world_path,
            'verbose': 'true',
        }.items()
    )

    return LaunchDescription([
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path),
        gazebo,
    ])
