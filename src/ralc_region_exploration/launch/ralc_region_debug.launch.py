from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_motion = LaunchConfiguration('enable_motion')

    common_frames = {
        'map_frame': 'map',
        'base_frame': 'base_footprint',
        'fallback_base_frame': 'base_link',
    }

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('enable_motion', default_value='true'),

        LogInfo(
            msg=(
                'R-ALC paper-faithful baseline: RegionManager + ALCPlanner + '
                'FrontierPlanner + PGSPlanner + Executive + ExplorationManager.'
            )
        ),

        Node(
            package='ralc_region_exploration',
            executable='region_manager_node',
            name='ralc_region_manager',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'region_min_width': 4.0,
                'region_min_height': 4.0,
                'region_max_width': 10.0,
                'region_max_height': 10.0,
                'robot_neighborhood_radius': 0.8,
                'region_transition_step': 1.0,
                'max_new_region_overlap_ratio': 0.45,
                'min_new_region_outside_ratio': 0.40,
                'completed_region_margin': 0.5,
                'completed_region_inner_margin': 0.5,
                'checkpoint_root': 'maps/ralc_checkpoints',
                **common_frames,
            }],
        ),

        Node(
            package='ralc_region_exploration',
            executable='ralc_frontier_planner_node',
            name='ralc_frontier_planner',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'min_frontier_cluster_size': 3,
                'min_actionable_frontier_cluster_size_cells': 15,
                'max_non_actionable_rejections': 5,
                'failed_goal_blacklist_seconds': 30.0,
                'failed_goal_aliasing_distance': 0.5,
                'max_failed_goals_per_region': 5,
                'beta_A': 0.5,
                'beta_S': 0.5,
                'beta_G': 0.02,
                'same_frontier_distance': 0.5,
                'min_goal_distance_from_robot': 0.35,
                'frontier_goal_inward_shift': 0.15,
                'frontier_goal_candidate_shifts': [0.35, 0.55, 0.80, 1.10, 1.40],
                'max_frontier_observation_distance': 2.0,
                'occupied_safety_margin': 0.10,
                'unknown_safety_margin': 0.10,
                'region_unknown_completion_threshold': 0.03,
                'region_reachable_unknown_completion_threshold': 0.03,
                'robot_coverage_radius': 0.75,
                **common_frames,
            }],
        ),

        Node(
            package='ralc_region_exploration',
            executable='ralc_alc_planner_node',
            name='ralc_alc_planner',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),

        Node(
            package='ralc_region_exploration',
            executable='ralc_pgs_planner_node',
            name='ralc_pgs_planner',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'trajectory_proxy_min_spacing': 0.75,
                'min_waypoint_spacing': 0.4,
            }],
        ),

        Node(
            package='ralc_region_exploration',
            executable='ralc_executive_node',
            name='ralc_executive',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'enable_motion': ParameterValue(enable_motion, value_type=bool),
            }],
        ),

        Node(
            package='ralc_region_exploration',
            executable='ralc_exploration_manager_node',
            name='ralc_exploration_manager',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'checkpoint_root': 'maps/ralc_checkpoints',
                'enable_map_saving': True,
                'require_region_at_max_before_no_actionable_completion': True,
                'allow_pgs_unavailable_completion': True,
                'region_unknown_completion_threshold': 0.03,
                'region_reachable_unknown_completion_threshold': 0.03,
            }],
        ),
    ])
