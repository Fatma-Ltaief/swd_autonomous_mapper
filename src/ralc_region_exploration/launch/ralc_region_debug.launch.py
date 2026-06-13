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
                'next_region_seed_search_min_radius': 0.25,
                'next_region_seed_search_max_radius': 1.20,
                'next_region_seed_search_step': 0.10,
                'next_region_seed_occupied_clearance': 0.25,
                'max_next_region_frontier_boundary_distance': 2.0,
                'next_region_robot_distance_penalty': 0.35,
                'next_region_boundary_distance_penalty': 0.50,
                'next_region_seed_adjustment_penalty': 0.25,
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

                # Frontier clustering
                'min_frontier_cluster_size': 3,
                'min_actionable_frontier_cluster_size_cells': 6,

                # Recovery / blacklist
                'max_non_actionable_rejections': 8,
                'failed_goal_blacklist_seconds': 10.0,
                'failed_goal_aliasing_distance': 0.35,
                'max_failed_goals_per_region': 10,
                'min_observation_travel_distance': 0.20,
                'min_frontier_reduction_ratio': 0.10,
                'ineffective_goal_blacklist_seconds': 15.0,
                'ineffective_goal_aliasing_distance': 0.5,
                'max_ineffective_observation_attempts_per_cluster': 2,
                'min_visible_unknown_gain': 5,
                'reject_observation_pose_no_visible_unknown': False,
                'deferred_frontier_recheck_distance': 1.0,
                'deferred_frontier_timeout_sec': 20.0,
                'max_deferred_frontiers_before_region_growth': 3,

                # Scoring
                'beta_A': 0.5,
                'beta_S': 0.5,
                'beta_G': 0.02,
                'same_frontier_distance': 0.4,

                # Goal filtering
                'min_goal_distance_from_robot': 0.25,
                'frontier_goal_inward_shift': 0.25,
                'frontier_goal_candidate_shifts': [0.25, 0.40, 0.60, 0.80, 1.00, 1.30],
                'max_frontier_observation_distance': 3.0,
                'enable_recovery_frontier_without_astar': False,

                # Safety margins
                'occupied_safety_margin': 0.25,
                'unknown_safety_margin': 0.20,

                # Region completion
                'region_unknown_completion_threshold': 0.03,
                'region_reachable_unknown_completion_threshold': 0.05,

                # Coverage
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
                'use_trajectory_proxy_if_backend_missing': False,
                'min_pgs_regional_keyframes': 3,
                'pgs_keyframe_dedup_distance': 0.10,
                'close_hull_loop': True,
                'trajectory_proxy_min_spacing': 0.75,
                'pgs_min_waypoint_spacing': 0.45,
                'pgs_use_region_owned_keyframes': True,
                'pgs_filter_keyframes_by_region_entry': True,
                'pgs_region_entry_anchor_node_count': 0,
                'pgs_use_costmap_waypoint_filter': True,
                'pgs_costmap_topic': '/global_costmap/costmap',
                'pgs_waypoint_cost_threshold': 35,
                'pgs_waypoint_clearance': 0.25,
                'pgs_waypoint_snap_radius': 0.80,
                'pgs_min_safe_waypoints': 2,
                'pgs_require_costmap_path_connectivity': False,
                'pgs_max_waypoint_path_length': 8.0,
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
                'nav_goal_timeout_sec': 90.0,
                'nav_stuck_timeout_sec': 12.0,
                'nav_stuck_progress_distance': 0.08,
                'cancel_nav_goal_on_stuck': True,
                'publish_zero_cmd_on_abort': True,
                **common_frames,
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
                'allow_pgs_unavailable_completion': False,
                'region_unknown_completion_threshold': 0.03,
                'region_reachable_unknown_completion_threshold': 0.03,
                'min_observation_travel_distance': 0.20,
                'min_frontier_reduction_ratio': 0.10,
                'observation_update_timeout_sec': 3.0,
                'observation_validation_wait_sec': 2.0,
                'min_observation_map_updates': 2,
                'min_successful_frontier_goals_before_region_completion': 8,
                'min_robot_travel_distance_before_region_completion': 2.0,
                'min_region_active_time_before_completion_sec': 30.0,
                'min_map_updates_before_region_completion': 20,
                'require_motion_or_validated_frontier_for_completion': True,
                'pgs_retry_cooldown_sec': 5.0,
                'pgs_request_timeout_sec': 4.0,
                'max_pgs_refinement_failures_per_region': 2,
            }],
        ),
        Node(
            package='ralc_region_exploration',
            executable='ralc_slam_backend_node',
            name='ralc_slam_backend',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'graph_visualization_topic': '/slam_toolbox/graph_visualization',
                'node_marker_namespace': 'slam_toolbox',
                'edge_marker_namespace': 'slam_toolbox_edges',
                'publish_period_sec': 1.0,
                'assign_keyframes_to_refinement_regions': True,
            }],
        ),
    ])
