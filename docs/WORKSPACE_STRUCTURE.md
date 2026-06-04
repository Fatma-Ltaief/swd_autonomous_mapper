# Autonomous SLAM Baseline Workspace Structure

This workspace is focused on the PPI office SLAM, Nav2, frontier exploration, and the shared SSI message package used by the radio pipeline.

## Active Packages And Folders

- `src/swd_starter_kit_description`: robot meshes, URDF, and description assets for the SWD robot.
- `src/swd_sim`: active Gazebo and SLAM simulation package.
- `src/swd_nav2`: active Nav2 configuration package, including PPI-specific navigation parameters.
- `src/nav2_wavefront_frontier_exploration`: active wavefront frontier exploration package.
- `src/router-data`: active ROS 2 message package. Keep this in `src`; it defines `Ssi` and `RssiPosition`, which are used by the radio/SSI pipeline.

Additional package currently left untouched:

- `src/swd_autonomous_mapper`: not listed as active or archive in the cleanup request, so it was left in place.

## Active PPI Simulation Assets

- `src/swd_sim/worlds/ppi_office.world`: current PPI office Gazebo world.
- `src/swd_sim/models/ppi_office`: current PPI office model.
- `src/swd_sim/PPI`: current PPI assets.
- `src/swd_sim/launch/ppi_frontier_exploration.launch.py`: current PPI frontier exploration launch file.
- `src/swd_sim/launch/gazebo_robot.launch.py`: current Gazebo robot launch file.
- `src/swd_sim/launch/imt_slam.launch.py`: current SLAM launch file.
- `src/swd_sim/config/swd_slam_toolbox.yaml`: current SLAM Toolbox configuration.

## Archived Items

Archived files were moved under `archive/` with their original relative paths preserved.

- `archive/src/my_robot_nav`: old robot navigation package. Archived because the active Nav2 configuration now lives in `src/swd_nav2`.
- `archive/src/pkg-nav`: older navigation package and launch/config experiments. Archived because it is not part of the current PPI workflow.
- `archive/src/swd_sim/models/ivy_office`: Ivy office model. Archived because PPI office is the active simulation environment.
- `archive/src/swd_sim/worlds/ivy_office.world`: Ivy office world. Archived because PPI office is the active simulation environment.
- `archive/src/swd_sim/launch/ivy_office_world.launch.py`: Ivy office launch file. Archived because it is not part of the current PPI launch path.
- `archive/src/swd_sim/worlds/simple_colored_warehouse.sdf`: warehouse experiment world. Archived because the current workflow uses PPI office.
- `archive/src/swd_sim/worlds/warehouse_world.sdf`: warehouse experiment world. Archived because the current workflow uses PPI office.
- `archive/src/swd_sim/launch/warehouse_robot.launch.py`: warehouse experiment launch file. Archived because the current workflow uses PPI office.
- `archive/src/swd_sim/launch/warehouse_frontier_exploration.launch.py`: warehouse frontier experiment launch file. Archived because `ppi_frontier_exploration.launch.py` is the active frontier workflow.
- `archive/src/warehouse_world.png`: warehouse experiment image. Archived because it belongs to the old warehouse workflow.

Requested optional files not found during cleanup:

- `src/warehouse_slam_map.pgm`
- `src/warehouse_slam_map.yaml`

## Recommended Launch Commands

Build active packages after structural changes:

```bash
cd ~/autonomous_slam_baseline
colcon build --packages-select swd_starter_kit_description swd_sim swd_nav2 nav2_wfd rutx_snmp_read
source install/setup.bash
```

Launch the active PPI frontier exploration stack:

```bash
ros2 launch swd_sim ppi_frontier_exploration.launch.py
```

When running the radio/SSI pipeline with this workspace, source this workspace before the radio workspace so `rutx_snmp_read` messages are available:

```bash
source ~/autonomous_slam_baseline/install/setup.bash
source ~/radio_gazebo_sim/install/setup.bash
```
