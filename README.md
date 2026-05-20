# Autonomous SLAM Baseline (ROS 2 Humble)

This repository provides a **clean baseline for robot autonomous SLAM** using:

* ROS 2 Humble
* `slam_toolbox` for 2D SLAM
* Nav2 (Navigation2) for planning and control
* A custom reactive node (`wander_avoid`) for initial motion validation

This setup has been validated on a **real robot platform** and serves as the starting point for:

* Autonomous exploration
* Frontier-based SLAM
* Future RF-aware exploration extensions

## 1. Install

Clone this repository, make sure the main dependencies are installed, then use colcon to build all our packages...

```sh
export ROS_DISTRO=humble

git clone https://github.com/Fatma-Ltaief/swd_autonomous_mapper.git
cd swd_autonomous_mapper 

sudo apt update
sudo apt install -y \
  ros-$ROS_DISTRO-slam-toolbox \
  ros-$ROS_DISTRO-navigation2 \
  ros-$ROS_DISTRO-nav2-bringup \
  ros-$ROS_DISTRO-tf2-tools \
  ros-$ROS_DISTRO-rviz2

source /opt/ros/$ROS_DISTRO/setup.bash
colcon build
source ./install/setup.bash
```

Or with your favorite _ROS2_ already installed (test: `env | grep ROS_DISTRO`): 

```sh
curl -sSL https://raw.githubusercontent.com/Fatma-Ltaief/swd_autonomous_mapper/refs/heads/main/bin/install | bash
cd swd_autonomous_mapper 
source ./install/setup.bash
```

---

## 2. Requirements

### 2.1 System Requirements

* Ubuntu 22.04
* ROS 2 Humble installed

### 2.2 Required ROS 2 Packages

Install the main dependencies:

```bash
sudo apt update
sudo apt install -y \
  ros-humble-slam-toolbox \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-tf2-tools \
  ros-humble-rviz2
```

Optional but useful:

```bash
sudo apt install -y \
  ros-humble-rqt \
  ros-humble-rqt-graph \
  ros-humble-teleop-twist-keyboard
```

---

## 3. Workspace Setup

```bash
mkdir -p ~/autonomous_slam_ws/src
cd ~/autonomous_slam_ws
colcon build
source install/setup.bash
```

---

## 4. Connecting to the Robot

### 4.1 ROS Domain ID

Ensure both your laptop and the robot share the same ROS domain:

```bash
export ROS_DOMAIN_ID=40
```

---

### 4.2 SSH Access

#### Ethernet connection

```bash
ssh swd_sk@192.168.50.2
```

#### IMT IoT WiFi network

```bash
ssh swd_sk@10.120.2.43
```

---

### 4.3 Environment Setup (both sides)

On **both the robot and your laptop**, always source ROS:

```bash
source /opt/ros/humble/setup.bash
source ~/autonomous_slam_ws/install/setup.bash
```

---

## 5. Running the System

The system is typically launched using **multiple terminals**.

---

### 5.1 Terminal 1 — Robot Bringup

(On the robot)

Launch the robot drivers (LiDAR, odometry, TF, etc.):

```bash
<robot_bringup_command>
```

Expected topics:

* `/scan`
* `/odom`
* `/tf`
* `/cmd_vel`

---

### 5.2 Terminal 2 — SLAM (slam_toolbox)

(On your laptop or robot)

```bash
ros2 launch slam_toolbox online_async_launch.py
```

Expected outputs:

* `/map`
* `/map_updates`
* `/tf` (map → odom)

---

### 5.3 Terminal 3 — Reactive Motion (Wander Node)

```bash
ros2 run swd_autonomous_mapper wander_avoid
```

This node:

* Moves the robot forward
* Rotates when obstacles are detected
* Uses `/scan` and publishes `/cmd_vel`

---

### 5.4 Terminal 4 — RViz

```bash
rviz2
```

Recommended displays:

* Map (`/map`)
* LaserScan (`/scan`)
* TF
* Odometry

Set:

```text
Fixed Frame = map
```

---

## 6. Verified Working Topics

```text
/scan
/odom
/cmd_vel
/map
/tf
/tf_static
```

---

## 7. Current Limitations

* The `wander_avoid` node is **purely reactive**
* It:

  * Avoids walls well
  * Struggles with clutter (chairs, desks)
* No global planning
* No optimal coverage strategy
* No guarantee of full exploration

---

## 8. Next Steps

Planned evolution of this baseline:

1. Integrate **Nav2 navigation stack**
2. Validate **manual goal navigation**
3. Add **frontier-based exploration**
4. Benchmark exploration performance
5. Extend toward **RF-aware exploration**

---

## 9. Notes

* Always ensure:

  * Same `ROS_DOMAIN_ID`
  * Correct topic names
  * Proper TF tree (`map → odom → base_link`)
* If `/map` is not visible:

  * Check SLAM is running
  * Verify TF frames
  * Confirm `/scan` is publishing

---

## 10. Repository Purpose

This repository serves as:

* A **reproducible baseline**
* A **stable starting point for research**
* A **reference configuration** before adding complexity

---
## 11. Robot Bringup

The robot bringup is responsible for launching all **hardware-related nodes** required for operation.

### 11.1 Purpose

This step initializes:

* LiDAR driver → `/scan`
* Odometry → `/odom`
* TF tree → `odom → base_link`
* Motor controller → `/cmd_vel`

---

### 11.2 Expected Topics

After bringup, verify:

```bash
ros2 topic list
```

You should see:

```text
/scan
/odom
/cmd_vel
/tf
/tf_static
```

---

### 11.3 TF Tree

Verify transforms:

```bash
ros2 run tf2_tools view_frames
```

Expected structure:

```text
map → odom → base_link → laser_frame
```

At this stage (before SLAM), you may only have:

```text
odom → base_link → laser_frame
```

---

### 11.4 Quick Validation

Check LiDAR:

```bash
ros2 topic echo /scan --once
```

Check odometry:

```bash
ros2 topic echo /odom --once
```

If both work, the robot is ready for SLAM and navigation.

---

## 12. Nav2 (Navigation Stack)

Nav2 is responsible for:

* Global planning
* Local obstacle avoidance
* Safe trajectory execution
* Recovery behaviors

---

### 12.1 Role in the Pipeline

```text
SLAM → map
        ↓
     Nav2
        ↓
  path planning + control
        ↓
     /cmd_vel
```

Nav2 replaces the simple `wander_avoid` behavior with **intelligent navigation**.

---

### 12.2 Launching Nav2

Nav2 can run **with SLAM (mapping mode)** or **with a saved map**.

#### Option A — With SLAM (recommended now)

```bash
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=false
```

Make sure:

* `/map` is being published by `slam_toolbox`
* TF includes `map → odom`

---

#### Option B — With a saved map

```bash
ros2 launch nav2_bringup navigation_launch.py \
  map:=/path/to/map.yaml \
  use_sim_time:=false
```

---

### 12.3 RViz Interaction

In RViz:

* Set **Fixed Frame = `map`**
* Use:

  * **"2D Pose Estimate"** → initialize robot position
  * **"2D Goal Pose"** → send navigation goal

---

### 12.4 Expected Behavior

When Nav2 is working correctly:

* The robot:

  * Plans a path to the goal
  * Avoids obstacles (walls, chairs, desks)
  * Replans dynamically if needed
* You should see:

  * `/plan` topic
  * costmaps updating
  * smooth motion instead of reactive turning

---

### 12.5 Important Topics

```text
/cmd_vel            → velocity commands
/plan               → global path
/local_costmap      → obstacle map (local)
/global_costmap     → global planning map
/behavior_tree_log  → decision logic
```

---

### 12.6 Key Parameters to Tune

For real robot safety:

* **Inflation radius**
* **Robot footprint**
* **Obstacle layer**
* **Max velocity**
* **Acceleration limits**

These are defined in Nav2 config YAML files.

---

### 12.7 Common Issues

#### No movement

* Check `/cmd_vel`
* Ensure robot accepts commands
* Verify controller is active

#### Robot hits obstacles

* Costmap not configured correctly
* Inflation radius too small
* Laser not integrated in costmap

#### “No map received”

* SLAM not running
* TF missing `map → odom`

---
## 13. Autonomous Exploration (Frontier-Based)

This section explains how to run the **working autonomous frontier-based exploration baseline** on the real robot.

The current pipeline is:

- robot base already running
- `slam_toolbox` for online mapping
- Nav2 for planning and control
- `nav2_wavefront_frontier_exploration` for automatic frontier selection

The robot will:

- build the map online
- detect frontiers from unknown map regions
- send navigation goals automatically
- continue until no more frontiers remain

---
Terminal 1: SLAM
```bash
export ROS_DOMAIN_ID=40
source /opt/ros/humble/setup.bash
source ~/ws-ros2/install/setup.bash

ros2 launch slam_toolbox online_async_launch.py
```
This starts online SLAM and publishes:
* /map
* /map_updates
* map -> odom

Terminal 2: Nav2
```bash
export ROS_DOMAIN_ID=40
source /opt/ros/humble/setup.bash
source ~/ws-ros2/install/setup.bash

ros2 launch swd_nav2 nav2_with_slam.launch.py
```
This starts the navigation stack, including:

* planner
* controller
* behavior tree navigator
* local and global costmaps
* waypoint follower
Terminal 3: RViz
```bash
export ROS_DOMAIN_ID=40
source /opt/ros/humble/setup.bash
source ~/ws-ros2/install/setup.bash

rviz2
```
Recommended RViz displays:

* Map
* TF
* LaserScan
* Path on /plan
* Path on /local_plan

Set:

Fixed Frame = map
Terminal 4: Frontier Exploration
``` bash
export ROS_DOMAIN_ID=40
source /opt/ros/humble/setup.bash
source ~/ws-ros2/install/setup.bash

ros2 run nav2_wfd explore
```
This starts the frontier exploration node.

The node:

* listens to /map
* gets robot pose from /odom
* detects frontier regions
* sends waypoint goals to Nav2
* repeats until exploration is finished

---

# Gazebo Simulation

Launch the robot in Gazebo:

```bash
ros2 launch swd_sim gazebo_robot.launch.py
```

Move the robot:

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.5}}" -r 10
```

Check odometry:

```bash
ros2 topic echo /odom
```
