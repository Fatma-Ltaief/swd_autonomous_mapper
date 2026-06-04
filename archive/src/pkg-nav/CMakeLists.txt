cmake_minimum_required(VERSION 3.5)
project(pkg-nav)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)


#ament_python_install_package("realsense_human_detection")

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
) 

install(PROGRAMS 
  scripts/human_following.py
  scripts/goal_delay.py
  scripts/velocity_multiplexer.py
  scripts/sim_clock.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
