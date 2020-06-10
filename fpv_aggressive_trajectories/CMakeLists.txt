cmake_minimum_required(VERSION 2.8.3)
project(fpv_aggressive_trajectories)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
add_compile_options(-O3)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_add_library(visualize src/visualize.cpp)

cs_add_executable(fpv_aggressive_trajectories src/fpv_aggressive_trajectories.cpp)
target_include_directories(fpv_aggressive_trajectories PUBLIC
        ../../rpg_mpc/model/quadrotor_mpc_codegen/
        ../../rpg_mpc/externals/qpoases
        ../../rpg_mpc/externals/qpoases/INCLUDE
        ../../rpg_mpc/externals/qpoases/SRC)
target_link_libraries(fpv_aggressive_trajectories visualize)
cs_add_executable(odometry_republisher src/odometry_republisher.cpp)

cs_install()
cs_export()
