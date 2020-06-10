cmake_minimum_required(VERSION 2.8.3)
project(odometry_converter)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_add_library(odometry_converter src/odometry_converter.cpp)

cs_add_executable(odometry_converter_node src/odometry_converter_node.cpp)
target_link_libraries(odometry_converter_node odometry_converter)

cs_install()
cs_export()

# CMake Indexing
FILE(GLOB_RECURSE LibFiles "include/*")
add_custom_target(headers SOURCES ${LibFiles})
FILE(GLOB_RECURSE ProtoFiles "proto/*")
add_custom_target(protos SOURCES ${ProtoFiles})
