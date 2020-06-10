cmake_minimum_required(VERSION 2.8.3)
project(custom_rotors_interface)

# Add plain cmake packages
find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_add_library(custom_rotors_interface src/custom_rotors_interface.cpp)

cs_add_executable(custom_rotors_interface_node src/custom_rotors_interface_node.cpp)
target_link_libraries(custom_rotors_interface_node custom_rotors_interface)

cs_install()
cs_export()

# CMake Indexing
FILE(GLOB_RECURSE LibFiles "include/*")
add_custom_target(headers SOURCES ${LibFiles})
FILE(GLOB_RECURSE ProtoFiles "proto/*")
add_custom_target(protos SOURCES ${ProtoFiles})
