#include <memory>

#include "odometry_converter/odometry_converter.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "odometry_converter");

  odometry_converter::OdometryConverter odometry_converter;

  ros::spin();
  return 0;
}
