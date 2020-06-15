#pragma once

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <pose_utils/pose.h>

namespace odometry_republisher {

class OdometryRepublisher {
 public:
  OdometryRepublisher();
  virtual ~OdometryRepublisher();

 private:
  ros::NodeHandle nh_;

  ros::Subscriber state_estimate_sub_;
  ros::Publisher odometry_out_pub_;

  rpg::Pose T_B_S_;

  void odometryInCallback(const nav_msgs::OdometryConstPtr& msg);
  void transformOdometry(const nav_msgs::OdometryConstPtr& odom_in,
                         nav_msgs::Odometry* odom_out);
};

}  // namespace odometry_republisher
