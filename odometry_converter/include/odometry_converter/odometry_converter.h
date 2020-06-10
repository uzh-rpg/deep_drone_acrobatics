#pragma once

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <fstream>

#include <geometry_msgs/PoseStamped.h>
#include <kindr/minimal/quat-transformation.h>
#include <tf/transform_broadcaster.h>
#include "nav_msgs/Odometry.h"
#include "quadrotor_msgs/AutopilotFeedback.h"
#include "quadrotor_msgs/ControlCommand.h"
#include "rpg_common/pose.h"
#include "sensor_msgs/Imu.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Int8.h"

namespace odometry_converter {

class OdometryConverter {
 public:
  OdometryConverter(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);

  OdometryConverter()
      : OdometryConverter(ros::NodeHandle(), ros::NodeHandle("~")) {}

 private:
  enum STATES { kGroundTruth = 0, kVIO = 1, kChimaera = 2 };

  void odometryCallback(const nav_msgs::OdometryConstPtr& msg);

  void groundTruthCallback(const nav_msgs::OdometryConstPtr& msg);

  void switchCallback(const std_msgs::Int8ConstPtr& msg);

  bool loadParameters();

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber odometry_sub_;
  ros::Subscriber ground_truth_sub_;
  ros::Subscriber switch_sub_;
  ros::Publisher odometry_pub_;
  ros::Publisher odometry_pub_vio_;

  nav_msgs::Odometry vision_based_odometry_;
  nav_msgs::Odometry ground_truth_odometry_;

  rpg::Pose T_W_B_;
  rpg::Pose T_V_B_;
  rpg::Pose T_W_V_;
  rpg::Pose T_B_S_;
  STATES switch_odometry_ = STATES::kGroundTruth;
  tf::TransformBroadcaster tf_broadcaster_;
};

}  // namespace odometry_converter
