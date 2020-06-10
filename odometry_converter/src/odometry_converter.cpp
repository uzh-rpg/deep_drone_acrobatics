#include "odometry_converter/odometry_converter.h"

#include "minkindr_conversions/kindr_msg.h"

namespace odometry_converter {

OdometryConverter::OdometryConverter(const ros::NodeHandle& nh,
                                     const ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh) {
  if (!loadParameters()) {
    ROS_ERROR("[%s] Could not load parameters.", pnh_.getNamespace().c_str());
  }

  odometry_sub_ =
      nh_.subscribe("odometry_in", 1, &OdometryConverter::odometryCallback,
                    this, ros::TransportHints().tcpNoDelay());
  ground_truth_sub_ = nh_.subscribe("ground_truth_in", 1,
                                    &OdometryConverter::groundTruthCallback,
                                    this, ros::TransportHints().tcpNoDelay());
  switch_sub_ = nh_.subscribe("switch_odometry", 1,
                              &OdometryConverter::switchCallback, this);
  odometry_pub_ = nh_.advertise<nav_msgs::Odometry>("odometry_out", 0);
  odometry_pub_vio_ = nh_.advertise<nav_msgs::Odometry>(
      "/hummingbird/odometry_converted_vio", 0);
}

void OdometryConverter::groundTruthCallback(
    const nav_msgs::OdometryConstPtr& msg) {
  double quat_norm = std::sqrt(std::pow(msg->pose.pose.orientation.w, 2) +
                               std::pow(msg->pose.pose.orientation.x, 2) +
                               std::pow(msg->pose.pose.orientation.y, 2) +
                               std::pow(msg->pose.pose.orientation.z, 2));
  if (quat_norm < 0.9999 || quat_norm > 1.0001) {
    ROS_WARN("Invalid orientation of ground truth (norm of quaternion = %.3f)!",
             quat_norm);
  }
  ground_truth_odometry_ = *msg;
  if (quat_norm > 0.0) {
    // fix qw to be in the positive half space
    double multiplier = 1.0;
    if (msg->pose.pose.orientation.w < 0.0) {
      multiplier = -1.0;
    }
    ground_truth_odometry_.pose.pose.orientation.w =
        multiplier / quat_norm * msg->pose.pose.orientation.w;
    ground_truth_odometry_.pose.pose.orientation.x =
        multiplier / quat_norm * msg->pose.pose.orientation.x;
    ground_truth_odometry_.pose.pose.orientation.y =
        multiplier / quat_norm * msg->pose.pose.orientation.y;
    ground_truth_odometry_.pose.pose.orientation.z =
        multiplier / quat_norm * msg->pose.pose.orientation.z;
  } else {
    ground_truth_odometry_.pose.pose.orientation.w = 1.0;
    ground_truth_odometry_.pose.pose.orientation.x = 0.0;
    ground_truth_odometry_.pose.pose.orientation.y = 0.0;
    ground_truth_odometry_.pose.pose.orientation.z = 0.0;
  }

  tf::Transform transform;
  transform.setOrigin(tf::Vector3(msg->pose.pose.position.x,
                                  msg->pose.pose.position.y,
                                  msg->pose.pose.position.z));
  tf::Quaternion q;
  q.setRPY(0, 0, 0);
  transform.setRotation(q);
  tf_broadcaster_.sendTransform(tf::StampedTransform(
      transform, ros::Time::now(), "world", "/follower_camera_target"));

  if (switch_odometry_ == STATES::kGroundTruth) {
    odometry_pub_.publish(ground_truth_odometry_);
    odometry_pub_vio_.publish(ground_truth_odometry_);
  }
}

void OdometryConverter::odometryCallback(
    const nav_msgs::OdometryConstPtr& msg) {
  double quat_norm = std::sqrt(std::pow(msg->pose.pose.orientation.w, 2) +
                               std::pow(msg->pose.pose.orientation.x, 2) +
                               std::pow(msg->pose.pose.orientation.y, 2) +
                               std::pow(msg->pose.pose.orientation.z, 2));
  if (quat_norm < 0.9999 || quat_norm > 1.0001) {
    ROS_WARN_THROTTLE(1.0,
                      "Invalid orientation of VIO (norm of quaternion = %.3f)!",
                      quat_norm);
  }
  vision_based_odometry_ = *msg;
  if (quat_norm > 0.0) {
    vision_based_odometry_.pose.pose.orientation.w =
        1.0 / quat_norm * msg->pose.pose.orientation.w;
    vision_based_odometry_.pose.pose.orientation.x =
        1.0 / quat_norm * msg->pose.pose.orientation.x;
    vision_based_odometry_.pose.pose.orientation.y =
        1.0 / quat_norm * msg->pose.pose.orientation.y;
    vision_based_odometry_.pose.pose.orientation.z =
        1.0 / quat_norm * msg->pose.pose.orientation.z;
  } else {
    vision_based_odometry_.pose.pose.orientation.w = 1.0;
    vision_based_odometry_.pose.pose.orientation.x = 0.0;
    vision_based_odometry_.pose.pose.orientation.y = 0.0;
    vision_based_odometry_.pose.pose.orientation.z = 0.0;
  }

  // VINS Mono publishes linear velocity estimates in world frame, transform
  // them to body frame:
  Eigen::Vector3d vlin_world =
      Eigen::Vector3d(vision_based_odometry_.twist.twist.linear.x,
                      vision_based_odometry_.twist.twist.linear.y,
                      vision_based_odometry_.twist.twist.linear.z);
  Eigen::Quaterniond q_eigen =
      Eigen::Quaterniond(vision_based_odometry_.pose.pose.orientation.w,
                         vision_based_odometry_.pose.pose.orientation.x,
                         vision_based_odometry_.pose.pose.orientation.y,
                         vision_based_odometry_.pose.pose.orientation.z);
  Eigen::Vector3d vlin_body = q_eigen.inverse() * vlin_world;
  vision_based_odometry_.twist.twist.linear.x = vlin_body.x();
  vision_based_odometry_.twist.twist.linear.y = vlin_body.y();
  vision_based_odometry_.twist.twist.linear.z = vlin_body.z();

  tf::poseMsgToKindr(vision_based_odometry_.pose.pose, &T_V_B_);

  nav_msgs::Odometry converted_odometry;
  T_W_B_ = T_W_V_ * T_V_B_;
  tf::poseKindrToMsg(T_W_B_, &converted_odometry.pose.pose);
  // fix qw to be in the positive half space
  if (converted_odometry.pose.pose.orientation.w < 0.0) {
    converted_odometry.pose.pose.orientation.w =
        -1.0 * converted_odometry.pose.pose.orientation.w;
    converted_odometry.pose.pose.orientation.x =
        -1.0 * converted_odometry.pose.pose.orientation.x;
    converted_odometry.pose.pose.orientation.y =
        -1.0 * converted_odometry.pose.pose.orientation.y;
    converted_odometry.pose.pose.orientation.z =
        -1.0 * converted_odometry.pose.pose.orientation.z;
  }

  if (switch_odometry_ == STATES::kVIO) {
    converted_odometry.header = ground_truth_odometry_.header;
    converted_odometry.twist = vision_based_odometry_.twist;
    odometry_pub_.publish(ground_truth_odometry_);
    odometry_pub_vio_.publish(converted_odometry);
  } else if (switch_odometry_ == STATES::kChimaera) {
    // We take Position&Orientation from the VIO (Position is anyway not used
    // later...)
    converted_odometry.header = ground_truth_odometry_.header;
    // But we take linear&angular velocities from ground truth
    converted_odometry.twist = ground_truth_odometry_.twist;
    odometry_pub_.publish(ground_truth_odometry_);
    odometry_pub_vio_.publish(converted_odometry);
  }
}

void OdometryConverter::switchCallback(const std_msgs::Int8ConstPtr& msg) {
  if (msg->data > 0 && switch_odometry_ == STATES::kGroundTruth) {
    ROS_INFO("Received switch odometry signal: %d", msg->data);
    tf::poseMsgToKindr(vision_based_odometry_.pose.pose, &T_V_B_);
    tf::poseMsgToKindr(ground_truth_odometry_.pose.pose, &T_W_B_);
    T_W_V_ = T_W_B_ * T_V_B_.inverse();
    switch (msg->data) {
      case STATES::kVIO: {
        ROS_INFO("Switch to VIO mode!");
        switch_odometry_ = STATES::kVIO;
        break;
      }
      case STATES::kChimaera: {
        ROS_INFO("Switch to Chimaera mode!");
        switch_odometry_ = STATES::kChimaera;
        break;
      }
    }
  } else if (msg->data && switch_odometry_ == true) {
    // these are the subsequent VIO-reinitializations. Here we have to make sure
    // that the autopilot does NOT see the discontinuities. Therefore, the
    // reinitialization state should be transformed to the previous hover state
    // or the previous trajectory end state.
  } else {
    ROS_INFO("Received switch odometry signal: %d", msg->data);
    ROS_INFO("Switch to Ground Truth mode!");
    switch_odometry_ = STATES::kGroundTruth;
  }
}

bool OdometryConverter::loadParameters() {
  bool check = true;

  // initialize T_B_S_
  geometry_msgs::Pose t_B_S;
  t_B_S.position.x = 0.0;
  t_B_S.position.y = 0.0;
  t_B_S.position.z = -0.1;

  // standard realsense mount (60deg downward looking)
  t_B_S.orientation.w = 0.1830127;
  t_B_S.orientation.x = 0.6830127;
  t_B_S.orientation.y = 0.6830127;
  t_B_S.orientation.z = 0.1830127;

  tf::poseMsgToKindr(t_B_S, &T_B_S_);

  return check;
}

}  // namespace odometry_converter
