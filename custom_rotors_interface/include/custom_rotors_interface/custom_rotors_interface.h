#pragma once

#include <ros/ros.h>
#include <Eigen/Eigen>

#include <geometry_msgs/PoseStamped.h>
#include <quadrotor_common/control_command.h>
#include <quadrotor_msgs/ControlCommand.h>
#include "mav_msgs/Actuators.h"
#include "nav_msgs/Odometry.h"
#include "quadrotor_msgs/AutopilotFeedback.h"
#include "pose_utils/pose.h"
#include "std_msgs/Bool.h"

namespace custom_rotors_interface {

struct TorquesAndThrust {
  Eigen::Vector3d body_torques;
  double collective_thrust;
};

class CustomRotorsInterface {
 public:
  CustomRotorsInterface(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);

  CustomRotorsInterface()
      : CustomRotorsInterface(ros::NodeHandle(), ros::NodeHandle("~")) {}

 private:
  void armCallback(const std_msgs::BoolConstPtr& msg);

  void controlCommandCallback(
      const quadrotor_msgs::ControlCommandConstPtr& msg);

  void odometryCallback(const nav_msgs::OdometryConstPtr& msg);

  void autopilotCallback(const quadrotor_msgs::AutopilotFeedbackConstPtr& msg);

  void motorSpeedCallback(const mav_msgs::Actuators::ConstPtr& msg);

  void lowLevelControlLoop();

  TorquesAndThrust bodyRateControl(
      const quadrotor_common::ControlCommand& rate_cmd,
      const Eigen::Vector3d& body_rate_estimate);

  mav_msgs::Actuators mixer(const TorquesAndThrust& torques_and_thrust);

  bool loadParameters();

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber arm_sub_;
  ros::Subscriber motor_speed_sub_;
  ros::Subscriber ctrl_cmd_sub_;
  ros::Subscriber imu_sub_;
  ros::Subscriber odometry_sub_;
  ros::Subscriber controller_sub_;
  ros::Subscriber network_sub_;
  ros::Subscriber autopilot_sub_;

  ros::Publisher desired_motor_speed_pub_;

  ros::WallTime start_time_;

  std::string data_dir_;

  Eigen::Vector3d prev_v_lin_, prev_v_ang_;
  Eigen::Vector3d incr_v_lin_, incr_v_ang_;

  ros::Time last_hover_time_ = ros::Time::now();
  int maneuver_idx_ = 0;
  int action_idx_ = 0;
  bool velocity_estimate_in_world_frame_;
  bool enable_navigation_ = false;
  bool reference_frame_set_ = false;
  bool interface_armed_ = false;

  TorquesAndThrust torques_and_thrust_estimate_;

  double inertia_x_;
  double inertia_y_;
  double inertia_z_;
  double body_rates_p_xy_;
  double body_rates_d_xy_;
  double body_rates_p_z_;
  double body_rates_d_z_;

  // Parameters
  Eigen::Matrix3d inertia_;
  Eigen::MatrixXd K_lqr_;
  double low_level_control_frequency_;
  double roll_pitch_cont_gain_;
  double arm_length_;
  double rotor_drag_coeff_;
  double rotor_thrust_coeff_;
  double mass_;
  double max_rotor_speed_;

  quadrotor_msgs::ControlCommand control_command_;
  nav_msgs::Odometry odometry_;
  quadrotor_msgs::AutopilotFeedback autopilot_feedback_;
  double rel_time_;
};

}  // namespace custom_rotors_interface
