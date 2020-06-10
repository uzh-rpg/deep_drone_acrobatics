#include "custom_rotors_interface/custom_rotors_interface.h"

#include <quadrotor_common/math_common.h>
#include "quadrotor_common/geometry_eigen_conversions.h"

namespace custom_rotors_interface {

CustomRotorsInterface::CustomRotorsInterface(const ros::NodeHandle& nh,
                                             const ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh) {
  if (!loadParameters()) {
    ROS_ERROR("[%s] Could not load parameters.", pnh_.getNamespace().c_str());
  }

  arm_sub_ =
      nh_.subscribe("bridge/arm", 1, &CustomRotorsInterface::armCallback, this);
  ctrl_cmd_sub_ = nh_.subscribe("control_command", 1,
                                &CustomRotorsInterface::controlCommandCallback,
                                this, ros::TransportHints().tcpNoDelay());
  autopilot_sub_ = nh_.subscribe("autopilot/feedback", 1,
                                 &CustomRotorsInterface::autopilotCallback,
                                 this, ros::TransportHints().tcpNoDelay());
  odometry_sub_ =
      nh_.subscribe("odometry", 1, &CustomRotorsInterface::odometryCallback,
                    this, ros::TransportHints().tcpNoDelay());
  motor_speed_sub_ = nh_.subscribe("motor_speed", 1,
                                   &CustomRotorsInterface::motorSpeedCallback,
                                   this, ros::TransportHints().tcpNoDelay());
  desired_motor_speed_pub_ =
      nh_.advertise<mav_msgs::Actuators>("command/motor_speed", 1);
}

void CustomRotorsInterface::armCallback(const std_msgs::BoolConstPtr& msg) {
  if (msg->data) {
    interface_armed_ = true;
    ROS_INFO("[%s] Interface armed", pnh_.getNamespace().c_str());
  } else {
    interface_armed_ = false;
    ROS_INFO("[%s] Interface disarmed", pnh_.getNamespace().c_str());
  }
}

void CustomRotorsInterface::controlCommandCallback(
    const quadrotor_msgs::ControlCommandConstPtr& msg) {
  control_command_ = *msg;
}

void CustomRotorsInterface::autopilotCallback(
    const quadrotor_msgs::AutopilotFeedbackConstPtr& msg) {
  autopilot_feedback_ = *msg;
}

void CustomRotorsInterface::odometryCallback(
    const nav_msgs::OdometryConstPtr& msg) {
  odometry_ = *msg;
}

void CustomRotorsInterface::motorSpeedCallback(
    const mav_msgs::Actuators::ConstPtr& msg) {
  // The hummingbird that we use in our simulations has the following rotor
  // configuration Rotor 0 spins clockwise
  //                 x
  //    0            ^
  //    |            |
  // 1--+--3    y <--+
  //    |
  //    2

  const double f0 = rotor_thrust_coeff_ * pow(msg->angular_velocities[0], 2.0);
  const double f1 = rotor_thrust_coeff_ * pow(msg->angular_velocities[1], 2.0);
  const double f2 = rotor_thrust_coeff_ * pow(msg->angular_velocities[2], 2.0);
  const double f3 = rotor_thrust_coeff_ * pow(msg->angular_velocities[3], 2.0);

  torques_and_thrust_estimate_.body_torques.x() = arm_length_ * (f1 - f3);
  torques_and_thrust_estimate_.body_torques.y() = arm_length_ * (f2 - f0);
  torques_and_thrust_estimate_.body_torques.z() =
      rotor_drag_coeff_ * (f0 - f1 + f2 - f3);
  torques_and_thrust_estimate_.collective_thrust = f0 + f1 + f2 + f3;
  lowLevelControlLoop();
}

void CustomRotorsInterface::lowLevelControlLoop() {
  mav_msgs::Actuators desired_motor_speed;
  quadrotor_msgs::ControlCommand control_command = control_command_;
  nav_msgs::Odometry quad_state = odometry_;

  if (!interface_armed_ || !control_command.armed) {
    for (int i = 0; i < 4; i++) {
      desired_motor_speed.angular_velocities.push_back(0.0);
    }
  } else {
    if (control_command.control_mode == control_command.BODY_RATES) {
      const quadrotor_common::ControlCommand rate_cmd =
          quadrotor_common::ControlCommand(control_command);
      const TorquesAndThrust torques_and_thrust = bodyRateControl(
          rate_cmd,
          quadrotor_common::geometryToEigen(quad_state.twist.twist.angular));
      desired_motor_speed = mixer(torques_and_thrust);
    } else {
      ROS_ERROR_THROTTLE(1,
                         "[%s] Undefined control mode, will not apply command.",
                         ros::this_node::getName().c_str());
      return;
    }
  }

  desired_motor_speed_pub_.publish(desired_motor_speed);
}

TorquesAndThrust CustomRotorsInterface::bodyRateControl(
    const quadrotor_common::ControlCommand& rate_cmd,
    const Eigen::Vector3d& body_rate_estimate) {
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 3);
  Eigen::VectorXd control_error = Eigen::VectorXd::Zero(6);
  //  control_error.segment(0, 3) = rate_cmd.bodyrates - body_rate_estimate;
  //  control_error.segment(3, 3) = rate_cmd.bodyrates.cross(inertia_ *
  //  rate_cmd.bodyrates)
  //                                + inertia_ * rate_cmd.angular_accelerations
  //                                - torques_and_thrust_estimate_.body_torques;
  control_error.segment(0, 3) = rate_cmd.bodyrates - body_rate_estimate;
  control_error.segment(3, 3) =
      rate_cmd.bodyrates.cross(inertia_ * rate_cmd.bodyrates) +
      inertia_ * rate_cmd.angular_accelerations -
      torques_and_thrust_estimate_.body_torques;

  TorquesAndThrust torques_and_thrust;
  //  torques_and_thrust.body_torques = K_lqr_ * control_error +
  //  body_rate_estimate.cross(inertia_ * body_rate_estimate)
  //                                    + inertia_ *
  //                                    rate_cmd.angular_accelerations;
  // new version
  torques_and_thrust.body_torques =
      K_lqr_ * control_error +
      body_rate_estimate.cross(inertia_ * body_rate_estimate) +
      inertia_ * rate_cmd.angular_accelerations;
  torques_and_thrust.collective_thrust = rate_cmd.collective_thrust;

  return torques_and_thrust;
}

mav_msgs::Actuators CustomRotorsInterface::mixer(
    const TorquesAndThrust& torques_and_thrust) {
  // Using Rotor Convention of the Hummingbird
  mav_msgs::Actuators rotor_speed_cmds;
  for (int i = 0; i < 4; i++) {
    rotor_speed_cmds.angular_velocities.push_back(0.0);
  }

  // Compute square of single rotor speed commands
  if (torques_and_thrust.collective_thrust < 0.05) {
    return rotor_speed_cmds;
  }
  // Compute the square of the single rotor speeds
  rotor_speed_cmds.angular_velocities[0] =
      ((arm_length_ * torques_and_thrust.body_torques.z() -
        2.0 * rotor_drag_coeff_ * torques_and_thrust.body_torques.y() +
        rotor_drag_coeff_ * arm_length_ * mass_ *
            torques_and_thrust.collective_thrust) /
       (4.0 * rotor_drag_coeff_ * arm_length_)) /
      rotor_thrust_coeff_;
  rotor_speed_cmds.angular_velocities[1] =
      ((2.0 * rotor_drag_coeff_ * torques_and_thrust.body_torques.x() -
        arm_length_ * torques_and_thrust.body_torques.z() +
        rotor_drag_coeff_ * arm_length_ * mass_ *
            torques_and_thrust.collective_thrust) /
       (4.0 * rotor_drag_coeff_ * arm_length_)) /
      rotor_thrust_coeff_;
  rotor_speed_cmds.angular_velocities[2] =
      ((2.0 * rotor_drag_coeff_ * torques_and_thrust.body_torques.y() +
        arm_length_ * torques_and_thrust.body_torques.z() +
        rotor_drag_coeff_ * arm_length_ * mass_ *
            torques_and_thrust.collective_thrust) /
       (4.0 * rotor_drag_coeff_ * arm_length_)) /
      rotor_thrust_coeff_;
  rotor_speed_cmds.angular_velocities[3] =
      (-(2.0 * rotor_drag_coeff_ * torques_and_thrust.body_torques.x() +
         arm_length_ * torques_and_thrust.body_torques.z() -
         rotor_drag_coeff_ * arm_length_ * mass_ *
             torques_and_thrust.collective_thrust) /
       (4.0 * rotor_drag_coeff_ * arm_length_)) /
      rotor_thrust_coeff_;

  // Apply limits and take square root
  for (int i = 0; i < 4; i++) {
    quadrotor_common::limit(&rotor_speed_cmds.angular_velocities[i], 0.0,
                            pow(max_rotor_speed_, 2.0));
    rotor_speed_cmds.angular_velocities[i] =
        sqrt(rotor_speed_cmds.angular_velocities[i]);
  }

  rotor_speed_cmds.header.stamp = ros::Time::now();
  return rotor_speed_cmds;
}

bool CustomRotorsInterface::loadParameters() {
  bool check = true;

  check &= pnh_.getParam("inertia_x", inertia_x_);
  check &= pnh_.getParam("inertia_y", inertia_y_);
  check &= pnh_.getParam("inertia_z", inertia_z_);

  inertia_ = Eigen::Matrix3d::Zero();
  inertia_(0, 0) = inertia_x_;
  inertia_(1, 1) = inertia_y_;
  inertia_(2, 2) = inertia_z_;

  check &= pnh_.getParam("body_rates_p_xy", body_rates_p_xy_);
  check &= pnh_.getParam("body_rates_d_xy", body_rates_d_xy_);
  check &= pnh_.getParam("body_rates_p_z", body_rates_p_z_);
  check &= pnh_.getParam("body_rates_d_z", body_rates_d_z_);

  // nothing to do with lqr, just a PD controller
  K_lqr_ = Eigen::MatrixXd::Zero(3, 6);
  K_lqr_(0, 0) = body_rates_p_xy_;
  K_lqr_(1, 1) = body_rates_p_xy_;
  K_lqr_(2, 2) = body_rates_p_z_;
  K_lqr_(0, 3) = body_rates_d_xy_;
  K_lqr_(1, 4) = body_rates_d_xy_;
  K_lqr_(2, 5) = body_rates_d_z_;

  check &= pnh_.getParam("roll_pitch_cont_gain", roll_pitch_cont_gain_);
  check &= pnh_.getParam("arm_length", arm_length_);
  check &= pnh_.getParam("rotor_drag_coeff", rotor_drag_coeff_);
  check &= pnh_.getParam("rotor_thrust_coeff", rotor_thrust_coeff_);
  check &= pnh_.getParam("mass", mass_);
  check &= pnh_.getParam("max_rotor_speed", max_rotor_speed_);

  return check;
}

}  // namespace custom_rotors_interface
