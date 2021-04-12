#include "fpv_aggressive_trajectories/fpv_aggressive_trajectories.h"

#include <string>

#include <minimum_jerk_trajectories/RapidTrajectoryGenerator.h>
#include <quadrotor_common/geometry_eigen_conversions.h>
#include <quadrotor_common/parameter_helper.h>
#include <quadrotor_common/trajectory_point.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include "acrobatic_trajectory_helper/acrobatic_sequence.h"

namespace fpv_aggressive_trajectories {

FPVAggressiveTrajectories::FPVAggressiveTrajectories(const ros::NodeHandle& nh,
                                                     const ros::NodeHandle& pnh)
    : nh_(nh),
      pnh_(pnh),
      state_predictor_(nh_, pnh_),
      state_predictor_gt_(nh_, pnh_) {
  if (!loadParameters()) {
    ROS_ERROR("[%s] Failed to load all parameters",
              ros::this_node::getName().c_str());
    ros::shutdown();
  }

  base_controller_params_.loadParameters(nh_);
  base_controller_params_gt_.loadParameters(nh_);

  visualizer_ = std::make_shared<fpv_aggressive_trajectories::Visualizer>(
      fpv_aggressive_trajectories::Visualizer(nh_, pnh_));

  toggle_experiment_sub_ =
      nh_.subscribe("fpv_quad_looping/execute_trajectory", 1,
                    &FPVAggressiveTrajectories::startExecutionCallback, this);
  odometry_sub_ = nh_.subscribe("state_estimate", 1,
                                &FPVAggressiveTrajectories::odometryCallback,
                                this, ros::TransportHints().tcpNoDelay());
  odometry_gt_sub_ =
      nh_.subscribe("ground_truth/odometry", 1,
                    &FPVAggressiveTrajectories::odometryGTCallback, this,
                    ros::TransportHints().tcpNoDelay());
  control_command_pub_ = nh_.advertise<quadrotor_msgs::ControlCommand>(
      "autopilot/control_command_input", 1);
  control_command_gt_pub_ = nh_.advertise<quadrotor_msgs::ControlCommand>(
      "/hummingbird/control_command_gt", 1);
  marker_pub_ref_ =
      nh_.advertise<visualization_msgs::MarkerArray>("reference_trajectory", 1);
  aggressive_end_pub_ = nh_.advertise<std_msgs::Bool>("switch_to_network", 1);
  vio_ref_pub_ =
      nh_.advertise<quadrotor_msgs::TrajectoryPoint>("vio_reference", 1);
  traj_comp_finish_pub_ =
      nh_.advertise<std_msgs::Bool>("trajectory_computation_finish", 1);
}

FPVAggressiveTrajectories::~FPVAggressiveTrajectories() {}

quadrotor_common::QuadStateEstimate
FPVAggressiveTrajectories::getPredictedStateEstimate(
    const ros::Time& time, const state_predictor::StatePredictor* predictor) {
  return predictor->predictState(time);
}

void FPVAggressiveTrajectories::startExecutionCallback(
    const std_msgs::BoolConstPtr& msg) {
  ROS_INFO("Received startExecutionCallback message!");
  if (msg->data) {
    // publish the reference of the current state already before maneuver
    // computation
    quadrotor_common::QuadStateEstimate odom_start_computation =
        received_state_est_;
    quadrotor_common::TrajectoryPoint start_state;
    start_state.position = odom_start_computation.position;
    start_state.velocity = odom_start_computation.velocity;
    start_state.orientation = odom_start_computation.orientation;
    quadrotor_msgs::TrajectoryPoint ref_msg = start_state.toRosMessage();
    vio_ref_pub_.publish(ref_msg);
    computeManeuver();
  }
  if (!msg->data) {
    ROS_INFO("Restarting experiment, going to OFF state!");
    state_machine_ = STATES::kOff;
  }
}

void FPVAggressiveTrajectories::computeManeuver() {
  ros::Time time_start_computation = ros::Time::now();
  trajectory_queue_.clear();
  trajectory_queue_gt_.clear();
  sent_maneuver_end_msg_ = false;

  quadrotor_common::QuadStateEstimate odom_start_computation =
      received_state_est_;
  quadrotor_common::QuadStateEstimate odom_start_computation_gt =
      received_state_est_gt_;

  nav_msgs::Odometry odom_gt_start_computation = odometry_gt_;

  quadrotor_common::TrajectoryPoint start_state;
  start_state.position = odom_start_computation.position;
  start_state.velocity = odom_start_computation.velocity;

  quadrotor_common::TrajectoryPoint start_state_gt;
  start_state_gt.position = odom_start_computation_gt.position;
  start_state_gt.velocity = odom_start_computation_gt.velocity;

  AcrobaticSequence acrobatic_sequence(start_state);
  AcrobaticSequence acrobatic_sequence_gt(start_state_gt);

  bool success = true;
  Eigen::Vector3d offset_circle_from_start = Eigen::Vector3d(4.0, 0.0, 0.0);
  Eigen::Vector3d offset_circle_from_end = Eigen::Vector3d(-1.0, 0.0, 1.5);
  // standard loop
  success = success && acrobatic_sequence.appendLoops(
                           1, 4.5, 1.5, offset_circle_from_start,
                           offset_circle_from_end, true, traj_sampling_freq_);
  // matty loop
  // success = success && acrobatic_sequence.appendMattyLoop(1, 4.5, 1.5, offset_circle_from_start, offset_circle_from_end);
  // barrel roll
  // success = success && acrobatic_sequence.appendBarrelRoll(1, 4.5, 1.5, offset_circle_from_start, offset_circle_from_end, true);

  visualizer_->visualizeTrajectories(acrobatic_sequence.getManeuverList());

  if (success) {
    for (const quadrotor_common::Trajectory& trajectory :
         acrobatic_sequence.getManeuverList()) {
      trajectory_queue_.push_back(trajectory);
    }
    for (const quadrotor_common::Trajectory& trajectory :
         acrobatic_sequence_gt.getManeuverList()) {
      trajectory_queue_gt_.push_back(trajectory);
    }
    state_machine_ = STATES::kExecuteTrajectory;
    first_time_in_new_state_ = true;
    ROS_INFO("Maneuver computation successful!");
  } else {
    ROS_ERROR("Maneuver computation failed! Will not execute trajectory.");
  }
  std_msgs::Bool bool_msg;
  bool_msg.data = success;
  traj_comp_finish_pub_.publish(bool_msg);
  ROS_INFO("Maneuver computation took %.4f seconds.",
           (ros::Time::now() - time_start_computation).toSec());
}

void FPVAggressiveTrajectories::odometryCallback(
    const nav_msgs::OdometryConstPtr& msg) {
  received_state_est_ = quadrotor_common::QuadStateEstimate(*msg);
  received_state_est_gt_ = quadrotor_common::QuadStateEstimate(odometry_gt_);

  // both velocity estimates are expressed in bodyframe
  received_state_est_.transformVelocityToWorldFrame();
  received_state_est_gt_.transformVelocityToWorldFrame();

  // Push received state estimate into predictor
  state_predictor_.updateWithStateEstimate(received_state_est_);
  state_predictor_gt_.updateWithStateEstimate(received_state_est_gt_);

  if (state_machine_ == STATES::kExecuteTrajectory) {
    quadrotor_common::ControlCommand control_cmd;
    quadrotor_common::ControlCommand control_cmd_gt;
    ros::Time wall_time_now = ros::Time::now();
    double control_command_delay = 0.05;  // TODO: read from parameter file
    ros::Time command_execution_time =
        wall_time_now + ros::Duration(control_command_delay);
    quadrotor_common::QuadStateEstimate predicted_state =
        getPredictedStateEstimate(command_execution_time, &state_predictor_);
    quadrotor_common::QuadStateEstimate predicted_state_gt =
        getPredictedStateEstimate(command_execution_time, &state_predictor_gt_);

    ros::Duration trajectory_execution_left_duration(0.0);
    int trajectories_left_in_queue = 0;

    ros::Duration trajectory_execution_left_duration_gt(0.0);

    const ros::Time start_control_command_computation = ros::Time::now();

    control_cmd = computeControlCommand(predicted_state,
                                        &trajectory_execution_left_duration,
                                        &trajectories_left_in_queue);
    control_cmd_gt = control_cmd;
    control_cmd_gt.collective_thrust = 0.0;
    control_cmd.timestamp = wall_time_now;
    control_cmd.expected_execution_time = command_execution_time;
    const ros::Duration control_computation_time =
        ros::Time::now() - start_control_command_computation;

    publishControlCommand(control_cmd, control_cmd_gt);
  } else if (state_machine_ == STATES::kNetworkControl) {
    // we publish some dummy message here to prevent autopilot from going into
    // HOVER
    quadrotor_common::ControlCommand control_cmd;
    control_cmd.armed = true;
    control_cmd.collective_thrust =
        5.0;  // on purpose low thrust value to detect if this control command
              // was executed
    control_cmd.bodyrates = Eigen::Vector3d::Zero();
    control_cmd.control_mode = quadrotor_common::ControlMode::BODY_RATES;
    quadrotor_common::ControlCommand control_cmd_gt = control_cmd;
    publishControlCommand(control_cmd, control_cmd_gt);
  }
}

void FPVAggressiveTrajectories::odometryGTCallback(
    const nav_msgs::OdometryConstPtr& msg) {
  odometry_gt_ = *msg;
  visualizer_->displayQuadrotor();
}

quadrotor_common::ControlCommand
FPVAggressiveTrajectories::computeControlCommand(
    const quadrotor_common::QuadStateEstimate& state_estimate,
    ros::Duration* trajectory_execution_left_duration,
    int* trajectories_left_in_queue) {
  quadrotor_common::TrajectoryPoint reference_state_;
  quadrotor_common::Trajectory reference_trajectory_;

  ros::Time time_now = ros::Time::now();
  if (first_time_in_new_state_) {
    first_time_in_new_state_ = false;
    time_start_trajectory_execution_ = time_now;
  }

  if (trajectory_queue_.empty()) {
    ROS_ERROR("[%s] Trajectory queue was unexpectedly emptied!",
              pnh_.getNamespace().c_str());
    *trajectory_execution_left_duration = ros::Duration(0.0);
    *trajectories_left_in_queue = 0;
    reference_trajectory_ = quadrotor_common::Trajectory(reference_state_);
    quadrotor_msgs::TrajectoryPoint ref_msg =
        reference_trajectory_.points.front().toRosMessage();
    vio_ref_pub_.publish(ref_msg);
    return base_controller_.run(state_estimate, reference_trajectory_,
                                base_controller_params_);
  }

  // enable the NW a bit early
  double a_bit_early = 0.0;
  if (trajectory_queue_.size() == 1 && !sent_maneuver_end_msg_ &&
      (time_now - time_start_trajectory_execution_ +
       ros::Duration(a_bit_early)) >
          trajectory_queue_.front().points.back().time_from_start) {
    ROS_INFO("Enable network!");
    sent_maneuver_end_msg_ = true;
    std_msgs::Bool bool_msg;
    bool_msg.data = true;
    aggressive_end_pub_.publish(bool_msg);
  }

  if ((time_now - time_start_trajectory_execution_) >
      trajectory_queue_.front().points.back().time_from_start) {
    if (trajectory_queue_.size() == 1) {
      ROS_WARN("This was the last trajectory! t = %.4f\n",
               (time_now - time_start_trajectory_execution_).toSec());
      // This was the last trajectory in the queue -> go back to hover
      reference_state_ = trajectory_queue_.back().points.back();
      *trajectory_execution_left_duration = ros::Duration(0.0);
      *trajectories_left_in_queue = 0;
      trajectory_queue_.pop_front();
      state_machine_ = STATES::kAutopilot;  // kNetworkControl;
      reference_trajectory_ = quadrotor_common::Trajectory(reference_state_);
      quadrotor_msgs::TrajectoryPoint ref_msg =
          reference_trajectory_.points.front().toRosMessage();
      vio_ref_pub_.publish(ref_msg);
      return base_controller_.run(state_estimate, reference_trajectory_,
                                  base_controller_params_);
    } else {
      time_start_trajectory_execution_ +=
          trajectory_queue_.front().points.back().time_from_start;
      trajectory_queue_.pop_front();
    }
  }

  // Time from trajectory start and corresponding reference state.
  const ros::Duration dt = time_now - time_start_trajectory_execution_;
  reference_state_ = trajectory_queue_.front().getStateAtTime(dt);

  // New trajectory where we fill in our lookahead horizon.
  reference_trajectory_ = quadrotor_common::Trajectory();
  reference_trajectory_.trajectory_type =
      quadrotor_common::Trajectory::TrajectoryType::GENERAL;

  bool lookahead_reached(false);  // Boolean break flag
  // Time wrap if lookahead spans multiple trajectories:
  double time_wrapover(0.0);

  for (auto trajectory : trajectory_queue_) {
    for (auto point : trajectory.points) {
      // Check wether we reached our lookahead.
      // Use boolean flag to also break the outer loop.
      if (point.time_from_start.toSec() >
          (dt.toSec() - time_wrapover + predictive_control_lookahead_)) {
        lookahead_reached = true;
        break;
      }
      // Add a point if the time corresponds to a sample on the lookahead.
      if (point.time_from_start.toSec() > (dt.toSec() - time_wrapover)) {
        // check if two trajectory points are the same...
        if (reference_trajectory_.points.size() > 1) {
          point.time_from_start += ros::Duration(time_wrapover);
          reference_trajectory_.points.push_back(point);
        } else {
          // this is the first point of the reference trajectory
          reference_trajectory_.points.push_back(point);
        }
      }
    }
    if (lookahead_reached) break;  // Break on boolean flag.
    // Sum up the wrap-over time if lookahead spans multiple trajectories.
    time_wrapover += trajectory.points.back().time_from_start.toSec();
  }

  *trajectory_execution_left_duration =
      trajectory_queue_.front().points.back().time_from_start -
      reference_state_.time_from_start;
  if (trajectory_queue_.size() > 1) {
    std::list<quadrotor_common::Trajectory>::const_iterator it;
    for (it = std::next(trajectory_queue_.begin(), 1);
         it != trajectory_queue_.end(); it++) {
      *trajectory_execution_left_duration += it->points.back().time_from_start;
    }
  }
  *trajectories_left_in_queue = trajectory_queue_.size();

  // handle case of empty reference_trajectory
  if (reference_trajectory_.points.empty()) {
    ROS_WARN("Empty reference trajectory!");
    *trajectory_execution_left_duration = ros::Duration(0.0);
    *trajectories_left_in_queue = 0;
    reference_trajectory_ = quadrotor_common::Trajectory(reference_state_);
    quadrotor_msgs::TrajectoryPoint ref_msg =
        reference_trajectory_.points.front().toRosMessage();
    vio_ref_pub_.publish(ref_msg);
    return base_controller_.run(state_estimate, reference_trajectory_,
                                base_controller_params_);
  }

  // visualize reference trajectory
  visualizer_->visualizeReferenceTrajectory(&reference_trajectory_);

  quadrotor_msgs::TrajectoryPoint ref_msg =
      reference_trajectory_.points.front().toRosMessage();
  vio_ref_pub_.publish(ref_msg);

  quadrotor_common::ControlCommand ctrl_cmd = base_controller_.run(
      state_estimate, reference_trajectory_, base_controller_params_);
  return ctrl_cmd;
}

void FPVAggressiveTrajectories::publishControlCommand(
    const quadrotor_common::ControlCommand& control_command,
    const quadrotor_common::ControlCommand& control_command_gt) {
  if (state_machine_ == STATES::kOff) {
    return;
  }
  quadrotor_msgs::ControlCommand control_cmd_msg;
  control_cmd_msg = control_command.toRosMessage();
  control_command_pub_.publish(control_cmd_msg);
  state_predictor_.pushCommandToQueue(control_command);
}

bool FPVAggressiveTrajectories::loadParameters() {
  if (!quadrotor_common::getParam("desired_yaw_P", desired_heading_, 0.0))
    return false;

  if (!quadrotor_common::getParam("loop_rate", exec_loop_rate_, 55.0))
    return false;

  if (!quadrotor_common::getParam("circle_velocity", circle_velocity_, 1.0))
    return false;

  if (!quadrotor_common::getParam("n_loops", n_loops_, 1)) return false;

  return true;
}

}  // namespace fpv_aggressive_trajectories

int main(int argc, char** argv) {
  ros::init(argc, argv, "fpv_aggressive_trajectories");
  fpv_aggressive_trajectories::FPVAggressiveTrajectories
      fpv_aggressive_trajectories;

  ros::MultiThreadedSpinner spinner(2);
  spinner.spin();

  return 0;
}
