#pragma once

#include <ros/ros.h>
#include <Eigen/Core>

#include <autopilot/autopilot_helper.h>
#include <nav_msgs/Odometry.h>
#include <position_controller/position_controller.h>
#include <quadrotor_common/control_command.h>
#include <quadrotor_common/trajectory.h>
#include <quadrotor_msgs/ControlCommand.h>
#include <rpg_mpc/mpc_controller.h>
#include <state_predictor/state_predictor.h>
#include <std_msgs/Empty.h>
#include <visualization_msgs/MarkerArray.h>

#include "fpv_aggressive_trajectories/visualize.h"

namespace fpv_aggressive_trajectories {

class FPVAggressiveTrajectories {
 public:
  FPVAggressiveTrajectories(const ros::NodeHandle& nh,
                            const ros::NodeHandle& pnh);

  FPVAggressiveTrajectories()
      : FPVAggressiveTrajectories(ros::NodeHandle(), ros::NodeHandle("~")) {}

  virtual ~FPVAggressiveTrajectories();

 private:
  enum STATES {
    kOff = 0,
    kComputeTrajectory = 1,
    kExecuteTrajectory = 2,
    kNetworkControl = 3,
    kAutopilot = 4
  };
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  // Subscribers
  ros::Subscriber toggle_experiment_sub_;
  ros::Subscriber odometry_sub_;
  ros::Subscriber odometry_gt_sub_;

  // Publishers
  ros::Publisher control_command_pub_;
  ros::Publisher control_command_gt_pub_;
  ros::Publisher aggressive_end_pub_;
  ros::Publisher marker_pub_ref_;
  ros::Publisher vio_ref_pub_;
  ros::Publisher traj_comp_finish_pub_;

  quadrotor_common::QuadStateEstimate getPredictedStateEstimate(
      const ros::Time& time, const state_predictor::StatePredictor* predictor);

  void computeManeuver();

  quadrotor_common::ControlCommand computeControlCommand(
      const quadrotor_common::QuadStateEstimate& state_estimate,
      ros::Duration* trajectory_execution_left_duration,
      int* trajectories_left_in_queue);

  void publishControlCommand(
      const quadrotor_common::ControlCommand& control_command,
      const quadrotor_common::ControlCommand& control_command_gt);

  void startExecutionCallback(const std_msgs::BoolConstPtr& msg);

  void odometryCallback(const nav_msgs::OdometryConstPtr& msg);

  void odometryGTCallback(const nav_msgs::OdometryConstPtr& msg);

  bool loadParameters();

  std::list<quadrotor_common::Trajectory> trajectory_queue_;
  // will be the same trajectories, but transformed to GT frame
  std::list<quadrotor_common::Trajectory> trajectory_queue_gt_;

  // Parameters
  double desired_heading_;
  double exec_loop_rate_;
  double circle_velocity_;
  int n_loops_;
  nav_msgs::Odometry odometry_gt_;
  STATES state_machine_{kComputeTrajectory};
  ros::Time time_start_trajectory_execution_;
  quadrotor_common::QuadStateEstimate received_state_est_;
  quadrotor_common::QuadStateEstimate received_state_est_gt_;
  bool first_time_in_new_state_;
  double predictive_control_lookahead_ = 2.0;
  bool sent_maneuver_end_msg_ = false;
  int maneuver_counter_ = 0;
  double traj_sampling_freq_ = 50.0;

  // MPC controller variant
  rpg_mpc::MpcController<double> base_controller_ =
      rpg_mpc::MpcController<double>(ros::NodeHandle(), ros::NodeHandle("~"),
                                     "vio_mpc_path");
  rpg_mpc::MpcParams<double> base_controller_params_;
  state_predictor::StatePredictor state_predictor_;

  // Second controller instance that operates on ground truth data
  rpg_mpc::MpcParams<double> base_controller_params_gt_;
  state_predictor::StatePredictor state_predictor_gt_;

  std::shared_ptr<fpv_aggressive_trajectories::Visualizer> visualizer_;
};

}  // namespace fpv_aggressive_trajectories
