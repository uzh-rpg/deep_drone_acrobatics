#include "acrobatic_trajectory_helper/acrobatic_sequence.h"

#include <acrobatic_trajectory_helper/circle_trajectory_helper.h>
#include <acrobatic_trajectory_helper/heading_trajectory_helper.h>
#include <acrobatic_trajectory_helper/polynomial_trajectory_helper.h>
#include <minimum_jerk_trajectories/RapidTrajectoryGenerator.h>
#include <quadrotor_common/geometry_eigen_conversions.h>
#include <quadrotor_common/math_common.h>
#include <quadrotor_common/parameter_helper.h>
#include <quadrotor_common/trajectory_point.h>
#include <fstream>

namespace fpv_aggressive_trajectories {
AcrobaticSequence::AcrobaticSequence(
    const quadrotor_common::TrajectoryPoint& start_state) {
  printf("Initiated acrobatic sequence\n");
  quadrotor_common::Trajectory init_trajectory;
  quadrotor_common::TrajectoryPoint init_point;
  init_point = start_state;
  init_trajectory.points.push_back(init_point);
  maneuver_list_.push_back(init_trajectory);
}

AcrobaticSequence::~AcrobaticSequence() {}

bool AcrobaticSequence::appendLoops(
    const int n_loops, const double& circle_velocity, const double& radius,
    const Eigen::Vector3d& circle_center_offset,
    const Eigen::Vector3d& circle_center_offset_end, const bool break_at_end,
    const double& traj_sampling_freq) {
  printf("appending loop\n");

  // get start state
  quadrotor_common::TrajectoryPoint init_state =
      maneuver_list_.back().points.back();

  printf(
      "Enter state of loop maneuver: Pos: %.2f, %.2f, %.2f | Vel: %.2f, %.2f, "
      "%.2f\n",
      init_state.position.x(), init_state.position.y(), init_state.position.z(),
      init_state.velocity.x(), init_state.velocity.y(),
      init_state.velocity.z());

  const double exec_loop_rate = traj_sampling_freq;
  const double desired_heading = 0.0;

  const double figure_z_rotation_angle = 0.0;
  //  const double figure_z_rotation_angle = 0.785398163;

  const Eigen::Quaterniond q_W_P = Eigen::Quaterniond(
      Eigen::AngleAxisd(figure_z_rotation_angle, Eigen::Vector3d::UnitZ()));
  double desired_heading_loop = quadrotor_common::wrapMinusPiToPi(
      desired_heading + figure_z_rotation_angle);

  // cirlce center RELATIVE to start position
  const Eigen::Vector3d circle_center =
      init_state.position + q_W_P.inverse() * circle_center_offset;

  printf("circle center: %.2f, %.2f, %.2f\n", circle_center.x(),
         circle_center.y(), circle_center.z());

  const double max_thrust = 9.81 + 1.5 * pow(circle_velocity, 2.0) / radius;
  const double max_roll_pitch_rate = 3.0;

  // Compute Circle trajectory
  printf("compute circle trajectory\n");

  quadrotor_common::Trajectory circle_trajectory =
      acrobatic_trajectory_helper::circles::computeVerticalCircleTrajectory(
          circle_center, figure_z_rotation_angle, radius, circle_velocity,
          M_PI / 2.0, -(3.0 / 2.0 + 2 * (n_loops - 1)) * M_PI, exec_loop_rate);
  acrobatic_trajectory_helper::heading::addConstantHeading(
      desired_heading_loop, &circle_trajectory);

  quadrotor_common::TrajectoryPoint circle_enter_state =
      circle_trajectory.points.front();

  // Start position relative to circle center
  quadrotor_common::TrajectoryPoint start_state;
  start_state = init_state;

  // enter trajectory
  printf("compute enter trajectory\n");
  //  printf("Maximum speed: %.3f, current speed: %.3f\n", 1.1*circle_velocity,
  //  start_state.velocity.norm());
  quadrotor_common::Trajectory enter_trajectory =
      acrobatic_trajectory_helper::polynomials::computeTimeOptimalTrajectory(
          start_state, circle_enter_state, 4,
          1.1 * std::max(start_state.velocity.norm(), circle_velocity),
          max_thrust, 2.0 * max_roll_pitch_rate, exec_loop_rate);
  acrobatic_trajectory_helper::heading::addConstantHeading(
      desired_heading_loop, &enter_trajectory);

  // End position RELATIVE to circle center
  printf("compute exit trajectory\n");

  const Eigen::Vector3d end_pos_P =
      circle_center_offset_end;  // Eigen::Vector3d(circle_center_offset.x(),
  // scircle_center_offset.y(),-circle_center_offset.z()); // nice breaking
  // forward flip
  quadrotor_common::TrajectoryPoint end_state;
  end_state.position = q_W_P * end_pos_P + circle_center;
  end_state.velocity = q_W_P * Eigen::Vector3d(circle_velocity, 0.0, 0.0);

  quadrotor_common::Trajectory exit_trajectory =
      acrobatic_trajectory_helper::polynomials::computeTimeOptimalTrajectory(
          circle_enter_state, end_state, 4, 1.1 * circle_velocity, max_thrust,
          2.0 * max_roll_pitch_rate, exec_loop_rate);
  acrobatic_trajectory_helper::heading::addConstantHeading(
      desired_heading_loop, &exit_trajectory);

  quadrotor_common::Trajectory breaking_trajectory;
  breaking_trajectory.trajectory_type =
      quadrotor_common::Trajectory::TrajectoryType::GENERAL;

  maneuver_list_.push_back(enter_trajectory);
  maneuver_list_.push_back(circle_trajectory);
  maneuver_list_.push_back(exit_trajectory);

  if (break_at_end) {
    // append breaking trajectory at end
    quadrotor_common::TrajectoryPoint end_state_hover;
    end_state_hover.position =
        (end_state.position + Eigen::Vector3d(2.0, 0.0, 0.0));
    end_state_hover.velocity = Eigen::Vector3d::Zero();
    breaking_trajectory =
        acrobatic_trajectory_helper::polynomials::computeTimeOptimalTrajectory(
            end_state, end_state_hover, 4, 1.1 * circle_velocity, 15.0,
            max_roll_pitch_rate, exec_loop_rate);
    acrobatic_trajectory_helper::heading::addConstantHeading(
        0.0, &breaking_trajectory);
    maneuver_list_.push_back(breaking_trajectory);
  }

  return !(enter_trajectory.trajectory_type ==
               quadrotor_common::Trajectory::TrajectoryType::UNDEFINED ||
           circle_trajectory.trajectory_type ==
               quadrotor_common::Trajectory::TrajectoryType::UNDEFINED ||
           exit_trajectory.trajectory_type ==
               quadrotor_common::Trajectory::TrajectoryType::UNDEFINED ||
           breaking_trajectory.trajectory_type ==
               quadrotor_common::Trajectory::TrajectoryType::UNDEFINED);
}

std::list<quadrotor_common::Trajectory> AcrobaticSequence::getManeuverList() {
  return maneuver_list_;
}

}  // namespace fpv_aggressive_trajectories
