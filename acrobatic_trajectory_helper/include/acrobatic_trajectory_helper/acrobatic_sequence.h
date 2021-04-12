#pragma once

#include <quadrotor_common/trajectory.h>
#include <list>

namespace fpv_aggressive_trajectories {
class AcrobaticSequence {
 public:
  explicit AcrobaticSequence(
      const quadrotor_common::TrajectoryPoint& start_state);

  virtual ~AcrobaticSequence();

  bool appendLoops(const int n_loops, const double& circle_velocity,
                   const double& radius,
                   const Eigen::Vector3d& circle_center_offset,
                   const Eigen::Vector3d& circle_center_offset_end,
                   const bool break_at_end, const double& traj_sampling_freq);

  bool appendBarrelRoll(const int n_loops, const double& circle_velocity, const double& radius,
                        const Eigen::Vector3d& circle_center_offset,
                        const Eigen::Vector3d& circle_center_offset_end,
                        const bool break_at_end);

  bool appendMattyLoop(const int n_loops, const double& circle_velocity, const double& radius,
                       const Eigen::Vector3d& circle_center_offset,
                       const Eigen::Vector3d& circle_center_offset_end);

  std::list<quadrotor_common::Trajectory> getManeuverList();

 private:
  std::list<quadrotor_common::Trajectory> maneuver_list_;
};
}  // namespace fpv_aggressive_trajectories
