#pragma once
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

#include "quadrotor_common/trajectory.h"

namespace fpv_aggressive_trajectories {
class Visualizer {
 public:
  Visualizer(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);

  Visualizer() : Visualizer(ros::NodeHandle(), ros::NodeHandle("~")) {}

  virtual ~Visualizer();

  void visualizeReferenceTrajectory(
      const quadrotor_common::Trajectory* trajectory);

  void visualizeTrajectories(
      const std::list<quadrotor_common::Trajectory>& maneuver_list);

  void create_vehicle_markers(int num_rotors, float arm_len, float body_width,
                              float body_height);

  void displayQuadrotor();

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Publisher bodyrates_viz_pub_;
  ros::Publisher marker_ref_pub_;
  ros::Publisher marker_pub_;
  ros::Publisher vehicle_marker_pub_;

  std::shared_ptr<visualization_msgs::MarkerArray> vehicle_marker_;
};
}  // namespace fpv_aggressive_trajectories