#include "fpv_aggressive_trajectories/odometry_republisher.h"

#include <quadrotor_common/geometry_eigen_conversions.h>

namespace odometry_republisher {

OdometryRepublisher::OdometryRepublisher() {
  Eigen::Quaterniond q_B_S = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
  Eigen::Vector3d t_B_S = Eigen::Vector3d(0.2, 0.0, 0.0);
  T_B_S_ = rpg::Pose(q_B_S, t_B_S);
  state_estimate_sub_ =
      nh_.subscribe("odometry_in", 1, &OdometryRepublisher::odometryInCallback,
                    this, ros::TransportHints().tcpNoDelay());
  odometry_out_pub_ = nh_.advertise<nav_msgs::Odometry>("odometry_out", 1);
}

OdometryRepublisher::~OdometryRepublisher() {}

void OdometryRepublisher::odometryInCallback(
    const nav_msgs::OdometryConstPtr& msg) {
  nav_msgs::Odometry odometry_out_msg;
  odometry_out_msg.header = msg->header;
  transformOdometry(msg, &odometry_out_msg);
  odometry_out_msg.header.frame_id = "world";
  odometry_out_pub_.publish(odometry_out_msg);
}

void OdometryRepublisher::transformOdometry(
    const nav_msgs::OdometryConstPtr& odom_in, nav_msgs::Odometry* odom_out) {
  // transform odometry estimate to body frame of quadrotor
  Eigen::Quaterniond q_W_S = Eigen::Quaterniond(
      odom_in->pose.pose.orientation.w, odom_in->pose.pose.orientation.x,
      odom_in->pose.pose.orientation.y, odom_in->pose.pose.orientation.z);
  q_W_S.normalize();
  Eigen::Vector3d t_W_S = Eigen::Vector3d(odom_in->pose.pose.position.x,
                                          odom_in->pose.pose.position.y,
                                          odom_in->pose.pose.position.z);
  rpg::Pose T_W_S = rpg::Pose(q_W_S, t_W_S);
  // transform position & orientation
  rpg::Pose T_W_B = T_W_S * T_B_S_.inverse();

  // transform angular velocity
  Eigen::Vector3d omega_body = T_B_S_.inverse().getEigenQuaternion() *
                               Eigen::Vector3d(odom_in->twist.twist.angular.x,
                                               odom_in->twist.twist.angular.y,
                                               odom_in->twist.twist.angular.z);

  // transform linear velocity
  Eigen::Vector3d vlin_world =
      T_B_S_.inverse().getEigenQuaternion() *
          Eigen::Vector3d(odom_in->twist.twist.linear.x,
                          odom_in->twist.twist.linear.y,
                          odom_in->twist.twist.linear.z) +
      omega_body.cross(-T_B_S_.getPosition());

  Eigen::Vector3d vlin_body = q_W_S.inverse() * vlin_world;

  odom_out->pose.pose.position.x = T_W_B.getPosition().x();
  odom_out->pose.pose.position.y = T_W_B.getPosition().y();
  odom_out->pose.pose.position.z = T_W_B.getPosition().z();
  odom_out->pose.pose.orientation.w = T_W_B.getRotation().w();
  odom_out->pose.pose.orientation.x = T_W_B.getRotation().x();
  odom_out->pose.pose.orientation.y = T_W_B.getRotation().y();
  odom_out->pose.pose.orientation.z = T_W_B.getRotation().z();

  odom_out->twist.twist.linear.x = vlin_world.x();
  odom_out->twist.twist.linear.y = vlin_world.y();
  odom_out->twist.twist.linear.z = vlin_world.z();
  odom_out->twist.twist.angular.x = omega_body.x();
  odom_out->twist.twist.angular.y = omega_body.y();
  odom_out->twist.twist.angular.z = omega_body.z();
}

}  // namespace odometry_republisher

int main(int argc, char** argv) {
  ros::init(argc, argv, "odometry_republisher");
  odometry_republisher::OdometryRepublisher odometry_republisher;

  ros::spin();

  return 0;
}
