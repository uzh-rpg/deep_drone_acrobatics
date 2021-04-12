#include "acrobatic_trajectory_helper/heading_trajectory_helper.h"

#include <quadrotor_common/math_common.h>

namespace acrobatic_trajectory_helper {

namespace heading {

void addConstantHeading(const double heading,
                        quadrotor_common::Trajectory* trajectory) {
  auto iterator(trajectory->points.begin());
  auto iterator_prev(trajectory->points.begin());
  iterator_prev = std::prev(iterator_prev);
  auto iterator_next(trajectory->points.begin());
  iterator_next = std::next(iterator_next);
  auto last_element = trajectory->points.end();
  last_element = std::prev(last_element);
  double time_step;

  for (int i = 0; i < trajectory->points.size(); i++) {
    // do orientation first, since bodyrate conversion will depend on it
    Eigen::Vector3d I_eZ_I(0.0, 0.0, 1.0);
    Eigen::Quaterniond quatDes = Eigen::Quaterniond::FromTwoVectors(
        I_eZ_I, iterator->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81));

    // set full orientation and heading to zero
    Eigen::Quaternion<double> q_heading =
        Eigen::Quaternion<double>(Eigen::AngleAxis<double>(
            heading, Eigen::Matrix<double, 3, 1>::UnitZ()));
    Eigen::Quaternion<double> q_orientation = quatDes * q_heading;
    iterator->orientation = q_orientation;
    iterator->heading = 0.0;  // heading is now absorbed in orientation
    iterator->heading_rate = 0.0;
    iterator->heading_acceleration = 0.0;

    Eigen::Vector3d thrust_1;
    Eigen::Vector3d thrust_2;
    // catch case of first and last element
    if (i == 0) {
      thrust_1 = iterator->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
      time_step =
          (iterator_next->time_from_start - iterator->time_from_start).toSec();
      thrust_2 = iterator_next->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
    } else if (i < trajectory->points.size() - 1) {
      thrust_1 = iterator_prev->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
      time_step =
          (iterator_next->time_from_start - iterator_prev->time_from_start)
              .toSec();
      thrust_2 = iterator_next->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
    } else {
      // at the last point, we extrapolate the acceleration
      thrust_1 = iterator_prev->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81);
      thrust_2 = iterator->acceleration + Eigen::Vector3d(0.0, 0.0, 9.81) +
                 time_step / 2.0 * iterator->jerk;
    }

    thrust_1.normalize();
    thrust_2.normalize();

    Eigen::Vector3d crossProd =
        thrust_1.cross(thrust_2);  // direction of omega, in inertial axes
    Eigen::Vector3d angular_rates_wf = Eigen::Vector3d(0, 0, 0);
    if (crossProd.norm() > 0.0) {
      angular_rates_wf = std::acos(thrust_1.dot(thrust_2)) / time_step *
                         crossProd / crossProd.norm();
    }
    // rotate bodyrates to bodyframe
    iterator->bodyrates = q_orientation.inverse() * angular_rates_wf;

    iterator_prev++;
    iterator++;
    iterator_next++;
  }
}


void addConstantHeadingRate(const double initial_heading,
                            const double final_heading,
                            quadrotor_common::Trajectory* trajectory) {
  if (trajectory->points.size() < 2) {
    return;
  }
  const double delta_angle =
      quadrotor_common::wrapAngleDifference(initial_heading, final_heading);
  const double trajectory_duration =
      (trajectory->points.back().time_from_start -
       trajectory->points.front().time_from_start)
          .toSec();

  const double heading_rate = delta_angle / trajectory_duration;

  std::list<quadrotor_common::TrajectoryPoint>::iterator it;
  for (it = trajectory->points.begin(); it != trajectory->points.end(); it++) {
    const double duration_ratio =
        (it->time_from_start - trajectory->points.front().time_from_start)
            .toSec() /
        trajectory_duration;
    it->heading = initial_heading + duration_ratio * delta_angle;
    it->heading_rate = heading_rate;
    it->heading_acceleration = 0.0;
  }
}



}  // namespace heading

}  // namespace acrobatic_trajectory_helper
