#pragma once

#include <quadrotor_common/trajectory.h>

namespace acrobatic_trajectory_helper {

namespace heading {

void addConstantHeading(const double heading,
                        quadrotor_common::Trajectory* trajectory);

void addConstantHeadingRate(const double initial_heading,
                            const double final_heading,
                            quadrotor_common::Trajectory* trajectory);

}  // namespace heading

}  // namespace acrobatic_trajectory_helper
