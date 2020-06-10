#include <custom_rotors_interface/custom_rotors_interface.h>
#include <memory>

int main(int argc, char **argv) {
  ros::init(argc, argv, "custom_rotors_interface");

  custom_rotors_interface::CustomRotorsInterface custom_rotors_interface;

  ros::spin();
  return 0;
}
