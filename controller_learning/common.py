import os
import time

def update_mpc_params():
    params = {}
    params['Q_pos_xy'] = 200.0
    params['Q_pos_z'] = 500.0
    params['Q_attitude'] = 50.0
    params['Q_velocity'] = 10.0
    params['R_thrust'] = 0.1
    params['R_pitchroll'] = 0.1
    params['R_yaw'] = 0.1
    params['state_cost_exp'] = 0.0
    params['input_cost_exp'] = 0.0
    params['max_bodyrate_xy'] = 20.0
    params['max_bodyrate_z'] = 5.0
    params['min_thrust'] = 1.0
    params['max_thrust'] = 40.0

    print("Setting control parameters of MPC to:")
    print(params)

    os.system("timeout 1s rosparam set /hummingbird/autopilot/Q_pos_xy " + str(params['Q_pos_xy']))
    os.system("timeout 1s rosparam set /hummingbird/autopilot/Q_pos_z " + str(params['Q_pos_z']))
    os.system("timeout 1s rosparam set /hummingbird/autopilot/Q_attitude " + str(params['Q_attitude']))
    os.system("timeout 1s rosparam set /hummingbird/autopilot/Q_velocity " + str(params['Q_velocity']))
    os.system("timeout 1s rosparam set /hummingbird/autopilot/R_thrust " + str(params['R_thrust']))
    os.system("timeout 1s rosparam set /hummingbird/autopilot/R_pitchroll " + str(params['R_pitchroll']))
    os.system("timeout 1s rosparam set /hummingbird/autopilot/R_yaw " + str(params['R_yaw']))
    os.system("timeout 1s rosparam set /hummingbird/autopilot/state_cost_exponential " + str(params['state_cost_exp']))
    os.system("timeout 1s rosparam set /hummingbird/autopilot/input_cost_exponential " + str(params['input_cost_exp']))
    os.system("timeout 1s rosparam set /hummingbird/autopilot/max_bodyrate_xy " + str(params['max_bodyrate_xy']))
    os.system("timeout 1s rosparam set /hummingbird/autopilot/max_bodyrate_z " + str(params['max_bodyrate_z']))
    os.system("timeout 1s rosparam set /hummingbird/autopilot/min_thrust " + str(params['min_thrust']))
    os.system("timeout 1s rosparam set /hummingbird/autopilot/max_thrust " + str(params['max_thrust']))

    os.system("timeout 1s rostopic pub /hummingbird/autopilot/reload_parameters std_msgs/Empty '{}'")


def setup_sim():
    print("==========================")
    print("     RESET SIMULATION     ")
    print("==========================")

    # set odometry converter back to ground truth state estimate
    # make sure fpv_aggressive_trajectories is not publishing anything!
    # turn off network
    os.system("timeout 1s rostopic pub /hummingbird/fpv_quad_looping/execute_trajectory std_msgs/Bool 'data: false'")
    os.system("timeout 1s  rostopic pub /hummingbird/switch_to_network std_msgs/Bool 'data: false'")
    # after this message, autopilot will automatically go to 'BREAKING' and 'HOVER' state since
    # no control_command_inputs are published any more
    os.system("timeout 1s rostopic pub /switch_odometry std_msgs/Int8 'data: 0'")
    os.system("rosservice call /gazebo/pause_physics")
    print("Unpausing Physics...")
    os.system("rosservice call /gazebo/unpause_physics")
    print("Placing quadrotor...")
    os.system("timeout 1s rostopic pub /hummingbird/autopilot/off std_msgs/Empty")
    os.system("rosservice call /gazebo/set_model_state "
              "'{model_state: { model_name: hummingbird, pose: { position: { x: 0.0, y: 0.0 ,z: 0.2 }, "
              "orientation: {x: 0, y: 0, z: 0.0, w: 1.0 }}, "
              "twist:{ linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 }}, "
              "reference_frame: world } }'")
    time.sleep(2)
    # start quadrotor
    # os.system("timeout 1s rostopic pub /hummingbird/bridge/arm std_msgs/Bool 'True'")
    # os.system("timeout 1s rostopic pub /hummingbird/autopilot/start std_msgs/Empty")


def random_replace():
    reset_success_str = 'rostopic pub /success_reset std_msgs/Empty "{}" -1'
    os.system(reset_success_str)


def initialize_vio():
    # Make sure to use GT odometry in this step
    os.system("timeout 1s rostopic pub /hummingbird/autopilot/off std_msgs/Empty")
    os.system("timeout 1s rostopic pub /switch_odometry std_msgs/Int8 'data: 0'")

    # reset quad to initial position
    os.system("timeout 1s rostopic pub /hummingbird/bridge/arm std_msgs/Bool 'True'")
    print("Start quadrotor")
    os.system("timeout 1s rostopic pub /hummingbird/autopilot/start std_msgs/Empty")
    time.sleep(10)
    # Restart VIO
    os.system("timeout 1s rostopic pub /feature_tracker/restart std_msgs/Bool 'data: true'  ")
    os.system(
        "timeout 1s rostopic pub /hummingbird/autopilot/pose_command geometry_msgs/PoseStamped '{header: {seq: 0, stamp: {secs: 0, nsecs: 0} , frame_id: world}, pose:{position: { x: 0.0, y: 5.0, z: 4.0}, orientation: { x: 0.0, y: 0.0, z: 0.0, w: 1.0} } }'")
    time.sleep(9)
    # TODO: maybe remove this
    # os.system("timeout 1s rostopic pub /switch_odometry std_msgs/Bool 'data: true'")
    return
