# /usr/bin/env python
import rospy
import os
import time
import json
import numpy as np

from std_msgs.msg import Empty
from quadrotor_msgs.msg import AutopilotFeedback
from mav_msgs.msg import Actuators


class TestPerformance(object):
    def __init__(self, test_mode):

        # os.system("rosservice call /gazebo/unpause_physics")

        self.finished = False
        self.airborne = False
        self.received_heartbeat = False
        self.test_mode = test_mode
        self.last_autopilot_state = 0
        self.reference_states = []
        self.state_estimates = []
        self.autopilot_states = []

        self.crash_sub = rospy.Subscriber("/hummingbird/command/motor_speed", Actuators,
                                          self.callback_motor_speed, queue_size=1)
        self.autopilot_sub = rospy.Subscriber("/hummingbird/autopilot/feedback", AutopilotFeedback,
                                              self.callback_autopilot, queue_size=1)

    def callback_motor_speed(self, data):
        self.received_heartbeat = True

    def callback_autopilot(self, data):
        if self.airborne:
            self.reference_states.append([data.reference_state.pose.position.x,
                                          data.reference_state.pose.position.y,
                                          data.reference_state.pose.position.z,
                                          data.reference_state.pose.orientation.w,
                                          data.reference_state.pose.orientation.x,
                                          data.reference_state.pose.orientation.y,
                                          data.reference_state.pose.orientation.z,
                                          data.reference_state.velocity.linear.x,
                                          data.reference_state.velocity.linear.y,
                                          data.reference_state.velocity.linear.z,
                                          data.reference_state.velocity.angular.x,
                                          data.reference_state.velocity.angular.y,
                                          data.reference_state.velocity.angular.z])

            self.state_estimates.append([data.state_estimate.pose.pose.position.x,
                                         data.state_estimate.pose.pose.position.y,
                                         data.state_estimate.pose.pose.position.z,
                                         data.state_estimate.pose.pose.orientation.w,
                                         data.state_estimate.pose.pose.orientation.x,
                                         data.state_estimate.pose.pose.orientation.y,
                                         data.state_estimate.pose.pose.orientation.z,
                                         data.state_estimate.twist.twist.linear.x,
                                         data.state_estimate.twist.twist.linear.y,
                                         data.state_estimate.twist.twist.linear.z,
                                         data.state_estimate.twist.twist.angular.x,
                                         data.state_estimate.twist.twist.angular.y,
                                         data.state_estimate.twist.twist.angular.z])

            self.autopilot_states.append(data.autopilot_state)

            # detect crash
            if data.state_estimate.pose.pose.position.z < 0.1:
                self.finished = True
            # Why finished with large X?
            if data.autopilot_state == 2 and self.last_autopilot_state == 9 and data.state_estimate.pose.pose.position.x > 2.0:
                print("I am done")
                self.finished = True

            self.last_autopilot_state = data.autopilot_state

    def run(self, num_iterations, params, idx, handtuned_params):
        # while not rospy.is_shutdown():
        # so far, this file is only for testing!
        self.finished = False
        self.received_heartbeat = False
        time_before_crash = []

        del self.reference_states[:]
        del self.state_estimates[:]
        del self.autopilot_states[:]

        for i in range(1, num_iterations + 1, 1):
            self.finished = False

            print("Replacing quad for new run...")
            os.system("rosservice call /gazebo/unpause_physics")
            os.system("timeout 1s rostopic pub /hummingbird/autopilot/off std_msgs/Empty")
            # reset quad to initial position
            os.system(
                "rosservice call /gazebo/set_model_state '{model_state: { model_name: hummingbird, pose: { position: { x: 0.0, y: 0.0 ,z: 0.2 }, orientation: {x: 0, y: 0, z: 0.0, w: 1.0 } }, twist:{ linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }'")
            # os.system(
            #     "rosservice call /gazebo/set_model_state '{model_state: { model_name: hummingbird, pose: { position: { x: 0.0, y: 0.0 ,z: 0.2 }, orientation: {x: 0, y: 0, z: 0.38268343236, w: 0.92387953251 } }, twist:{ linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }'")

            time.sleep(5)
            # start quadrotor
            os.system("timeout 1s rostopic pub /hummingbird/bridge/arm std_msgs/Bool 'True'")
            print("Start quadrotor")
            os.system("timeout 1s rostopic pub /hummingbird/autopilot/start std_msgs/Empty")
            time.sleep(10)
            self.airborne = True
            # some initialization motion for VIO
            os.system("timeout 1s rostopic pub /feature_tracker/restart std_msgs/Bool 'data: true'  ")
            os.system("timeout 1s rostopic pub /hummingbird/autopilot/pose_command geometry_msgs/PoseStamped '{header: {seq: 0, stamp: {secs: 0, nsecs: 0} , frame_id: world}, pose:{position: { x: 0.0, y: 5.0, z: 2.0}, orientation: { x: 0.0, y: 0.0, z: 0.0, w: 1.0} } }'")
            time.sleep(10)
            os.system(
                "timeout 1s rostopic pub /hummingbird/autopilot/pose_command geometry_msgs/PoseStamped '{header: {seq: 0, stamp: {secs: 0, nsecs: 0} , frame_id: world}, pose:{position: { x: 0.0, y: 0.0, z: 2.0}, orientation: { x: 0.0, y: 0.0, z: 0.0, w: 1.0} } }'")
            time.sleep(10)
            # Switch to vision-based estimate
            os.system("timeout 1s rostopic pub /switch_odometry std_msgs/Bool 'data: true'")

            # Network enabled
            print("Enable flight")
            os.system("timeout 1s rostopic pub /hummingbird/fpv_quad_looping/execute_trajectory std_msgs/Empty")

            start_time = time.time()

            while (time.time() - start_time < 30 and self.finished == False):
                time.sleep(0.5)
                # x_disturbance = np.sin(time.time() - start_time)
                # y_disturbance = np.sin(time.time() - start_time)
                # z_disturbance = np.sin(time.time() - start_time)
                # duration = 100000000  # duration in nanoseconds
                # os.system("rosservice call /gazebo/apply_body_wrench "
                #           + "'{body_name: \"hummingbird::hummingbird/base_link\", "
                #           + "wrench: { force: { x: " + str(x_disturbance)
                #           + ", y: " + str(y_disturbance)
                #           + ", z: " + str(z_disturbance) + " } }, start_time: 0, duration: " + str(duration) + " }' ")

            self.airborne = False
            time_before_crash.append(time.time() - start_time)
            print("Experiment finished")

        # save log to json file
        result_dict = {'reference_state': self.reference_states, 'state_estimate': self.state_estimates,
                       'autopilot_states': self.autopilot_states}
        result_dict.update(params)
        if handtuned_params:
            mark_handtuned = "handtuned_"
        else:
            mark_handtuned = ""
        # results_fname = '/home/elia/Desktop/controller_logs/test_' + mark_handtuned + str(idx) + '.json'
        # print("Saving data to %s" % results_fname)
        # # print(result_dict)
        # with open(results_fname, 'w') as outfile:
        #     json.dump(result_dict, outfile)

        return self.received_heartbeat
