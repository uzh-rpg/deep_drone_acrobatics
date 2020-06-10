#!/usr/bin/env python

import rospy
from TestPerformance import TestPerformance
import os, datetime
import numpy as np
import random
import glob
import yaml

import sys
import time
import subprocess
# from imutils import paths
# import gflags
import json
import yaml


class TestRoutine(object):
    def __init__(self):
        full_param_name = rospy.search_param('mpc_dir')
        mpc_dir = rospy.get_param(full_param_name)
        mpc_dir = mpc_dir['test_performance']['mpc_dir']
        mpc_dir = os.path.join(mpc_dir, 'parameters/default.yaml')
        self.mpc_params = os.path.abspath(mpc_dir)
        print(self.mpc_params)
        self.test_mode = 'mpc'  # 'pid' or 'mpc'
        self.sampling_mode = 'continuous'  # 'grid' or 'continuous'
        self.handtuned_params = True
        self.params = {}
        self.idx_array = None
        self.n_combinations = 1 # will be overwritten in case of grid sampling

    def init_params(self):
        print("Initializing parameter distribution")
        if self.test_mode == 'pid':
            # PID params
            self.kdxy = np.arange(20, 0, -2)
            self.kdz = np.arange(20, 0, -2)
            self.kpxy = np.arange(20, 0, -2)
            self.kpz = np.arange(20, 0, -2)
            self.krp = np.arange(20, 0, -2)
            self.n_combinations = self.kdxy.size * self.kdz.size * self.kpxy.size * self.kpz.size * self.krp.size

            self.idx_array = np.zeros((self.n_combinations, 5))
            for i in range(self.n_combinations):
                self.idx_array[i, 0] = i % self.kdxy.size
                self.idx_array[i, 1] = (i // self.kdxy.size) % self.kdz.size
                self.idx_array[i, 2] = (i // (self.kdxy.size * self.kdz.size)) % self.kpxy.size
                self.idx_array[i, 3] = (i // (self.kdxy.size * self.kdz.size * self.kpxy.size)) % self.kpz.size
                self.idx_array[i, 4] = (i // (
                        self.kdxy.size * self.kdz.size * self.kpxy.size * self.kpz.size)) % self.krp.size

        elif self.test_mode == 'mpc':
            if self.sampling_mode == 'grid':
                # MPC params
                self.Q_pos_xy = np.array([0.1, 10.0, 100.0, 1000.0])
                self.Q_pos_z = np.array([0.1, 10.0, 100.0, 1000.0])
                self.Q_attitude = np.array([0.1, 10.0, 100.0, 1000.0])
                self.Q_velocity = np.array([0.1, 10.0, 100.0, 1000.0])

                self.R_thrust = np.array([0.1, 1.0, 10.0])
                self.R_pitchroll = np.array([0.1, 1.0, 10.0])
                self.R_yaw = np.array([0.1, 1.0, 10.0])

                self.state_cost_exponential = np.array([0, 0.1, 1.0, 10.0])
                self.input_cost_exponential = np.array([0, 0.1, 1.0, 10.0])

                self.n_combinations = self.Q_pos_xy.size * self.Q_pos_z.size * self.Q_attitude.size * self.Q_velocity.size \
                                      * self.R_thrust.size * self.R_pitchroll.size * self.R_yaw.size \
                                      * self.state_cost_exponential.size * self.input_cost_exponential.size

                self.idx_array = np.zeros((self.n_combinations, 9))
                for i in range(self.n_combinations):
                    self.idx_array[i, 0] = i % self.Q_pos_xy.size
                    self.idx_array[i, 1] = (i // self.Q_pos_xy.size) % self.Q_pos_z.size
                    self.idx_array[i, 2] = (i // (self.Q_pos_xy.size * self.Q_pos_z.size)) % self.Q_attitude.size
                    self.idx_array[i, 3] = (i // (
                            self.Q_pos_xy.size * self.Q_pos_z.size * self.Q_attitude.size)) % self.Q_velocity.size
                    self.idx_array[i, 4] = (i // (
                            self.Q_pos_xy.size * self.Q_pos_z.size * self.Q_attitude.size * self.Q_velocity.size)) % self.R_thrust.size
                    self.idx_array[i, 5] = (i // (
                            self.Q_pos_xy.size * self.Q_pos_z.size * self.Q_attitude.size * self.Q_velocity.size * self.R_thrust.size)) % self.R_pitchroll.size
                    self.idx_array[i, 6] = (i // (
                            self.Q_pos_xy.size * self.Q_pos_z.size * self.Q_attitude.size * self.Q_velocity.size * self.R_thrust.size * self.R_pitchroll.size)) % self.R_yaw.size
                    self.idx_array[i, 7] = (i // (
                            self.Q_pos_xy.size * self.Q_pos_z.size * self.Q_attitude.size * self.Q_velocity.size * self.R_thrust.size * self.R_pitchroll.size * self.R_yaw.size)) % self.state_cost_exponential.size
                    self.idx_array[i, 8] = (i // (
                            self.Q_pos_xy.size * self.Q_pos_z.size * self.Q_attitude.size * self.Q_velocity.size * self.R_thrust.size * self.R_pitchroll.size * self.R_yaw.size * self.state_cost_exponential.size)) % self.input_cost_exponential.size

            elif self.sampling_mode == 'continuous':
                pass

    def set_params(self, curr_idx):
        self.params.clear()

        if self.test_mode == 'pid':
            pxy_error_max = 0.6
            pz_error_max = 0.3
            vxy_error_max = 1.0
            vz_error_max = 0.75
            yaw_error_max = 0.7

            self.params['kdxy'] = self.kdxy[int(self.idx_array[curr_idx, 0])]
            self.params['kdz'] = self.kdz[int(self.idx_array[curr_idx, 1])]
            self.params['kpxy'] = self.kpxy[int(self.idx_array[curr_idx, 2])]
            self.params['kpz'] = self.kpz[int(self.idx_array[curr_idx, 3])]
            self.params['krp'] = self.krp[int(self.idx_array[curr_idx, 4])]
            self.params['kyaw'] = 5.0

            print("Setting control parameters to:")
            print(self.params)

            self.params['pxy_error_max'] = pxy_error_max
            self.params['pz_error_max'] = pz_error_max
            self.params['vxy_error_max'] = vxy_error_max
            self.params['vz_error_max'] = vz_error_max
            self.params['yaw_error_max'] = yaw_error_max

            os.system("timeout 1s rosparam set /hummingbird/autopilot/position_controller/k_drag_x 0.0")
            os.system("timeout 1s rosparam set /hummingbird/autopilot/position_controller/k_drag_y 0.0")
            os.system("timeout 1s rosparam set /hummingbird/autopilot/position_controller/k_drag_z 0.0")
            os.system("timeout 1s rosparam set /hummingbird/autopilot/position_controller/k_thrust_horz 0.0")
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/position_controller/kdxy " + str(self.params['kdxy']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/position_controller/kdz " + str(self.params['kdz']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/position_controller/kpxy " + str(self.params['kpxy']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/position_controller/kpz " + str(self.params['kpz']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/position_controller/krp " + str(self.params['krp']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/position_controller/kyaw " + str(self.params['kyaw']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/position_controller/perform_aerodynamics_compensation false")
            os.system("timeout 1s rosparam set /hummingbird/autopilot/position_controller/pxy_error_max " + str(
                self.params['pxy_error_max']))
            os.system("timeout 1s rosparam set /hummingbird/autopilot/position_controller/pz_error_max " + str(
                self.params['pz_error_max']))
            os.system("timeout 1s rosparam set /hummingbird/autopilot/position_controller/use_rate_mode true")
            os.system("timeout 1s rosparam set /hummingbird/autopilot/position_controller/vxy_error_max " + str(
                self.params['vxy_error_max']))
            os.system("timeout 1s rosparam set /hummingbird/autopilot/position_controller/vz_error_max " + str(
                self.params['vz_error_max']))
            os.system("timeout 1s rosparam set /hummingbird/autopilot/position_controller/yaw_error_max " + str(
                self.params['yaw_error_max']))

        elif self.test_mode == 'mpc':
            max_bodyrate_xy = 20.0
            max_bodyrate_z = 5.0
            min_thrust = 1.0
            max_thrust = 40.0

            min_val = 0.0001
            max_val = 500.0

            if self.sampling_mode == 'grid':
                self.params['Q_pos_xy'] = self.Q_pos_xy[int(self.idx_array[curr_idx, 0])]
                self.params['Q_pos_z'] = self.Q_pos_z[int(self.idx_array[curr_idx, 1])]
                self.params['Q_attitude'] = self.Q_attitude[int(self.idx_array[curr_idx, 2])]
                self.params['Q_velocity'] = self.Q_velocity[int(self.idx_array[curr_idx, 3])]
                self.params['R_thrust'] = self.R_thrust[int(self.idx_array[curr_idx, 4])]
                self.params['R_pitchroll'] = self.R_pitchroll[int(self.idx_array[curr_idx, 5])]
                self.params['R_yaw'] = self.R_yaw[int(self.idx_array[curr_idx, 6])]
                self.params['state_cost_exponential'] = self.state_cost_exponential[int(self.idx_array[curr_idx, 7])]
                self.params['input_cost_exponential'] = self.input_cost_exponential[int(self.idx_array[curr_idx, 8])]
            elif self.sampling_mode == 'continuous':
                self.params['Q_pos_xy'] = np.random.uniform(low=min_val, high=max_val)
                self.params['Q_pos_z'] = np.random.uniform(low=min_val, high=max_val)
                self.params['Q_attitude'] = np.random.uniform(low=min_val, high=max_val)
                self.params['Q_velocity'] = np.random.uniform(low=min_val, high=max_val)
                self.params['R_thrust'] = np.random.uniform(low=min_val, high=max_val)
                self.params['R_pitchroll'] = np.random.uniform(low=min_val, high=max_val)
                self.params['R_yaw'] = np.random.uniform(low=min_val, high=max_val)
                self.params['state_cost_exponential'] = np.random.uniform(low=min_val, high=max_val)
                self.params['input_cost_exponential'] = np.random.uniform(low=min_val, high=max_val)

            print("Setting control parameters to:")
            print(self.params)

            self.params['max_bodyrate_xy'] = max_bodyrate_xy
            self.params['max_bodyrate_z'] = max_bodyrate_z
            self.params['min_thrust'] = min_thrust
            self.params['max_thrust'] = max_thrust

            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/Q_pos_xy " + str(self.params['Q_pos_xy']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/Q_pos_z " + str(self.params['Q_pos_z']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/Q_attitude " + str(self.params['Q_attitude']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/Q_velocity " + str(self.params['Q_velocity']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/R_thrust " + str(self.params['R_thrust']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/R_pitchroll " + str(self.params['R_pitchroll']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/R_yaw " + str(self.params['R_yaw']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/state_cost_exponential " + str(
                    self.params['state_cost_exponential']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/input_cost_exponential " + str(
                    self.params['input_cost_exponential']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/max_bodyrate_xy " + str(self.params['max_bodyrate_xy']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/max_bodyrate_z " + str(self.params['max_bodyrate_z']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/min_thrust " + str(self.params['min_thrust']))
            os.system(
                "timeout 1s rosparam set /hummingbird/autopilot/max_thrust " + str(self.params['max_thrust']))

        else:
            assert False, "unknown test mode"

    def set_handtuned_params(self):
        self.params.clear()
        assert self.test_mode == "mpc"

        max_bodyrate_xy = 30.0
        max_bodyrate_z = 10.0
        min_thrust = 1.0
        max_thrust = 30.0

        # read yaml file
        yaml_path = self.mpc_params
        with open(yaml_path, 'r') as stream:
            try:
                yaml_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.params['Q_pos_xy'] = yaml_dict['Q_pos_xy']
        self.params['Q_pos_z'] = yaml_dict['Q_pos_z']
        self.params['Q_attitude'] = yaml_dict['Q_attitude']
        self.params['Q_velocity'] = yaml_dict['Q_velocity']
        self.params['R_thrust'] = yaml_dict['R_thrust']
        self.params['R_pitchroll'] = yaml_dict['R_pitchroll']
        self.params['R_yaw'] = yaml_dict['R_yaw']
        self.params['state_cost_exponential'] = yaml_dict['state_cost_exponential']
        self.params['input_cost_exponential'] = yaml_dict['input_cost_exponential']
        self.params['max_bodyrate_xy'] = yaml_dict['max_bodyrate_xy']
        self.params['max_bodyrate_z'] = yaml_dict['max_bodyrate_z']
        self.params['min_thrust'] = yaml_dict['min_thrust']
        self.params['max_thrust'] = yaml_dict['max_thrust']

        print("Setting control parameters to:")
        print(self.params)

        os.system(
            "timeout 1s rosparam set /hummingbird/autopilot/Q_pos_xy " + str(self.params['Q_pos_xy']))
        os.system(
            "timeout 1s rosparam set /hummingbird/autopilot/Q_pos_z " + str(self.params['Q_pos_z']))
        os.system(
            "timeout 1s rosparam set /hummingbird/autopilot/Q_attitude " + str(self.params['Q_attitude']))
        os.system(
            "timeout 1s rosparam set /hummingbird/autopilot/Q_velocity " + str(self.params['Q_velocity']))
        os.system(
            "timeout 1s rosparam set /hummingbird/autopilot/R_thrust " + str(self.params['R_thrust']))
        os.system(
            "timeout 1s rosparam set /hummingbird/autopilot/R_pitchroll " + str(self.params['R_pitchroll']))
        os.system(
            "timeout 1s rosparam set /hummingbird/autopilot/R_yaw " + str(self.params['R_yaw']))
        os.system(
            "timeout 1s rosparam set /hummingbird/autopilot/state_cost_exponential " + str(
                self.params['state_cost_exponential']))
        os.system(
            "timeout 1s rosparam set /hummingbird/autopilot/input_cost_exponential " + str(
                self.params['input_cost_exponential']))
        os.system(
            "timeout 1s rosparam set /hummingbird/autopilot/max_bodyrate_xy " + str(self.params['max_bodyrate_xy']))
        os.system(
            "timeout 1s rosparam set /hummingbird/autopilot/max_bodyrate_z " + str(self.params['max_bodyrate_z']))
        os.system(
            "timeout 1s rosparam set /hummingbird/autopilot/min_thrust " + str(self.params['min_thrust']))
        os.system(
            "timeout 1s rosparam set /hummingbird/autopilot/max_thrust " + str(self.params['max_thrust']))

    def run_testing(self):
        self.init_params()

        rospy.init_node('test_performance_node', anonymous=True)

        test_performance = TestPerformance.TestPerformance(self.test_mode)

        # check if configuration was already tested
        # data_dir = "/home/elia/Desktop/controller_logs"
        # files = [f for f in glob.glob(data_dir + "/*.json")]
        # completed_configs = [f.split('/')[-1] for f in files]
        if self.handtuned_params:
            completed_configs = [] #[int(fname[15:-5]) for fname in completed_configs]
        else:
            completed_configs = []
            #completed_configs = [int(fname[5:-5]) for fname in completed_configs]

        num_iterations_per_config = 1

        for i in range(0, self.n_combinations):
            j = random.choice(range(0, self.n_combinations))
            if j in completed_configs:
                print("Configuration already tested, continue.")
                continue
            completed_configs.append(j)
            print("Iteration %d of %d, select configuration %d." % (i, self.n_combinations, j))
            print("Turn off platform")
            os.system("timeout 1s rostopic pub /hummingbird/autopilot/off std_msgs/Empty")

            if self.handtuned_params:
                print("Setting HANDTUNED parameters")
                self.set_handtuned_params()  # set parameters read from yaml file
            else:
                print("Setting RANDOM parameters")
                self.set_params(j)  # set parameters from random search

            # reload parameters
            os.system("timeout 1s rostopic pub /hummingbird/autopilot/reload_parameters std_msgs/Empty")

            print("Fly trajectory")
            simulation_alive = test_performance.run(num_iterations_per_config, self.params, j, self.handtuned_params)

            if not simulation_alive:
                print("Simulation seems to have crashed, shutting down.")
                break


if __name__ == "__main__":
    test_routine = TestRoutine()
    test_routine.run_testing()
