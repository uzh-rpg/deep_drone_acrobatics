#!/usr/bin/env python3
import collections
import copy
import csv
import datetime
import os
import random

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import ControlCommand
from quadrotor_msgs.msg import TrajectoryPoint
from sensor_msgs.msg import PointCloud
from std_msgs.msg import Bool
from std_msgs.msg import Empty
from scipy.spatial.transform import Rotation as R

from .models.bodyrate_learner import BodyrateLearner

TRACK_NUM_NORMALIZE = 10 # Normalization factor for feature length

class TrajectoryBase(object):
    def __init__(self, config, mode):
        self.config = config
        self.odometry = Odometry()
        self.gt_odometry = Odometry()
        self.rows_buffer = []
        self.ref_state = TrajectoryPoint()
        self.vins_odometry = None
        self.gt_control_command = ControlCommand()
        self.counter = 0
        self.features = None
        self.image = None
        self.images_input = None
        self.maneuver_complete = False
        self.shutdown_node = False
        self.record_data = False
        self.is_training = False
        self.use_network = False
        self.net_initialized = False
        self.reference_updated = False
        self.rollout_idx = 0
        self.n_times_net = 0.001 # Due to cope against weird gazebo behaviour
        self.n_times_expert = 0
        self.mode = mode
        self.fts_queue = collections.deque([], maxlen=self.config.seq_len)
        self.state_queue = collections.deque([], maxlen=self.config.seq_len)
        self.reset_queue()
        self.learner = BodyrateLearner(settings=config)
        if self.mode == 'training':
            return  # Nothing to initialize
        self.pub_actions = rospy.Publisher("/" + self.config.quad_name + "/control_command",
                                           ControlCommand, queue_size=1)
        self.odometry_sub = rospy.Subscriber("/" + self.config.quad_name + "/state_estimate",
                                             Odometry,
                                             self.callback_odometry,
                                             queue_size=1,
                                             tcp_nodelay=True)
        self.shutdown_sub = rospy.Subscriber("shutdown_learner", Empty,
                                             self.callback_shutdown,
                                             queue_size=1)
        self.ref_sub = rospy.Subscriber("/" + self.config.quad_name + "/vio_reference",
                                        TrajectoryPoint,
                                        self.callback_ref,
                                        queue_size=1,
                                        tcp_nodelay=True)
        self.control_command_sub = rospy.Subscriber("/" + self.config.quad_name + "/control_command_label",
                                                    ControlCommand,
                                                    self.callback_control_command, queue_size=1,
                                                    tcp_nodelay=True)
        if self.config.use_fts_tracks or self.mode == 'iterative':
            self.fts_sub = rospy.Subscriber("/feature_tracker/feature", PointCloud,
                                            self.callback_fts, queue_size=1)
        self.traj_done_sub = rospy.Subscriber("/" + self.config.quad_name + "/switch_to_network", Bool,
                                              self.callback_nw_switch, queue_size=1)
        self.trajectory_start = rospy.Subscriber("/" + self.config.quad_name + "/trajectory_computation_finish", Bool,
                                                 self.callback_start_trajectory, queue_size=10)
        if self.mode == "testing":
            self.success = 1

    def start_data_recording(self):
        print("Collecting data")
        self.record_data = True

    def stop_data_recording(self):
        print("Stop data collection")
        self.record_data = False
        expert_usage = self.n_times_expert / (self.n_times_net + self.n_times_expert)
        return expert_usage

    def reset_queue(self):
        self.fts_queue.clear()
        self.state_queue.clear()
        self.ref_rot = [0 for _ in range(9)]
        self.odom_rot = [0 for _ in range(9)]
        init_dict = {}
        if self.config.use_imu:
            n_init_states = 30
        else:
            n_init_states = 15
        for i in range(self.config.min_number_fts):
            init_dict[i] = np.zeros((5,), dtype=np.float32)
        for _ in range(self.config.seq_len):
            self.fts_queue.append(init_dict)
            self.state_queue.append(np.zeros((n_init_states,)))
        self.features_input = np.stack([np.stack([v for v in self.fts_queue[j].values()]) \
                                        for j in range(self.config.seq_len)])

    def publish_control_command(self, control_command):
        self.pub_actions.publish(control_command)

    def preprocess_fts(self, data):
        features_dict = {}
        for i in range(len(data.points)):
            ft_id = data.channels[0].values[i]
            x = data.points[i].x
            y = data.points[i].y
            z = data.points[i].z
            velocity_x = data.channels[3].values[i]
            velocity_y = data.channels[4].values[i]
            track_count = 2 * (data.channels[5].values[i] / TRACK_NUM_NORMALIZE) - 1
            assert z == 1
            feat = np.array([x, y, velocity_x, velocity_y, track_count])
            features_dict[ft_id] = feat
        return features_dict

    def add_missing_fts(self, features_dict):
        processed_dict = copy.copy(features_dict)
        # Could be both positive or negative
        missing_fts = self.config.min_number_fts - len(features_dict.keys())
        if missing_fts > 0:
            # Features are missing
            if missing_fts != self.config.min_number_fts:
                # There is something, we can sample
                new_features_keys = random.choices(list(features_dict.keys()), k=int(missing_fts))
                for j in range(missing_fts):
                    processed_dict[-j - 1] = features_dict[new_features_keys[j]]
            else:
                raise IOError("There should not be zero features!")
        elif missing_fts < 0:
            # There are more features than we need, so sample
            del_features_keys = random.sample(features_dict.keys(), int(-missing_fts))
            for k in del_features_keys:
                del processed_dict[k]
        return processed_dict

    def callback_fts(self, data):
        if (not self.config.use_fts_tracks) and (self.mode == 'testing'):
            return
        features = self.preprocess_fts(data)
        if len(features.keys()) != 0:
            # Update features only if something is available
            self.features = features
            preprocessed_fts = self.add_missing_fts(self.features)
            self.fts_queue.append(preprocessed_fts)
            self.features_input = np.stack([np.stack([v for v in self.fts_queue[j].values()]) \
                                            for j in range(self.config.seq_len)])

    def callback_shutdown(self, data):
        self.shutdown_node = True

    def callback_nw_switch(self, msg):
        self.use_network = False
        if msg.data:
            # Trajectory is done, stop everything
            self.maneuver_complete = True
            print("Maneuver is finished")

    def callback_start_trajectory(self, data):
        # VIO is ready, can fly new trajectory
        if data.data:
            print("Ready to start trajectory, network on")
            self.use_network = True
            self.counter = 0

    def callback_start(self, data):
        print("Callback START")
        self.pipeline_off = False

    def callback_off(self, data):
        print("Callback OFF")
        self.pipeline_off = True

    def maneuver_finished(self):
        return self.maneuver_complete

    def callback_odometry(self, data):
        self.odometry = data
        self.odom_rot = R.from_quat([self.odometry.pose.pose.orientation.x,
                                     self.odometry.pose.pose.orientation.y,
                                     self.odometry.pose.pose.orientation.z,
                                     self.odometry.pose.pose.orientation.w]).as_matrix().reshape((9,)).tolist()

    def callback_gt_odometry(self, data):
        self.gt_odometry = data

    def callback_ref(self, data):
        self.ref_state = data
        self.ref_rot = R.from_quat([self.ref_state.pose.orientation.x,
                                    self.ref_state.pose.orientation.y,
                                    self.ref_state.pose.orientation.z,
                                    self.ref_state.pose.orientation.w]).as_matrix().reshape((9,)).tolist()
        if not self.reference_updated:
            self.reference_updated = True

    def callback_control_command(self, data):
        self.control_command = data
        self._generate_control_command()

    def shutdown_requested(self):
        return self.shutdown_node

    def _prepare_net_inputs(self):
        if not self.net_initialized:
            # return fake input for init
            # TODO: change to features
            if self.config.use_imu:
                n_init_states = 30
            else:
                n_init_states = 15
            inputs = {'fts': np.zeros((1, self.config.seq_len, 40, 5), dtype=np.float64),
                      'state': np.zeros((1, self.config.seq_len, n_init_states),
                                        dtype=np.float64)}
            return inputs

        # Reference
        state_inputs = self.ref_rot + [ self.ref_state.velocity.linear.x,
                                        self.ref_state.velocity.linear.y,
                                        self.ref_state.velocity.linear.z,
                                        self.ref_state.velocity.angular.x,
                                        self.ref_state.velocity.angular.y,
                                        self.ref_state.velocity.angular.z]

        if self.config.use_imu:

            imu_states = self.odom_rot + [
                            self.odometry.twist.twist.linear.x,
                            self.odometry.twist.twist.linear.y,
                            self.odometry.twist.twist.linear.z,
                            self.odometry.twist.twist.angular.x,
                            self.odometry.twist.twist.angular.y,
                            self.odometry.twist.twist.angular.z]

            state_inputs = imu_states + state_inputs

        state_inputs = np.array(state_inputs)

        self.state_queue.append(state_inputs)
        state_inputs = np.stack(self.state_queue, axis=0)
        inputs = {'fts': np.expand_dims(self.features_input, axis=0),
                  'state': np.expand_dims(state_inputs, axis=0)}
        return inputs

    def _generate_control_command(self):
        pass # Implemented in derived class

    def write_csv_header(self):
        pass # Implemented in derived class
