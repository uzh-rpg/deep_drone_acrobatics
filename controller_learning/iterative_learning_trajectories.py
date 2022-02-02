#!/usr/bin/env python3

import argparse
import os
import sys
import time
import numpy as np
import rospy
from src.ControllerLearning import TrajectoryLearning
from std_msgs.msg import Bool
from common import update_mpc_params, setup_sim, random_replace, initialize_vio

from config.settings import create_settings


class Trainer():
    def __init__(self, settings):
        rospy.init_node('iterative_learning_node', anonymous=False)
        self.settings = settings
        self.trajectory_done = False
        self.traj_done_sub = rospy.Subscriber("/hummingbird/switch_to_network", Bool,
                                              self.callback_traj_done, queue_size=1)

    def callback_traj_done(self, data):
        self.trajectory_done = data.data

    def start_experiment(self, learner):
        reset_success_str = 'rostopic pub /success_reset std_msgs/Empty "{}" -1'
        os.system(reset_success_str)
        initialize_vio()
        learner.latest_thrust_factor = 1.0 # Could be changed to adapt for different quadrotors.
        print("Doing experiment {}, with such factor {}".format(learner.rollout_idx, learner.latest_thrust_factor))
        # if True, we will still use the VIO-orientation, even when initialization is poor.
        # If set to False, we will fall back to GT.
        # Set to True only if you are sure your VIO is well calibrated.
        use_chimaera = False
        # check if initialization was good. If not, we will perform rollout with ground truth to not waste time!
        vio_init_good = learner.vio_init_good
        if vio_init_good:
            rospy.loginfo("VINS-Mono initialization is good, switching to vision-based state estimate!")
            os.system("timeout 1s rostopic pub /switch_odometry std_msgs/Int8 'data: 1'")
        else:
            if use_chimaera:
                rospy.logerr("VINS-Mono initialization is poor, use orientation, bodyrates from VIO and linear velocity estimate from GT!")
                os.system("timeout 1s rostopic pub /switch_odometry std_msgs/Int8 'data: 2'")
            else:
                rospy.logerr("VINS-Mono initialization is poor, keeping ground truth estimate!")
                os.system("timeout 1s rostopic pub /switch_odometry std_msgs/Int8 'data: 0'")
        # Start Flying!
        os.system("timeout 1s rostopic pub /hummingbird/fpv_quad_looping/execute_trajectory std_msgs/Bool 'data: true'")
        return vio_init_good

    def perform_training(self):
        learner = TrajectoryLearning.TrajectoryLearning(self.settings, mode="iterative")
        shutdown_requested = False
        train_every_n_rollouts = self.settings.train_every_n_rollouts
        if self.settings.execute_nw_predictions:
            print("-------------------------------------------")
            print("Running Dagger with the following params")
            print("Rates threshold: {}; Rand Controller Th {}".format(
                self.settings.fallback_threshold_rates,
                self.settings.rand_controller_prob))
            print("-------------------------------------------")
        else:
            print("---------------------------")
            print("Collecting Data with Expert")
            print("---------------------------")
        while (not shutdown_requested) and (learner.rollout_idx < self.settings.max_rollouts):
            self.trajectory_done = False
            setup_sim()
            learner.start_data_recording()
            self.start_experiment(learner)
            print("Starting Experiment {}".format(learner.rollout_idx))
            start_time = time.time()
            time_run = 0
            ref_log = []
            gt_pos_log = []
            error_log = []
            while (not self.trajectory_done) and (time_run < 100):
                time.sleep(0.1)
                time_run = time.time() - start_time
                if learner.use_network and learner.reference_updated:
                    pos_ref_dict = learner.compute_trajectory_error()
                    gt_pos_log.append(pos_ref_dict["gt_pos"])
                    ref_log.append(pos_ref_dict["gt_ref"])
                    error_log.append(np.linalg.norm(pos_ref_dict["gt_pos"] - pos_ref_dict["gt_ref"]))
            # final logging
            tracking_error = np.mean(error_log)
            median_traj_error = np.median(error_log)
            t_log = np.stack((ref_log, gt_pos_log), axis=0)
            expert_usage = learner.stop_data_recording()
            shutdown_requested = learner.shutdown_requested()
            print("Expert used {:.03f}% of the times".format(100.0 * expert_usage))
            print("Mean Tracking Error is {:.03f}".format(tracking_error))
            print("Median Tracking Error is {:.03f}".format(median_traj_error))
            if learner.rollout_idx % train_every_n_rollouts == 0:
                os.system("rosservice call /gazebo/pause_physics")
                learner.train()
                os.system("rosservice call /gazebo/unpause_physics")
            if (learner.rollout_idx % self.settings.double_th_every_n_rollouts) == 0:
                self.settings.fallback_threshold_rates += 0.5
                print("Setting Rate Threshold to {}".format(self.settings.fallback_threshold_rates))
                self.settings.rand_controller_prob = np.minimum(0.3, self.settings.rand_controller_prob * 2)
                print("Setting Rand Controller Prob to {}".format(self.settings.rand_controller_prob))
            if self.settings.verbose:
                t_log_fname = os.path.join(self.settings.log_dir, "traj_log_{:5d}.npy".format(learner.rollout_idx))
                np.save(t_log_fname, t_log)

    def perform_testing(self):
        learner = TrajectoryLearning.TrajectoryLearning(self.settings, mode="testing")
        shutdown_requested = False
        rollout_idx = 0
        while (not shutdown_requested) and (rollout_idx < self.settings.max_rollouts):
            self.trajectory_done = False
            setup_sim()
            if self.settings.verbose:
                # Will save data for debugging reasons
                learner.start_data_recording()
            self.start_experiment(learner)
            start_time = time.time()
            time_run = 0
            ref_log = []
            gt_pos_log = []
            error_log = []
            while (not self.trajectory_done) and (time_run < 100):
                time.sleep(0.1)
                time_run = time.time() - start_time
                if learner.use_network and learner.reference_updated:
                    pos_ref_dict = learner.compute_trajectory_error()
                    gt_pos_log.append(pos_ref_dict["gt_pos"])
                    ref_log.append(pos_ref_dict["gt_ref"])
                    error_log.append(np.linalg.norm(pos_ref_dict["gt_pos"] - pos_ref_dict["gt_ref"]))
            # final logging
            tracking_error = np.mean(error_log)
            median_traj_error = np.median(error_log)
            t_log = np.stack((ref_log, gt_pos_log), axis=0)
            expert_usage = learner.stop_data_recording()
            shutdown_requested = learner.shutdown_requested()
            print("{} Rollout: Expert used {:.03f}% of the times".format(rollout_idx+1, 100.0 * expert_usage))
            print("Mean Tracking Error is {:.03f}".format(tracking_error))
            print("Median Tracking Error is {:.03f}".format(median_traj_error))
            rollout_idx +=1
            if self.settings.verbose:
                t_log_fname = os.path.join(self.settings.log_dir, "traj_log_{:05d}.npy".format(rollout_idx))
                np.save(t_log_fname, t_log)


def main():
    parser = argparse.ArgumentParser(description='Train RAF network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = create_settings(settings_filepath, mode='dagger')
    update_mpc_params()
    setup_sim()
    trainer = Trainer(settings)
    trainer.perform_training()


if __name__ == "__main__":
    main()
