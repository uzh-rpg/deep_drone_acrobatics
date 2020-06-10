import math
import os
import shutil
import sys
import time
import datetime

import yaml


def create_settings(settings_yaml, mode='test'):
    setting_dict = {'train': TrainSetting,
                    'test': TestSetting,
                    'dagger': DaggerSetting}
    settings = setting_dict.get(mode, None)
    if settings is None:
        raise IOError("Unidentified Settings")
    settings = settings(settings_yaml)
    return settings


class Settings:
    def __init__(self, settings_yaml, generate_log=True):
        assert os.path.isfile(settings_yaml), settings_yaml

        with open(settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)

            self.quad_name = settings['quad_name']

            self.seq_len = settings['seq_len']

            # --- checkpoint ---
            checkpoint = settings['checkpoint']
            self.resume_training = checkpoint['resume_training']
            assert isinstance(self.resume_training, bool)
            self.resume_ckpt_file = checkpoint['resume_file']

            # Save a copy of the parameters for reproducibility
            log_root = settings['log_dir']
            if not log_root == '' and generate_log:
                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                self.log_dir = os.path.join(log_root, current_time)
                os.makedirs(self.log_dir)
                net_file = "./src/ControllerLearning/models/nets.py"
                assert os.path.isfile(net_file)
                shutil.copy(net_file, self.log_dir)
                shutil.copy(settings_yaml, self.log_dir)

    def add_flags(self):
        self._add_flags()

    def _add_flags(self):
        raise NotImplementedError


class TrainSetting(Settings):
    def __init__(self, settings_yaml):
        super(TrainSetting, self).__init__(settings_yaml, generate_log=True)
        self.settings_yaml = settings_yaml
        self.add_flags()

    def _add_flags(self):
        with open(self.settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)
            # --- Train Time --- #
            train_conf = settings['train']
            self.max_training_epochs = train_conf['max_training_epochs']
            self.max_allowed_error = train_conf['max_allowed_error']
            self.batch_size = train_conf['batch_size']
            self.summary_freq = train_conf['summary_freq']
            self.train_dir = train_conf['train_dir']
            self.use_fts_tracks = train_conf['use_fts_tracks']
            self.use_imu = train_conf['use_imu']
            self.val_dir = train_conf['val_dir']
            self.min_number_fts = train_conf['min_number_fts']
            self.save_every_n_epochs = train_conf['save_every_n_epochs']


class TestSetting(Settings):
    def __init__(self, settings_yaml):
        super(TestSetting, self).__init__(settings_yaml, generate_log=True)
        self.settings_yaml = settings_yaml
        self.add_flags()

    def _add_flags(self):
        with open(self.settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)
            test_time = settings['test_time']
            self.execute_nw_predictions = test_time['execute_nw_predictions']
            assert isinstance(self.execute_nw_predictions, bool)
            self.max_rollouts = test_time['max_rollouts']
            self.fallback_threshold_rates = test_time['fallback_threshold_rates']
            self.fallback_threshold_thrust = test_time['fallback_threshold_thrust']
            self.min_number_fts = test_time['min_number_fts']
            self.use_imu = test_time['use_imu']
            self.use_fts_tracks = test_time['use_fts_tracks']
            self.verbose = settings['verbose']


class DaggerSetting(Settings):
    def __init__(self, settings_yaml):
        super(DaggerSetting, self).__init__(settings_yaml, generate_log=True)
        self.settings_yaml = settings_yaml
        self.add_flags()

    def _add_flags(self):
        with open(self.settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)
            # --- Data Generation --- #
            data_gen = settings['data_generation']
            self.max_rollouts = data_gen['max_rollouts']
            self.double_th_every_n_rollouts = data_gen['double_th_every_n_rollouts']
            self.train_every_n_rollouts = data_gen['train_every_n_rollouts']
            # --- Test Time --- #
            test_time = settings['test_time']
            self.execute_nw_predictions = test_time['execute_nw_predictions']
            assert isinstance(self.execute_nw_predictions, bool)
            self.fallback_threshold_rates = test_time['fallback_threshold_rates']
            self.rand_thrust_mag = test_time['rand_thrust_mag']
            self.rand_rate_mag = test_time['rand_rate_mag']
            self.rand_controller_prob = test_time['rand_controller_prob']
            # --- Train Time --- #
            train_conf = settings['train']
            self.max_training_epochs = train_conf['max_training_epochs']
            self.max_allowed_error = train_conf['max_allowed_error']
            self.batch_size = train_conf['batch_size']
            self.min_number_fts = train_conf['min_number_fts']
            self.summary_freq = train_conf['summary_freq']
            self.train_dir = train_conf['train_dir']
            self.val_dir = train_conf['val_dir']
            self.use_imu = train_conf['use_imu']
            self.use_fts_tracks = train_conf['use_fts_tracks']
            self.val_dir = train_conf['val_dir']
            self.save_every_n_epochs = train_conf['save_every_n_epochs']
            self.verbose = settings['verbose']
            assert isinstance(self.verbose, bool)
