import numpy as np
import glob
import os
import pandas as pd
import tensorflow as tf
import fnmatch
import cv2
import random
from scipy.spatial.transform import Rotation as R


def create_dataset(directory, settings, training=True):
    dataset = SafeDataset(directory, settings, training)
    return dataset


class BodyDataset:
    """
    Base Dataset Class
    """

    def __init__(self, directory, config, training=True):
        self.config = config
        self.directory = directory
        self.training = training
        self.samples = 0
        self.experiments = []
        self.features = []
        self.labels = []
        self.filenames = []
        self.stacked_filenames = [] # Will be used for passing stacked fnames
        img_rootname = 'img_data'
        for root, dirs, files in os.walk(directory, topdown=True, followlinks=True):
            for name in dirs:
                if name.startswith(img_rootname):
                    exp_dir = os.path.join(root, name)
                    self.experiments.append(os.path.abspath(exp_dir))

        self.num_experiments = len(self.experiments)
        self.img_format = 'npy'
        self.data_format = 'csv'

        for exp_dir in self.experiments:
            try:
                self._decode_experiment_dir(exp_dir)
            except:
                raise ImportWarning("Image reading in {} failed".format(
                    exp_dir))
        if self.samples == 0:
            raise IOError("Did not find any file in the dataset folder")
        print('Found {} images  belonging to {} experiments:'.format(
            self.samples, self.num_experiments))

    def _recursive_list(self, subpath, fmt='jpg'):
        return fnmatch.filter(os.listdir(subpath), '*.{}'.format(fmt))

    def build_dataset(self):
        self._build_dataset()

    def _build_dataset(self):
        raise NotImplementedError

    def _decode_experiment_dir(self, directory):
        raise NotImplementedError


class SafeDataset(BodyDataset):
    def __init__(self, directory, config, training=True):
        super(SafeDataset, self).__init__(directory, config, training)
        self.build_dataset()

    def _decode_experiment_dir(self, dir_subpath):
        base_path = os.path.basename(dir_subpath)
        parent_dict = os.path.dirname(dir_subpath)
        data_name = 'data' + base_path[8:] + ".csv"
        data_name = os.path.join(parent_dict, data_name)
        assert os.path.isfile(data_name), "Not Found data file"
        df = pd.read_csv(data_name, delimiter=',')
        num_files = df.shape[0]
        num_images = len(self._recursive_list(dir_subpath, fmt=self.img_format))
        assert num_files == num_images, "Number of features and images does not match"

        features_imu = [# VIO Estimate
                        "Orientation_x",
                        "Orientation_y",
                        "Orientation_z",
                        "Orientation_w",
                        "V_linear_x",
                        "V_linear_y",
                        "V_linear_z",
                        "V_angular_x",
                        "V_angular_y",
                        "V_angular_z"]

        features = [# Reference state
                    "Reference_orientation_x",
                    "Reference_orientation_y",
                    "Reference_orientation_z",
                    "Reference_orientation_w",
                    "Reference_v_linear_x",
                    "Reference_v_linear_y",
                    "Reference_v_linear_z",
                    "Reference_v_angular_x",
                    "Reference_v_angular_y",
                    "Reference_v_angular_z"]

        # Preprocessing: we select the good rollouts (no crash in training data)
        rollout_fts = ["Rollout_idx"]
        rollout_fts_v = df[rollout_fts].values
        position_gt = ["gt_Position_x",
                        "gt_Position_y",
                        "gt_Position_z"]
        position_gt_v = df[position_gt].values
        position_ref = ["Reference_position_x",
                        "Reference_position_y",
                        "Reference_position_z"]
        position_ref_v = df[position_ref].values

        good_rollouts = []

        if rollout_fts_v.shape[0] == 0:
            return

        for r in np.arange(1,np.max(rollout_fts_v)+1):
            rollout_positions = rollout_fts_v == r
            roll_gt = position_gt_v[np.squeeze(rollout_positions),:]
            roll_ref = position_ref_v[np.squeeze(rollout_positions),:]
            if roll_gt.shape[0] == 0:
                continue
            assert roll_ref.shape == roll_gt.shape
            error = np.mean(np.linalg.norm(roll_gt - roll_ref, axis=1))
            if error < self.config.max_allowed_error:
                good_rollouts.append(r)

        if self.config.use_imu:
            features = ["Rollout_idx"] + features_imu + features
        else:
            features = ["Rollout_idx"] + features

        labels = ["Gt_control_command_collective_thrust",
                  "Gt_control_command_bodyrates_x",
                  "Gt_control_command_bodyrates_y",
                  "Gt_control_command_bodyrates_z"]

        features_v = df[features].values
        labels_v = df[labels].values

        for frame_number in range(num_files):
            is_valid = False
            img_fname = os.path.join(dir_subpath,
                                     "{:08d}.{}".format(frame_number, self.img_format))
            if os.path.isfile(img_fname) and (rollout_fts_v[frame_number] in good_rollouts):
                is_valid = True
            if is_valid:
                self.features.append(self.preprocess_fts(features_v[frame_number]))
                self.labels.append(labels_v[frame_number])
                if self.config.use_fts_tracks:
                    self.filenames.append(img_fname)
                self.samples += 1

    def preprocess_fts(self, fts):
        """
        Converts rotations from quadrans to rotation matrix.
        Fts have the following indexing.
        rollout_idx, qx,qy,qz,qw, vx, vy, vz, ax, ay, az, rqx, rqy, rqz, rqw, ...
        """
        fts = fts.tolist()
        ref_rot =  R.from_quat(fts[11:15]).as_matrix().reshape((9,)).tolist()
        if self.config.use_imu:
            odom_rot =  R.from_quat(fts[1:5]).as_matrix().reshape((9,)).tolist()
            processed_fts = [fts[0]] + odom_rot + fts[5:11] + ref_rot + fts[15:]
        else:
            processed_fts = [fts[0]] + ref_rot + fts[5:11]
        return np.array(processed_fts)

    def add_missing_fts(self, features_dict):
        processed_dict = features_dict
        # Could be both positive or negative
        missing_fts = self.config.min_number_fts - len(features_dict.keys())
        if missing_fts > 0:
            # Features are missing
            if missing_fts != self.config.min_number_fts:
                # There is something, we can sample
                new_features_keys = random.choices(list(features_dict.keys()), k=int(missing_fts))
                for j in range(missing_fts):
                    processed_dict[-j-1] = features_dict[new_features_keys[j]]
            else:
                # Zero features, this is a transient
                for j in range(missing_fts):
                    processed_dict[-j-1] = np.zeros((5,))
        elif missing_fts < 0:
            # There are more features than we need, so sample
            del_features_keys = random.sample(features_dict.keys(), int(-missing_fts))
            for k in del_features_keys:
                del processed_dict[k]
        return processed_dict

    def load_fts_sequence(self, sample_num):
        fts_seq = []
        sample_num_np = sample_num.numpy()
        filenames_num = self.stacked_filenames[sample_num_np]
        for idx in range(self.config.seq_len):
            fname_idx = filenames_num[idx]
            if fname_idx < 0:
                fts = {}
            else:
                fname = self.filenames[fname_idx]
                fts = np.load(fname, allow_pickle=True).item()
            fts_seq.append(fts)
        # Reverse list to have it ordered in time (t-seq_len, ..., t)
        fts_seq = reversed(fts_seq)
        # Crop to the required lenght
        fts_seq = [self.add_missing_fts(ft) for ft in fts_seq]
        # Stack
        features_input = np.stack([np.stack([v for v in fts_seq[j].values()]) \
                                   for j in range(self.config.seq_len)])
        return features_input

    def _dataset_map(self, sample_num):
        # First is rollout idx
        label = tf.gather(self.labels, sample_num)
        state_seq = []

        # For states is easy: nothing to do.
        for idx in reversed(range(self.config.seq_len)):
            state = tf.gather(self.features, sample_num - idx)[1:]
            state_seq.append(state)

        state_seq = tf.stack(state_seq)

        # For images, take care they do not overlap
        if self.config.use_fts_tracks:
            fts_seq = tf.py_function(func=self.load_fts_sequence,
                                     inp=[sample_num],
                                     Tout=tf.float32)

            return (state_seq, fts_seq), label
        else:
            return state_seq, label

    def check_equal_dict(self, d1, d2):
        for k_1, v_1 in d1.items():
            try:
                v_2 = d2[k_1]
            except:
                return False
            if not np.array_equal(v_1,v_2):
                return False
        return True

    def _preprocess_fnames(self):
        # Append filenames up to seq_len for fast loading.
        # A bit ugly and inefficent, can be improved
        self.last_init_fts = None
        for k in range(len(self.filenames)):
            if k % 3000 == 0:
                print("Built {:.2f}% of the dataset".format(
                       k/len(self.filenames)*100), end='\r')
            # Check if you can copy the things before
            kth_fts = self.filenames[k]
            kth_fts = np.load(kth_fts, allow_pickle=True).item()
            if k > 0:
                if self.check_equal_dict(self.last_init_fts, kth_fts):
                    self.stacked_filenames.append(self.stacked_filenames[-1])
                    continue
            # This is the lastest observed feature track different from others
            self.last_init_fts = kth_fts
            idx = 0
            rollout_idxes = []
            fname_seq = []
            fts_seq = []
            while len(fts_seq) < self.config.seq_len:
                if k - idx < 0:
                    #this is transient, can only append zeros
                    fname_seq.append(-1)
                    fts_seq.append(0.)
                    continue
                current_idx = k - idx
                rollout_idx = self.features[current_idx][0]
                if idx == 0:
                    fname_seq.append(current_idx)
                    fts_seq.append(kth_fts)
                    rollout_idxes.append(rollout_idx)
                else:
                    if rollout_idx != rollout_idxes[-1]:
                        # it is a transient! Can only append zeros.
                        fname_seq.append(-1)
                        fts_seq.append(0.)
                    else:
                        # Check the features are different
                        fname = self.filenames[current_idx]
                        fts = np.load(fname, allow_pickle=True).item()
                        if not self.check_equal_dict(fts, fts_seq[-1]):
                            # Objects are not equal, can append
                            fname_seq.append(current_idx)
                            fts_seq.append(fts)
                            rollout_idxes.append(rollout_idx)
                idx += 1
            assert len(fts_seq) == len(fname_seq)
            self.stacked_filenames.append(fname_seq)
        # EndFor
        assert len(self.filenames) == len(self.stacked_filenames)

    def _build_dataset(self):
        # Need to take care that rollout_idxs are consistent
        self.features = np.stack(self.features)
        self.features = self.features.astype(np.float32)
        self.labels = np.stack(self.labels)
        self.labels = self.labels.astype(np.float32)
        last_fname_numbers = []
        if self.config.use_fts_tracks:
            self._preprocess_fnames()
        # Preprocess filenames to assess consistency of experiment
        for idx in range(self.config.seq_len-1, self.samples):
            if self.features[idx,0] == self.features[idx-self.config.seq_len+1,0]:
                last_fname_numbers.append(np.int32(idx))

        if self.training:
            np.random.shuffle(last_fname_numbers)

        # Form training batches
        dataset = tf.data.Dataset.from_tensor_slices(last_fname_numbers)
        if self.training:
            dataset = dataset.shuffle(buffer_size=len(last_fname_numbers))
        dataset = dataset.map(self._dataset_map,
                              num_parallel_calls=10 if self.training else 1)
        dataset = dataset.batch(self.config.batch_size,
                               drop_remainder=not self.training)
        dataset = dataset.prefetch(buffer_size=10*self.config.batch_size)
        self.batched_dataset = dataset
