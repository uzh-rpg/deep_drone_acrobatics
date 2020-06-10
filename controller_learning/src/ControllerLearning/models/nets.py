import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, LeakyReLU, Conv1D
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D

try:
    from .tf_addons_normalizations import InstanceNormalization
except:
    from tf_addons_normalizations import InstanceNormalization

def create_network(settings):
    net = AggressiveNet(settings)
    return net


class Network(Model):
    def __init__(self):
        super(Network, self).__init__()

    def create(self):
        self._create()

    def call(self, x):
        return self._internal_call(x)

    def _create(self):
        raise NotImplementedError

    def _internal_call(self):
        raise NotImplementedError

class AggressiveNet(Network):
    def __init__(self, config):
        super(AggressiveNet, self).__init__()
        self.config = config
        self._create(input_size=(config.seq_len, config.min_number_fts, 5))

    def _create(self, input_size, has_bias=True, learn_affine=True):
        """Init.
        Args:
            input_size (float): size of input
            has_bias (bool, optional): Defaults to True. Conv1d bias?
            learn_affine (bool, optional): Defaults to True. InstanceNorm1d affine?
        """
        if self.config.use_fts_tracks:
            f = 2.0
            self.pointnet = [Conv2D(int(16 * f), kernel_size=1, strides=1, padding='valid',
                                    dilation_rate=1, use_bias=has_bias, input_shape=input_size),
                             InstanceNormalization(axis=3, epsilon=1e-5, center=learn_affine, scale=learn_affine),
                             LeakyReLU(alpha=1e-2),
                             Conv2D(int(32 * f), kernel_size=1, strides=1, padding='valid', dilation_rate=1,
                                    use_bias=has_bias),
                             InstanceNormalization(axis=3, epsilon=1e-5, center=learn_affine, scale=learn_affine),
                             LeakyReLU(alpha=1e-2),
                             Conv2D(int(64 * f), kernel_size=1, strides=1, padding='valid', dilation_rate=1,
                                    use_bias=has_bias),
                             InstanceNormalization(axis=3, epsilon=1e-5, center=learn_affine, scale=learn_affine),
                             LeakyReLU(alpha=1e-2),
                             Conv2D(int(64 * f), kernel_size=1, strides=1, padding='valid', dilation_rate=1,
                                    use_bias=has_bias),
                             GlobalAveragePooling2D()]

            input_size = (self.config.seq_len, int(64*f))
            self.fts_mergenet = [Conv1D(int(64 * f), kernel_size=2, strides=1, padding='same',
                                    dilation_rate=1, input_shape=input_size),
                                 LeakyReLU(alpha=1e-2),
                                 Conv1D(int(32 * f), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                                 LeakyReLU(alpha=1e-2),
                                 Conv1D(int(32 * f), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                                 LeakyReLU(alpha=1e-2),
                                 Conv1D(int(32 * f), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                                 LeakyReLU(alpha=1e-2),
                                 Flatten(),
                                 Dense(int(64*f))]

        g = 2.0
        self.states_conv = [Conv1D(int(64 * g), kernel_size=2, strides=1, padding='same',
                                dilation_rate=1),
                             LeakyReLU(alpha=1e-2),
                             Conv1D(int(32 * g), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                             LeakyReLU(alpha=1e-2),
                             Conv1D(int(32 * g), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                             LeakyReLU(alpha=1e-2),
                             Conv1D(int(32 * g), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                             Flatten(),
                             Dense(int(64*g))]

        self.control_module = [Dense(64*g),
                               LeakyReLU(alpha=1e-2),
                               Dense(32*g),
                               LeakyReLU(alpha=1e-2),
                               Dense(16*g),
                               LeakyReLU(alpha=1e-2),
                               Dense(4)]

    def _pointnet_branch(self, single_t_features):
        x = tf.expand_dims(single_t_features, axis=1)
        for f in self.pointnet:
            x = f(x)
        return x

    def _features_branch(self, input_features):
        preprocessed_fts = tf.map_fn(self._pointnet_branch,
                                     elems=input_features,
                                     parallel_iterations=self.config.seq_len) # (seq_len, batch_size, 64)
        preprocessed_fts = tf.transpose(preprocessed_fts, (1,0,2)) # (batch_size, seq_len, 64)
        x = preprocessed_fts
        for f in self.fts_mergenet:
            x = f(x)
        return x

    def _states_branch(self, embeddings):
        x = embeddings
        for f in self.states_conv:
            x = f(x)
        return x

    def _control_branch(self, embeddings):
        x = embeddings
        for f in self.control_module:
            x = f(x)
        return x

    def _internal_call(self, inputs):
        states = inputs['state']
        if self.config.use_fts_tracks:
            fts_stack = inputs['fts']  # (batch_size, seq_len, min_numb_features, 5)
            fts_stack = tf.transpose(fts_stack, (1,0,2,3)) # (seq_len, batch_size, min_numb_features, 5)
            # Execute PointNet Part
            fts_embeddings = self._features_branch(fts_stack)
        states_embeddings = self._states_branch(states)
        if self.config.use_fts_tracks:
            total_embeddings = tf.concat((fts_embeddings, states_embeddings), axis=1)
        else:
            total_embeddings = states_embeddings
        output = self._control_branch(total_embeddings)
        return output
