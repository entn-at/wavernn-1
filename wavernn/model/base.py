import tensorflow as tf
from ..hparams import hyper_parameters
from ..utils.base import pillar_type


class pillar:

    def __init__(self, pillar_type):

        self.__initialize_pre_defaults()
        self.__sequential_layers()
        self.pillar_type = pillar_type

    def __initialize_weights(self):

        self.weights = {
            "relu": tf.Variable(tf.random_normal(shape=(
                hyper_parameters.cells_detail[-1],
                hyper_parameters.rnn_to_connected)), dtype=tf.float32),

            "softmax": tf.Variable(tf.random_normal(shape=(
                hyper_parameters.rnn_to_connected,
                hyper_parameters.connected_to_connected)), dtype=tf.float32),
        }

    def __initialize_bias(self):

        self.bias = {
            "relu": tf.Variable(tf.random_normal(shape=(
                hyper_parameters.rnn_to_connected)), dtype=tf.float32),

            "softmax": tf.Variable(tf.random_normal(shape=(
                hyper_parameters.connected_to_connected)), dtype=tf.float32),
        }

    def __initialize_connected_layers(self):

        self.__initialize_weights()
        self.__initialize_bias()

        layer_relu = tf.nn.relu(tf.matmul(self.output, self.weights['relu'])
                                + self.bias['relu'])
        self.output = (tf.matmul(layer_relu, self.weights['softmax'])
                       + self.bias['softmax'])

    def __initialize_pre_defaults(self):

        self.highest_level = 2**(hyper_parameters.bit_rate-1)

        self.split_level = 2**(hyper_parameters.bit_rate/2)

        self.regression = hyper_parameters.regression

        self.batch_size = hyper_parameters.batch_size

        self.sequence_length = hyper_parameters.max_sequence_length

        self.output = self.split_level if self.regression else (
            hyper_parameters.raw_output)

        self.y = tf.placeholder(
            shape=(self.batch_size, self.output),
            dtype=tf.float32, name="output_y")

        if self.pillar_type == pillar_type.coarse:
            self.X = tf.placeholder(
                shape=(self.batch_size, self.sequence_length,
                       hyper_parameters.coarse_features),
                dtype=tf.float32, name="coarse_X")
        else:
            self.X = tf.placeholder(
                shape=(self.batch_size, self.sequence_length,
                       hyper_parameters.fine_features),
                dtype=tf.float32, name="fine_X")

    def __sequential_layers(self):

        cells_structure = hyper_parameters.cells_detail
        sequential_layers = []

        if self.regression and cells_structure[-1] is not 1:
            raise ValueError(
                "With regression, cells should have funnel structure")

        for each_cell in cells_structure:
            sequential_layers.append(tf.nn.rnn_cell.GRUCell(
                each_cell, name="sequential_gru"))
        self.learning_cell = tf.nn.rnn_cell.MultiRNNCell(sequential_layers)

    def __output(self):

        self.output = tf.nn.dynamic_rnn(self.learning_cell, self.X)

        if not self.regression:
            self.__initialize_connected_layers()

    def __cost(self):

        if not self.regression:
            self.y = tf.scalar_mul(self.split_level, self.y)
            self.y = tf.one_hot(self.y, depth=self.split_level)
            self.cost = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.y, logits=self.output)
        else:
            self.cost = tf.losses.mean_squared_error(
                labels=self.y, predictions=self.output)
