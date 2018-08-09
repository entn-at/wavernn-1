import tensorflow as tf
from hparams import params as pm
from datetime import datetime


class wavernn:

    def __variables__(self):

        tf.Variable(tf.random_normal(
            shape=[pm.cells_detail[-1], pm.hidden_units]),
            dtype=tf.float32,
            name="hidden_weights")
        tf.Variable(tf.random_normal(
            shape=[pm.hidden_units, pm.output_units]),
            dtype=tf.float32,
            name="output_weights")
        tf.Variable(tf.random_normal(
            shape=[pm.hidden_units]),
            dtype=tf.float32,
            name="hidden_bias")
        tf.Variable(tf.random_normal(
            shape=[pm.output_units]),
            dtype=tf.float32,
            name="output_bias")

    def __placeholders__(self):

        self.X = tf.placeholder(dtype=tf.float32, shape=[100])

    def __preprocessing__(self):

        with tf.name_scope("factors"):
            scale_factor = tf.pow(2., 15, name="scale_factor")
            split_factor = tf.pow(2., 8, name="split_factor")

        with tf.name_scope("normalize_input"):
            non_negative_X = tf.add(1., self.X, name="nnz")
            scaled_X = tf.scalar_mul(scale_factor, non_negative_X)

        with tf.name_scope("split_input_to_coarse"):
            coarse = tf.floordiv(scaled_X, split_factor, name="coarse_from_X")
            fine = tf.mod(scaled_X, split_factor, name="fine_from_X")

        with tf.name_scope("scaled_input"):
            ct = tf.divide(coarse, split_factor)
            ft = tf.divide(fine, split_factor)

        with tf.name_scope("time_dilation"):
            ct_1 = ct[:-1]
            ft_1 = ft[:-1]
            ct_1 = tf.pad(ct_1, [[0, 1]])
            ft_1 = tf.pad(ft_1, [[0, 1]])
            time_dilated_coarse = tf.manip.roll(ct_1, shift=1, axis=0)
            time_dilated_fine = tf.manip.roll(ft_1, shift=1, axis=0)

        with tf.name_scope("cast_input"):
            coarse = tf.cast(coarse, dtype=tf.uint8, name="casted_coarse_8")
            fine = tf.cast(fine, dtype=tf.uint8, name="casted_fine_8")

        with tf.name_scope("coarse"):
            self.coarse_input = tf.stack(
                [time_dilated_coarse, time_dilated_fine],
                axis=1, name="coarse_input")
            self.coarse_y = tf.one_hot(
                coarse, depth=256, name="labels_coarse_y")

        print(self.coarse_input, self.coarse_y)

        with tf.name_scope("fine"):
            self.fine_input = tf.stack(
                [time_dilated_coarse, time_dilated_fine, ct],
                axis=1, name="fine_input")

            self.fine_y = tf.one_hot(fine, depth=256, name="labels_fine_y")

    def __batch__(self):
        self.coarse_input = tf.reshape(self.coarse_input, shape=[1, 100, 2])
        self.fine_input = tf.reshape(self.fine_input, shape=[1, 100, 3])

        self.coarse_y = tf.reshape(self.coarse_y, shape=[1, 100, 256])
        self.fine_y = tf.reshape(self.fine_y, shape=[1, 100, 256])

    def __rnn_cell__(self):

        coarse_gru = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(
            x, name="coarse_gru") for x in pm.cells_detail])
        fine_gru = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(
            x, name="fine_gru") for x in pm.cells_detail])

        coarse_output, _ = tf.nn.dynamic_rnn(
            coarse_gru, self.coarse_input, dtype=tf.float32, scope="coarse")
        fine_output, _ = tf.nn.dynamic_rnn(
            fine_gru, self.fine_input, dtype=tf.float32, scope="fine")

        self.coarse_output = coarse_output[:, -1]
        self.fine_output = fine_output[:, -1]

    def __dense_layer__(self):

        with tf.variable_scope("coarse"):
            weights = tf.get_variable(name="hidden_weights",
                                      shape=[pm.cells_detail[-1],
                                             pm.hidden_units])
            projection = tf.matmul(
                self.coarse_output, weights, name="projection")
            bias = tf.get_variable(name="hidden_bias", shape=[pm.hidden_units])
            hidden = tf.add(projection, bias, name="hidden_coarse")
            hidden_relu = tf.nn.relu(hidden, name="relu")
            weights = tf.get_variable(name="output_weights", shape=[
                                      pm.hidden_units, pm.output_units])
            projection = tf.matmul(
                hidden_relu, weights, name="projection")
            bias = tf.get_variable(name="output_bias", shape=[pm.output_units])
            self.coarse_output = tf.add(projection, bias, name="output_coarse")

        with tf.variable_scope("fine"):
            weights = tf.get_variable(name="hidden_weights",
                                      shape=[pm.cells_detail[-1],
                                             pm.hidden_units])
            projection = tf.matmul(
                self.fine_output, weights, name="projection")
            bias = tf.get_variable(name="hidden_bias", shape=[pm.hidden_units])
            hidden = tf.add(projection, bias, name="hidden_fine")
            hidden_relu = tf.nn.relu(hidden, name="relu")
            weights = tf.get_variable(name="output_weights", shape=[
                                      pm.hidden_units, pm.output_units])
            projection = tf.matmul(
                hidden_relu, weights, name="projection")
            bias = tf.get_variable(name="output_bias", shape=[pm.output_units])
            self.fine_output = tf.add(projection, bias, name="output_fine")

    def __cost__(self):

        coarse_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.coarse_y[:, -1],
            labels=self.coarse_output,
            name="coarse_cost_fn")
        coarse_cost_mean = tf.reduce_mean(coarse_cost, name="coarse_cost_mean")

        fine_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.fine_y[:, -1], labels=self.fine_output, name="fine_cost_fn")

        fine_cost_mean = tf.reduce_mean(fine_cost, name="fine_cost_mean")

        self.cost = tf.add(coarse_cost_mean, fine_cost_mean, name="total_cost")

    def __optimizer__(self):

        self.optimizer = tf.train.AdadeltaOptimizer(
            learning_rate=0.05).minimize(self.cost)

    def __summary__(self):
        self.training_summary = list()
        self.training_summary_index = 0

    def __training_metrics__(self):
        cost_sum = tf.summary.scalar("cost", self.cost)
        self.training_summary.append(cost_sum)

    def __summary_merger__(self):
        self.training_summary = tf.summary.merge(self.training_summary)

    def __summary_writer__(self):
        log_folder = "./log/{:%Y-%m-%d %H:%M:%S}/%s".format(datetime.now())
        self.training_writer = tf.summary.FileWriter(
            log_folder % "train", self.session.graph)

    def __init__(self):

        self.__placeholders__()
        self.__variables__()
        self.__preprocessing__()
        self.__batch__()
        self.__rnn_cell__()
        self.__dense_layer__()
        self.__cost__()
        self.__optimizer__()
        self.__summary__()
        self.__training_metrics__()
        self.__summary_merger__()

        self.session = tf.Session()
        var = tf.global_variables_initializer()
        self.session.run(var)

        self.__summary_writer__()


wavernn()
