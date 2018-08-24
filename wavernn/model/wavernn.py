import tensorflow as tf
from wavernn.hparams import params as pm
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

        self.X = tf.placeholder(dtype=tf.float32, shape=[None, pm.batch_size, 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None])

    def __preprocessing__(self):

        with tf.name_scope("factors"):
            self.scale_factor = tf.pow(2., 15, name="scale_factor")
            self.split_factor = tf.pow(2., 8, name="split_factor")
            self.second_split = tf.pow(2., 7, name="split_factor")

        with tf.name_scope("normalize_input"):
            non_negative_X = tf.add(1., self.y, name="nnz")
            scaled_X = tf.scalar_mul(self.scale_factor, non_negative_X)

        with tf.name_scope("split_input_to_coarse"):
            coarse = tf.floordiv(
                scaled_X, self.split_factor, name="coarse_from_X")
            fine = tf.mod(scaled_X, self.split_factor, name="fine_from_X")

        with tf.name_scope("scaled_input"):
            ct = tf.divide(coarse, self.second_split) - 1
            ft = tf.divide(fine, self.second_split) - 1

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
            print(coarse,fine)
            
        with tf.name_scope("coarse_input"):
            self.coarse_input = tf.stack(
                [time_dilated_coarse, time_dilated_fine],
                axis=1, name="coarse_input")
            self.coarse_y = tf.one_hot(
                coarse, depth=256, name="labels_coarse_y")

        with tf.name_scope("fine_input"):
            self.fine_input = tf.stack(
                [time_dilated_coarse, time_dilated_fine, ct],
                axis=1, name="fine_input")

            self.fine_y = tf.one_hot(fine, depth=256, name="labels_fine_y")

    def __batch__(self):
        self.coarse_input = tf.reshape(self.coarse_input, shape=[
                                       -1, pm.batch_size, 2])
        self.coarse_input = tf.concat([self.coarse_input, self.X], axis=2)
        self.fine_input = tf.reshape(self.fine_input, shape=[
                                     -1, pm.batch_size, 3])

        self.coarse_y = tf.reshape(self.coarse_y, shape=[
                                   -1, pm.batch_size, 256])
        self.fine_y = tf.reshape(
            self.fine_y, shape=[-1, pm.batch_size, 256])

    def __rnn_cell__(self):

        coarse_gru = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(
            x, name="coarse_gru") for x in pm.cells_detail])
        fine_gru = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(
            x, name="fine_gru") for x in pm.cells_detail])

        coarse_output, _ = tf.nn.dynamic_rnn(
            coarse_gru, self.coarse_input, dtype=tf.float32, scope="coarse_gru")
        fine_output, _ = tf.nn.dynamic_rnn(
            fine_gru, self.fine_input, dtype=tf.float32, scope="fine_gru")

        self.coarse_output = coarse_output
        self.fine_output = fine_output

    def __dense_layer__(self):

        with tf.variable_scope("coarse_dense"):
            weights = tf.get_variable(name="hidden_weights",
                                      shape=[pm.cells_detail[-1],
                                             pm.hidden_units])
            
            projection = tf.tensordot(
                self.coarse_output, weights, axes=[[2], [0]], name="projection")
            bias = tf.get_variable(name="hidden_bias", shape=[pm.hidden_units])
            hidden = tf.add(projection, bias, name="hidden_coarse")
            hidden_relu = tf.nn.relu(hidden, name="relu")
            weights = tf.get_variable(name="output_weights", shape=[
                                      pm.hidden_units, pm.output_units])
            projection = tf.tensordot(
                hidden_relu, weights, axes=[[2], [0]], name="projection")
            bias = tf.get_variable(name="output_bias", shape=[pm.output_units])
            self.coarse_output = tf.add(projection, bias, name="output_coarse")

        with tf.variable_scope("fine_dense"):
            weights = tf.get_variable(name="hidden_weights",
                                      shape=[pm.cells_detail[-1],
                                             pm.hidden_units])
            projection = tf.tensordot(
                self.fine_output, weights, axes=[[2],[0]], name="projection")
            bias = tf.get_variable(name="hidden_bias", shape=[pm.hidden_units])
            hidden = tf.add(projection, bias, name="hidden_fine")
            hidden_relu = tf.nn.relu(hidden, name="relu")
            weights = tf.get_variable(name="output_weights", shape=[
                                      pm.hidden_units, pm.output_units])
            projection = tf.tensordot(
                hidden_relu, weights, axes=[[2],[0]], name="projection")
            bias = tf.get_variable(name="output_bias", shape=[pm.output_units])
            self.fine_output = tf.add(projection, bias, name="output_fine")

    def __cost__(self):

        coarse_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.coarse_output,
            labels=self.coarse_y,
            name="coarse_cost_fn")
        coarse_cost_mean = tf.reduce_mean(coarse_cost, name="coarse_cost_mean")

        fine_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.fine_output, labels=self.fine_y, name="fine_cost_fn")

        fine_cost_mean = tf.reduce_mean(fine_cost, name="fine_cost_mean")

        self.cost = tf.add(coarse_cost_mean, fine_cost_mean, name="total_cost")

    def __signal__(self):

        softmax_coarse = tf.nn.softmax(self.coarse_output)
        softmax_fine = tf.nn.softmax(self.fine_output)
        coarse_band = tf.cast(
            tf.argmax(softmax_coarse, axis=2), dtype=tf.float32)
        fine_band = tf.cast(
            tf.argmax(softmax_fine, axis=2), dtype=tf.float32)
        signal_band = tf.scalar_mul(self.split_factor, coarse_band)
        signal_band = tf.add(signal_band, fine_band)
        self.signal = tf.divide(signal_band, self.scale_factor) - 1
        self.signal = tf.reshape(self.signal,shape=[-1])

    def __optimizer__(self):

        self.optimizer = tf.train.AdadeltaOptimizer(
            learning_rate=0.5).minimize(self.cost)

    def __summary__(self):
        self.training_summary = list()
        self.training_summary_index = 0

    def __training_metrics__(self):
        cost_sum = tf.summary.scalar("cost", self.cost)
        audio_sum = tf.summary.audio("output", [self.signal],sample_rate=22050)
        self.training_summary.append(cost_sum)
        self.training_summary.append(audio_sum)

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
        self.__signal__()
        self.__optimizer__()
        self.__summary__()
        self.__training_metrics__()
        self.__summary_merger__()

        self.session = tf.Session()
        var = tf.global_variables_initializer()
        self.session.run(var)

        self.__summary_writer__()
