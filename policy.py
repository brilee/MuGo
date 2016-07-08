'''
Neural network architecture.
The input to the policy network is a 19 x 19 x 48 image stack consisting of
48 feature planes. The first hidden layer zero pads the input into a 23 x 23
image, then convolves k filters of kernel size 5 x 5 with stride 1 with the
input image and applies a rectifier nonlinearity. Each of the subsequent
hidden layers 2 to 12 zero pads the respective previous hidden layer into a
21 x 21 image, then convolves k filters of kernel size 3 x 3 with stride 1,
again followed by a rectifier nonlinearity. The final layer convolves 1 filter
of kernel size 1 x 1 with stride 1, with a different bias for each position,
and applies a softmax function. The match version of AlphaGo used k = 192
filters; Fig. 2b and Extended Data Table 3 additionally show the results
of training with k = 128, 256 and 384 filters.

The input to the value network is also a 19 x 19 x 48 image stack, with an
additional binary feature plane describing the current colour to play.
Hidden layers 2 to 11 are identical to the policy network, hidden layer 12
is an additional convolution layer, hidden layer 13 convolves 1 filter of
kernel size 1 x 1 with stride 1, and hidden layer 14 is a fully connected
linear layer with 256 rectifier units. The output layer is a fully connected
linear layer with a single tanh unit.
'''
import itertools
import os
import tensorflow as tf

import features
import go

class PolicyNetwork(object):
    def __init__(self, num_input_planes, k=32, num_int_conv_layers=3):
        self.num_input_planes = num_input_planes
        self.k = k
        self.num_int_conv_layers = num_int_conv_layers
        self.test_summary_writer = None
        self.training_summary_writer = None
        self.session = tf.Session()
        self.set_up_network()

    def set_up_network(self):
        # a global_step variable allows epoch counts to persist through multiple training sessions
        global_step = tf.Variable(0, name="global_step", trainable=False)
        x = tf.placeholder(tf.float32, [None, go.N, go.N, self.num_input_planes])
        y = tf.placeholder(tf.float32, shape=[None, go.N ** 2])

        #convenience functions for initializing weights and biases
        def weight_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

        # initial conv layer is 5x5
        W_conv_init = weight_variable([5, 5, self.num_input_planes, self.k], name="W_conv_init")
        h_conv_init = tf.nn.relu(conv2d(x, W_conv_init))

        # followed by a series of 3x3 conv layers
        W_conv_intermediate = []
        h_conv_intermediate = []
        _current_h_conv = h_conv_init
        for i in range(self.num_int_conv_layers):
            W_conv_intermediate.append(weight_variable([3, 3, self.k, self.k], name="W_conv_inter" + str(i)))
            h_conv_intermediate.append(tf.nn.relu(conv2d(_current_h_conv, W_conv_intermediate[-1])))
            _current_h_conv = h_conv_intermediate[-1]

        W_conv_final = weight_variable([1, 1, self.k, 1], name="W_conv_final")
        b_conv_final = tf.Variable(tf.constant(0, shape=[go.N ** 2], dtype=tf.float32), name="b_conv_final")
        h_conv_final = conv2d(h_conv_intermediate[-1], W_conv_final)
        output = tf.nn.softmax(tf.reshape(h_conv_final, [-1, go.N ** 2]) + b_conv_final)

        log_likelihood_cost = -tf.reduce_mean(tf.reduce_sum(tf.mul(tf.log(output), y), reduction_indices=[1]))

        train_step = tf.train.AdamOptimizer(1e-4).minimize(log_likelihood_cost, global_step=global_step)
        was_correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(was_correct, tf.float32))

        saver = tf.train.Saver()

        weight_summaries = tf.merge_summary([
            tf.histogram_summary(weight_var.name, weight_var)
            for weight_var in itertools.chain(
                [W_conv_init],
                W_conv_intermediate,
                [W_conv_final])])
        _accuracy = tf.scalar_summary("accuracy", accuracy)
        _cost = tf.scalar_summary("log_likelihood_cost", log_likelihood_cost)
        accuracy_summaries = tf.merge_summary([_accuracy, _cost])

        # save everything to self.
        for name, thing in locals().items():
            if not name.startswith('_'):
                setattr(self, name, thing)

    def initialize_logging(self, tensorboard_logdir):
        self.test_summary_writer = tf.train.SummaryWriter(os.path.join(tensorboard_logdir, "test"), self.session.graph)
        self.training_summary_writer = tf.train.SummaryWriter(os.path.join(tensorboard_logdir, "training"), self.session.graph)

    def initialize_variables(self, save_file=None):
        if save_file is None:
            self.session.run(tf.initialize_all_variables())
        else:
            self.saver.restore(self.session, save_file)

    def save_variables(self, save_file):
        self.saver.save(self.session, save_file)

    def train(self, training_data, batch_size=32):
        num_minibatches = training_data.data_size // batch_size
        for i in range(num_minibatches):
            batch_x, batch_y = training_data.get_batch(batch_size)
            global_step = self.session.run(self.global_step)
            if global_step % 100 == 0:
                summary_str, train_accuracy = self.session.run(
                    [self.accuracy_summaries, self.accuracy],
                    feed_dict={self.x: batch_x, self.y: batch_y})
                if self.training_summary_writer is not None:
                    self.training_summary_writer.add_summary(summary_str, global_step)
                print("Step %d, training data accuracy: %g" % (global_step, train_accuracy))
            self.session.run(self.train_step, feed_dict={self.x: batch_x, self.y: batch_y})

    def run(self, position):
        processed_position = features.DEFAULT_FEATURES.extract(position)
        return self.session.run(self.output, feed_dict={self.x: processed_position[None, :]})[0]

    def check_accuracy(self, test_data):
        weight_summaries = self.session.run(self.weight_summaries)
        accuracy_summaries, test_accuracy = self.session.run(
            [self.accuracy_summaries, self.accuracy],
            feed_dict={self.x: test_data.pos_features, self.y: test_data.next_moves})
        global_step = self.session.run(self.global_step)
        if self.test_summary_writer is not None:
            self.test_summary_writer.add_summary(weight_summaries, global_step)
            self.test_summary_writer.add_summary(accuracy_summaries, global_step)
        print("Step %s test data accuracy: %g" % (global_step, test_accuracy))

