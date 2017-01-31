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
import math
import os
import sys
import tensorflow as tf

import features
import go
import utils

EPSILON = 1e-35

class PolicyNetwork(object):
    def __init__(self, features=features.DEFAULT_FEATURES, k=32, num_int_conv_layers=3, use_cpu=False):
        self.num_input_planes = sum(f.planes for f in features)
        self.features = features
        self.k = k
        self.num_int_conv_layers = num_int_conv_layers
        self.test_summary_writer = None
        self.training_summary_writer = None
        self.test_stats = StatisticsCollector()
        self.training_stats = StatisticsCollector()
        self.session = tf.Session()
        if use_cpu:
            with tf.device("/cpu:0"):
                self.set_up_network()
        else:
            self.set_up_network()

    def set_up_network(self):
        # a global_step variable allows epoch counts to persist through multiple training sessions
        global_step = tf.Variable(0, name="global_step", trainable=False)
        x = tf.placeholder(tf.float32, [None, go.N, go.N, self.num_input_planes])
        y = tf.placeholder(tf.float32, shape=[None, go.N ** 2])

        #convenience functions for initializing weights and biases
        def _weight_variable(shape, name):
            # If shape is [5, 5, 20, 32], then each of the 32 output planes
            # has 5 * 5 * 20 inputs.
            number_inputs_added = utils.product(shape[:-1])
            stddev = 1 / math.sqrt(number_inputs_added)
            # http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
            return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

        def _conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

        # initial conv layer is 5x5
        W_conv_init = _weight_variable([5, 5, self.num_input_planes, self.k], name="W_conv_init")
        h_conv_init = tf.nn.relu(_conv2d(x, W_conv_init), name="h_conv_init")

        # followed by a series of 3x3 conv layers
        W_conv_intermediate = []
        h_conv_intermediate = []
        _current_h_conv = h_conv_init
        for i in range(self.num_int_conv_layers):
            with tf.name_scope("layer"+str(i)):
                W_conv_intermediate.append(_weight_variable([3, 3, self.k, self.k], name="W_conv"))
                h_conv_intermediate.append(tf.nn.relu(_conv2d(_current_h_conv, W_conv_intermediate[-1]), name="h_conv"))
                _current_h_conv = h_conv_intermediate[-1]

        W_conv_final = _weight_variable([1, 1, self.k, 1], name="W_conv_final")
        b_conv_final = tf.Variable(tf.constant(0, shape=[go.N ** 2], dtype=tf.float32), name="b_conv_final")
        h_conv_final = _conv2d(h_conv_intermediate[-1], W_conv_final)

        # Add epsilon to avoid taking the log of 0 in following step
        output = tf.nn.softmax(tf.reshape(h_conv_final, [-1, go.N ** 2]) + b_conv_final) + tf.constant(EPSILON)

        log_likelihood_cost = -tf.reduce_mean(tf.reduce_sum(tf.mul(tf.log(output), y), reduction_indices=[1]))

        # The step size was initialized to 0.003 and was halved every 80 million training steps
        _learning_rate = tf.train.exponential_decay(3e-3, global_step,
                                           8e7, 0.5)
        train_step = tf.train.GradientDescentOptimizer(_learning_rate).minimize(log_likelihood_cost, global_step=global_step)
        was_correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(was_correct, tf.float32))

        weight_summaries = tf.summary.merge([
            tf.summary.histogram(weight_var.name, weight_var)
            for weight_var in [W_conv_init] +  W_conv_intermediate + [W_conv_final, b_conv_final]],
            name="weight_summaries"
        )
        activation_summaries = tf.summary.merge([
            tf.summary.histogram(act_var.name, act_var)
            for act_var in [h_conv_init] + h_conv_intermediate + [h_conv_final]],
            name="activation_summaries"
        )
        saver = tf.train.Saver()

        # save everything to self.
        for name, thing in locals().items():
            if not name.startswith('_'):
                setattr(self, name, thing)

    def initialize_logging(self, tensorboard_logdir):
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(tensorboard_logdir, "test"), self.session.graph)
        self.training_summary_writer = tf.summary.FileWriter(os.path.join(tensorboard_logdir, "training"), self.session.graph)

    def initialize_variables(self, save_file=None):
        self.session.run(tf.global_variables_initializer())
        if save_file is not None:
            self.saver.restore(self.session, save_file)

    def get_global_step(self):
        return self.session.run(self.global_step)

    def save_variables(self, save_file):
        if save_file is not None:
            print("Saving checkpoint to %s" % save_file, file=sys.stderr)
            self.saver.save(self.session, save_file)

    def train(self, training_data, batch_size=32):
        num_minibatches = training_data.data_size // batch_size
        for i in range(num_minibatches):
            batch_x, batch_y = training_data.get_batch(batch_size)
            _, accuracy, cost = self.session.run(
                [self.train_step, self.accuracy, self.log_likelihood_cost],
                feed_dict={self.x: batch_x, self.y: batch_y})
            self.training_stats.report(accuracy, cost)

        avg_accuracy, avg_cost, accuracy_summaries = self.training_stats.collect()
        global_step = self.get_global_step()
        print("Step %d training data accuracy: %g; cost: %g" % (global_step, avg_accuracy, avg_cost))
        if self.training_summary_writer is not None:
            activation_summaries = self.session.run(
                self.activation_summaries,
                feed_dict={self.x: batch_x, self.y: batch_y})
            self.training_summary_writer.add_summary(activation_summaries, global_step)
            self.training_summary_writer.add_summary(accuracy_summaries, global_step)


    def run(self, position):
        'Return a sorted list of (probability, move) tuples'
        processed_position = features.extract_features(position, features=self.features)
        probabilities = self.session.run(self.output, feed_dict={self.x: processed_position[None, :]})[0]
        return probabilities.reshape([go.N, go.N])

    def check_accuracy(self, test_data, batch_size=128):
        num_minibatches = test_data.data_size // batch_size
        weight_summaries = self.session.run(self.weight_summaries)

        for i in range(num_minibatches):
            batch_x, batch_y = test_data.get_batch(batch_size)
            accuracy, cost = self.session.run(
                [self.accuracy, self.log_likelihood_cost],
                feed_dict={self.x: batch_x, self.y: batch_y})
            self.test_stats.report(accuracy, cost)

        avg_accuracy, avg_cost, accuracy_summaries = self.test_stats.collect()
        global_step = self.get_global_step()
        print("Step %s test data accuracy: %g; cost: %g" % (global_step, avg_accuracy, avg_cost))

        if self.test_summary_writer is not None:
            self.test_summary_writer.add_summary(weight_summaries, global_step)
            self.test_summary_writer.add_summary(accuracy_summaries, global_step)

class StatisticsCollector(object):
    '''
    Accuracy and cost cannot be calculated with the full test dataset
    in one pass, so they must be computed in batches. Unfortunately,
    the built-in TF summary nodes cannot be told to aggregate multiple
    executions. Therefore, we aggregate the accuracy/cost ourselves at
    the python level, and then shove it through the accuracy/cost summary
    nodes to generate the appropriate summary protobufs for writing.
    '''
    graph = tf.Graph()
    with tf.device("/cpu:0"), graph.as_default():
        accuracy = tf.placeholder(tf.float32, [])
        cost = tf.placeholder(tf.float32, [])
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)
        cost_summary = tf.summary.scalar("log_likelihood_cost", cost)
        accuracy_summaries = tf.summary.merge([accuracy_summary, cost_summary], name="accuracy_summaries")
    session = tf.Session(graph=graph)

    def __init__(self):
        self.accuracies = []
        self.costs = []

    def report(self, accuracy, cost):
        self.accuracies.append(accuracy)
        self.costs.append(cost)

    def collect(self):
        avg_acc = sum(self.accuracies) / len(self.accuracies)
        avg_cost = sum(self.costs) / len(self.costs)
        self.accuracies = []
        self.costs = []
        summary = self.session.run(self.accuracy_summaries,
            feed_dict={self.accuracy:avg_acc, self.cost: avg_cost})
        return avg_acc, avg_cost, summary
