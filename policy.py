'''
The neural network architecture is some mix of AlphaGo's input features and
Cazenave's resnet architecture.

See features.py for a list of input features used. All colors are flipped so
that it is always "black to play". Thus, the same policy is used to estimate
both white and black moves. However, in the case of the value network, the
value of komi, or whose turn to play, must also be passed in, because there
is an asymmetry between black and white there.

The policy and value networks share a majority of their architecture.
This helps the intermediate layers extract concepts that are relevant to both 
move prediction and score estimation.

Within the DNN, the layer width is configurable, but 128 is a good compromise
between network size and compute time. All layers use ReLu nonlinearities and
zero-padding for convolutions.

The policy and value networks can be evaluated independently or together;
if executed together, the shared part of the network only needs to be computed
once. When training, you must either alternate training both halves, or freeze
the shared part of the network, or else the half that isn't being trained will
start producing inaccurate outputs.
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
    def __init__(self,
                 k=128,
                 fc_width=1000,
                 num_shared_res_layers=8,
                 num_policy_res_layers=3,
                 num_value_fc_layers=3,
                 use_cpu=False):
        self.num_input_planes = sum(f.planes for f in features.DEFAULT_FEATURES)
        self.k = k
        self.fc_width = fc_width
        self.num_shared_res_layers = num_shared_res_layers
        self.num_policy_res_layers = num_policy_res_layers
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
        RL_global_step = tf.Variable(0, name="RL_global_step", trainable=False)
        x = tf.placeholder(tf.float32, [None, go.N, go.N, self.num_input_planes])
        move = tf.placeholder(tf.float32, shape=[None, go.N ** 2])
        outcome = tf.placeholder(tf.float32, shape=[None])
        # whether this example should be positively or negatively reinforced.
        # Set to 1 for positive, -1 for negative.
        reinforce_direction = tf.placeholder(tf.float32, shape=[])

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

        def _res_layer(x, layer_name):
            with tf.name_scope(layer_name):
                resnet_weights1 = _weight_variable([3, 3, self.k, self.k], name="W_conv_resnet1")
                resnet_weights2 = _weight_variable([3, 3, self.k, self.k], name="W_conv_resnet2")
                int_conv = tf.nn.relu(
                    _conv2d(x, resnet_weights1),
                    name="h_conv_intermediate")
                output = tf.nn.relu(
                    x + _conv2d(int_conv, resnet_weights2),
                    name="h_conv")
            return output, [resnet_weights1, resnet_weights2]

        def _fc_layer(x, layer_name):
            with tf.name_scope(layer_name):
                fc_weights = _weight_variable([x.get_shape()[-1], self.fc_width], name="W_fc")
                fc_biases = tf.Variable(tf.zeros([self.fc_width]), name="bias_fc")
                output = tf.nn.relu(
                    tf.matmul(x, fc_weights, name="h_fc") + fc_biases)
            return output, [fc_weights, fc_biases]


        # Keep track of all weights and activations, for debugging purposes.
        weight_vars = []
        activation_values = []

        # initial conv layer is 5x5
        W_conv_init55 = _weight_variable([5, 5, self.num_input_planes, self.k], name="W_conv_init55")
        W_conv_init11 = _weight_variable([1, 1, self.num_input_planes, self.k], name="W_conv_init11")
        h_conv_init = tf.nn.relu(
            tf.concat(
                [_conv2d(x, W_conv_init55), _conv2d(x, W_conv_init11)], 3),
            name="h_conv_init")
        weight_vars.extend([W_conv_init55, W_conv_init11])
        activation_values.append(h_conv_init)

        # followed by a series of shared resnet 3x3 conv layers
        _current_h = h_conv_init
        for i in range(self.num_shared_res_layers):
            _current_h, _resnet_weights = _res_layer(
                _current_h, "shared_layer" + str(i))
            weight_vars.extend(_resnet_weights)
            activation_values.append(_current_h)

        # here, the NN branches, feeding into two outputs: policy, value
        last_shared_activation = _current_h

        # policy half
        _current_h = last_shared_activation
        for i in range(self.num_policy_res_layers):
            _current_h, _resnet_weights = _res_layer(
                _current_h, "policy_layer" + str(i))
            weight_vars.extend(_resnet_weights)
            activation_values.append(_current_h)

        W_conv_final = _weight_variable([1, 1, self.k, 1], name="W_conv_final")
        b_conv_final = tf.Variable(tf.constant(0, shape=[go.N ** 2], dtype=tf.float32), name="b_conv_final")
        h_conv_final = _conv2d(_current_h_conv, W_conv_final)
        weight_vars.append(W_conv_final)
        activation_values.append(h_conv_final)

        logits = tf.reshape(h_conv_final, [-1, go.N ** 2]) + b_conv_final
        policy_output = tf.nn.softmax(logits)

        # value half
        _current_h = last_shared_activation.reshape([-1, go.N * go.N * self.k])
        for i in range(self.num_value_fc_layers):
            _current_h, _fc_vars = _fc_layer(
                _current_h, "value_layer" + str(i))
            weight_vars.extend(_fc_vars)
            activation_values.append(_current_h)

        W_fc_final = _weight_variable([self.fc_width, 1], name="W_fc_final")
        b_fc_final = tf.Variable(0, name="bias_fc_final")
        value_output = tf.nn.tanh(
            tf.matmul(_current_h, W_fc_final) + b_fc_final).reshape([-1])

        # training ops
        # policy training
        log_likelihood_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=move))
        learning_rate = tf.train.exponential_decay(1e-2, global_step, 4 * 10 ** 6, 0.5)
        policy_train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(log_likelihood_cost, global_step=global_step)
        policy_reinforce_step = tf.train.GradientDescentOptimizer(1e-2).minimize(
            log_likelihood_cost * reinforce_direction, global_step=RL_global_step)

        # value training
        mse_cost = tf.reduce_mean(tf.square(value_output - outcome))
        value_train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse_cost, global_step=global_step)

        # misc ops
        was_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(move, 1))
        accuracy = tf.reduce_mean(tf.cast(was_correct, tf.float32))


        weight_summaries = tf.summary.merge([
            tf.summary.histogram(weight_var.name, weight_var)
            for weight_var in weight_vars],
            name="weight_summaries"
        )

        activation_summaries = tf.summary.merge([
            tf.summary.histogram(act_var.name, act_var)
            for act_var in activation_values],
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
            try:
                self.saver.restore(self.session, save_file)
            except:
                # some wizardry here... basically, only restore variables
                # that are in the save file; otherwise, initialize them normally.
                from tensorflow.python.framework import meta_graph
                meta_graph_def = meta_graph.read_meta_graph_file(save_file + '.meta')
                stored_var_names = set([n.name
                    for n in meta_graph_def.graph_def.node
                    if n.op == 'VariableV2'])
                print(stored_var_names)
                var_list = [v for v in tf.global_variables()
                    if v.op.name in stored_var_names]
                # initialize all of the variables
                self.session.run(tf.global_variables_initializer())
                # then overwrite the ones we have in the save file
                # by using a throwaway saver, saved models are automatically
                # "upgraded" to the latest graph definition.
                throwaway_saver = tf.train.Saver(var_list=var_list)
                throwaway_saver.restore(self.session, save_file)


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
                [self.policy_train_step, self.accuracy, self.log_likelihood_cost],
                feed_dict={self.x: batch_x, self.move: batch_y, self.reinforce_direction: 1})
            self.training_stats.report(accuracy, cost)

        avg_accuracy, avg_cost, accuracy_summaries = self.training_stats.collect()
        global_step = self.get_global_step()
        print("Step %d training data accuracy: %g; cost: %g" % (global_step, avg_accuracy, avg_cost))
        if self.training_summary_writer is not None:
            activation_summaries = self.session.run(
                self.activation_summaries,
                feed_dict={self.x: batch_x, self.move: batch_y, self.reinforce_direction: 1})
            self.training_summary_writer.add_summary(activation_summaries, global_step)
            self.training_summary_writer.add_summary(accuracy_summaries, global_step)

    def reinforce(self, dataset, direction=1, batch_size=32):
        num_minibatches = dataset.data_size // batch_size
        for i in range(num_minibatches):
            batch_x, batch_y = dataset.get_batch(batch_size)
            self.session.run(
                self.policy_reinforce_step,
                feed_dict={self.x: batch_x, self.move: batch_y, self.reinforce_direction: direction})

    def run(self, position):
        'Return a sorted list of (probability, move) tuples'
        processed_position = features.extract_features(position)
        probabilities = self.session.run(self.output, feed_dict={self.x: processed_position[None, :]})[0]
        return probabilities.reshape([go.N, go.N])

    def run_many(self, positions):
        processed_positions = features.bulk_extract_features(positions)
        probabilities = self.session.run(self.output, feed_dict={self.x:processed_positions})
        return probabilities.reshape([-1, go.N, go.N])

    def check_accuracy(self, test_data, batch_size=128):
        num_minibatches = test_data.data_size // batch_size
        weight_summaries = self.session.run(self.weight_summaries)

        for i in range(num_minibatches):
            batch_x, batch_y = test_data.get_batch(batch_size)
            accuracy, cost = self.session.run(
                [self.accuracy, self.log_likelihood_cost],
                feed_dict={self.x: batch_x, self.move: batch_y, self.reinforce_direction: 1})
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
