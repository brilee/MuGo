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
import tensorflow as tf

import features
import go

class PolicyNetwork(object):
    def __init__(self, num_input_planes, k=32, num_int_conv_layers=3):
        self.num_input_planes = num_input_planes
        self.k = k
        self.num_int_conv_layers = num_int_conv_layers
        self.set_up_network()

    def set_up_network(self):
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

        train_step = tf.train.AdamOptimizer(1e-4).minimize(log_likelihood_cost)
        was_correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(was_correct, tf.float32))

        saver = tf.train.Saver()
        # save everything to self.
        for name, thing in locals().items():
            if not name.startswith('_'):
                setattr(self, name, thing)

    def initialize_variables(self, save_file=None):
        self.session = tf.Session()
        if save_file is None:
            self.session.run(tf.initialize_all_variables())
        else:
            self.saver.restore(self.session, save_file)

    def save_variables(self, save_file):
        self.saver.save(self.session, save_file)

    def train(self, training_data, batch_size=16):
        num_minibatches = training_data.data_size // batch_size
        for i in range(num_minibatches):
            batch_x, batch_y = training_data.get_batch(batch_size)
            if i % 10 == 0:
                train_accuracy = self.session.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y})
                print("Step %d, training data accuracy: %g" % (i, train_accuracy))
            self.session.run(self.train_step, feed_dict={self.x: batch_x, self.y: batch_y})

    def run(self, position):
        processed_position = features.DEFAULT_FEATURES.extract(position)
        return self.session.run(self.output, feed_dict={self.x: processed_position[None, :]})[0]

    def check_accuracy(self, test_data):
        test_accuracy = self.session.run(self.accuracy, feed_dict={self.x: test_data.input, self.y: test_data.labels})
        print("Test data accuracy: %g" % test_accuracy)

