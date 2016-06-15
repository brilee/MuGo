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
from load_data_sets import load_data_sets

kgs = load_data_sets([features.StoneColorFeature, features.LibertyFeature], "kgs-micro")

k = 64
num_intermediate_conv_layers = 3

x = tf.placeholder(tf.float32, [None, go.N, go.N, kgs.input_planes])
y = tf.placeholder(tf.float32, shape=[None, go.N ** 2])

#convenience functions for initializing weights and biases
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

# initial conv layer is 5x5
W_conv_init = weight_variable([5, 5, kgs.input_planes, k])
h_conv_init = tf.nn.relu(conv2d(x, W_conv_init))

# followed by a series of 3x3 conv layers
W_conv_intermediate = []
h_conv_intermediate = []
current_h_conv = h_conv_init
for i in range(num_intermediate_conv_layers):
    W_conv_intermediate.append(weight_variable([3, 3, k, k]))
    h_conv_intermediate.append(tf.nn.relu(conv2d(current_h_conv, W_conv_intermediate[-1])))
    current_h_conv = h_conv_intermediate[-1]

W_conv_final = weight_variable([1, 1, k, 1])
b_conv_final = tf.Variable(tf.constant(0, shape=[go.N ** 2], dtype=tf.float32))
h_conv_final = conv2d(h_conv_intermediate[-1], W_conv_final)
output = tf.nn.softmax(tf.reshape(h_conv_final, [-1, go.N ** 2]) + b_conv_final)

log_likelihood_cost = -tf.reduce_mean(tf.reduce_sum(tf.mul(tf.log(output), y), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(log_likelihood_cost)
was_correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(was_correct, tf.float32))

# Train!
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(10000):
    batch_x, batch_y = kgs.training.get_batch(16)
    if i % 10 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        print("Step %d, training data accuracy: %g" % (i, train_accuracy))
    if i % 1000 == 0:
        test_accuracy = sess.run(accuracy, feed_dict={x: kgs.test.input, y: kgs.test.labels})
        print("Step %d, test data accuracy: %g" % (i, test_accuracy))
    sess.run(train_step, feed_dict={x: batch_x, y: batch_y})

test_accuracy = sess.run(accuracy, feed_dict={x: kgs.test.input, y: kgs.test.labels})
print("Final test data accuracy: %g" % test_accuracy)
