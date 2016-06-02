from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np  # pylint: disable=unused-import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from six.moves import xrange

# Create a debug Session, different from a normal session
sess = tf.Session()

# MNIST data input object
mnist = input_data.read_data_sets("/tmp/mnist_data", one_hot=True)

# Toy parameters for a simple one-hidden-layer NN
IMAGE_PIXELS = 28
HIDDEN_UNITS = 4
NUM_TARGETS = 10
BATCH_SIZE = 5
LEARNING_RATE = 0.01

# Define variables, use zeros and ones as initial values for the ease of
# demonstrating debugger
hid_w = tf.Variable(
    tf.ones([IMAGE_PIXELS * IMAGE_PIXELS, HIDDEN_UNITS]), name="hid_w")
hid_b = tf.Variable(tf.zeros([HIDDEN_UNITS]), name="hid_b")

sm_w = tf.Variable(
    tf.ones([HIDDEN_UNITS, NUM_TARGETS]), name="sm_w")
sm_b = tf.Variable(tf.zeros([NUM_TARGETS]), name="sm_b")

x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS], name="x")
y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b, name="hid_lin")
hid = tf.nn.relu(hid_lin, name="hid")

y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b), name="y")
cross_entropy = -tf.reduce_sum(y_ *
                               tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
                               name="xent")

# The optimizer
opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)

# The training step
train_step = opt.minimize(cross_entropy, name="train_step")

# Initialize Variables
sess.run(tf.initialize_all_variables())

for i in xrange(100):
  batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
  train_feed = {
      x: batch_xs,
      y_: batch_ys
   }

  sess.run(train_step, feed_dict=train_feed)



