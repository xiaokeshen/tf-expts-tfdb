"""tfdb CLI demo based on MNIST."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np  # pylint: disable=unused-import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Import for tfdb and its CLI
import tfdb_cli

# Create a debug Session, different from a normal session
sess = tf.Session("debug")

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

batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
train_feed = {
    x: batch_xs,
    y_: batch_ys
}

# Initialize the Variables
print("\n=== Initializing all variables ===\n")
init_op = tf.initialize_all_variables()

# auto_step=True let the CLI automatically continues through all steps without
# breaking.
debug_cli = tfdb_cli.CommandLineDebugger(sess, init_op, feed=train_feed,
                                         auto_step=True)

# Do two steps of training
print("\n=== Running one step of training ===\n")
tfdb_cli.CommandLineDebugger(sess, train_step, num_times=2, feed=train_feed)
