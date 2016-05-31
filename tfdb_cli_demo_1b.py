import numpy as np
import tensorflow as tf
from tensorflow.python.client import debugger
import tfdb_cli

sess = tf.Session("debug")

m = tf.constant(
    np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float32),
    name="M")
mt = tf.transpose(m, name="Mt")

sess = tf.Session("debug")

# Create the debugger CLI object
debug_cli = tfdb_cli.CommandLineDebugger(sess, mt)

# Start CLI loop
debug_cli.cli_loop()
