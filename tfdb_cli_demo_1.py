import numpy as np
import tensorflow as tf
from tensorflow.python.client import debugger
import tfdb_cli

sess = tf.Session("debug")

a0 = np.array([[1.0, 0.0], [0.0, 2.0]]).astype(np.float32)
b0 = np.array([[0.0, -1.0], [0.0, 0.0]]).astype(np.float32)

is_placeholder=False
if is_placeholder:
  a = tf.placeholder(dtype=np.float32, name="a")
  b = tf.placeholder(dtype=np.float32, name="b")

  feed = {
      a: a0,
      b: b0
  }
else:
  a = tf.constant(a0, name="a")
  b = tf.constant(b0, name="b")

  feed = {}

c = tf.matmul(a, b, name="c")
ct = tf.transpose(c, name="ct")
ctc = tf.concat(0, [ct, c], name="ctc")

# Create the debugger CLI object
debug_cli = tfdb_cli.CommandLineDebugger(sess, ctc, feed=feed)

# Start CLI loop
debug_cli.cli_loop()
