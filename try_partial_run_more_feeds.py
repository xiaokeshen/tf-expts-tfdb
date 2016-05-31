import sys
import time

import tensorflow as tf


a = tf.constant(1.0, name="a")
b = tf.constant(2.0, name="b")
c = tf.add(a, b, name="c")
d = tf.mul(a, b, name="d")
e = tf.div(c, d, name="e")
x = tf.constant(10.0, name="x")
y = tf.mul(e, x, name="y")

with tf.Session() as sess:
  h1 = sess.partial_run_setup([e], [])

  # This will error out because you didn't say a will be fed during the
  # partial_run_setup() call.
  print(sess.partial_run(h1, e, feed_dict={a: 1.1}))
  raw_input()
