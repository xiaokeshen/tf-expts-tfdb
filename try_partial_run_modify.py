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
  h1 = sess.partial_run_setup([e], [a, b])
  print(sess.partial_run(h1, e, feed_dict={a: 1.0, b: 2.0}))
  raw_input()

  # Partial run without injection. This will lead to the recompute of e
  h2 = sess.partial_run_setup([y], [x])
  print(sess.partial_run(h2, y, feed_dict={x: 10.0}))
  raw_input()

  # Partial run with injection, in which we modify e.
  # This will not lead to the recomute of e.
  h3 = sess.partial_run_setup([y], [e, x])
  print(sess.partial_run(h3, y, feed_dict={e: 3.5, x: 10.0}))
  raw_input()
