import sys
import time

import tensorflow as tf


a = tf.constant(1.1, name="a")
b = tf.constant(2.2, name="b")
c = tf.add(a, b, name="c")
d = tf.mul(a, b, name="d")
e = tf.div(c, d, name="e")
x = tf.constant(10.0, name="x")
y = tf.mul(e, x, name="y")

sess = tf.Session()
if sys.argv[1] == "--non-partial":
  print(sess.run(e))
elif sys.argv[1] == "--partial":
#  prh = sess.partial_run_setup([c, d, e, y], [a, b, x])
  prh = sess.partial_run_setup([d, y], [a, b, x])

  feed={a: 1.1, b: 2.2, x: 10.0}
  print(sess.partial_run(prh, d, feed_dict=feed))
  raw_input()

  print(sess.partial_run(prh, y, feed_dict={
    e: 100.0
  }))
  raw_input()

  # If you try to fetch d again, you'll get a StatusNotOK error
  # print(sess.partial_run(prh, d))
  # raw_input()

  # print(sess.partial_run(prh, e))
  # raw_input()
else:
  raise ValueError("Unrecognized option")
