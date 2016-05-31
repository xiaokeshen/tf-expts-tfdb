import sys

import tensorflow as tf
from tensorflow.python.ops import session_ops


# Return a handle.
a = tf.constant(10.0, name="a")
v = tf.Variable(1.0, name="v")
aav = tf.assign_add(v, a)


with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  h_v = session_ops.get_session_handle(v).eval()
  print(h_v)

  # Feed a tensor handle.
  result = sess.run(aav, feed_dict={v: h_v.handle})
