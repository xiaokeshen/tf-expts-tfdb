import tensorflow as tf

with tf.Session() as sess:
  a = tf.constant(10)
  b = tf.constant(5)
  c = tf.mul(a, b)
  h = tf.get_session_handle(c)
  v = tf.mul(a, c)
  h, v = sess.run([h, v])
