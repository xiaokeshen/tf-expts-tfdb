import tensorflow as tf

with tf.Session() as sess:
  a = tf.constant(10)
  b = tf.constant(5)
  c = tf.mul(a, b)

  h = tf.get_session_handle(c)
  h = sess.run(h)
  print(h)

  f, x = tf.get_session_tensor(tf.int32)
  y = tf.mul(x, 10)
  print(sess.run(y, feed_dict={f: h.handle}))
