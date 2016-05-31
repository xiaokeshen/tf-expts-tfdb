import tensorflow as tf

a = tf.placeholder(tf.float32, name="a")
b = tf.placeholder(tf.float32, name="b")
c = tf.add(a, b, name="c")
d = tf.placeholder(tf.float32, name="d")
e = tf.add(c, d, name="e")

f = tf.mul(c, e, name="f")
g = tf.div(c, e, name="g")
r = tf.add(f, g, name="r")

with tf.Session() as sess:
  h = sess.partial_run_setup([c, e], [a, b, d])
  raw_input()

  print("c = %g" % sess.partial_run(h, c, feed_dict={a: 10.0, b: 20.0}))
  raw_input()
  print("e = %g" % sess.partial_run(h, e, feed_dict={d: 30.0}))
  raw_input()

  h2 = sess.partial_run_setup([r], [c, e])
  print("r = %g" % sess.partial_run(h2, r, feed_dict={c: 31.0, e: 61.0}))

  # h2 = sess.partial_run_setup([r], [])
  # print("r = %g" % sess.partial_run(h2, r, feed_dict={}))

print("Success")
