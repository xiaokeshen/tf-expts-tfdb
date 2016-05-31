import tensorflow as tf

a = tf.constant(10, name="a")
b = tf.constant(20, name="b")
c = tf.mul(a, b, name="c")

sess = tf.Session()
h_a = tf.get_session_handle(a)
h_b = tf.get_session_handle(b)
h_c = tf.get_session_handle(c)

print("c = %g" % sess.run(h_c).eval())
print("c = %g" % sess.run(h_c, feed_dict=).eval())
