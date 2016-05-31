"""Alternative tfdb implementation based on persistent tensors"""

import sys

import tensorflow as tf

a = tf.constant([[1, 2], [3, 4]], name="a")
b = tf.constant([[10, 20], [30, 40]], name="b")
c = tf.add(a, b, name="c")
d = tf.transpose(c, name="d")

e = tf.constant([[10, 20], [30, 40]], name="e")
f = tf.mul(d, e, name="f")

sess = tf.Session()
g = tf.get_default_graph()

print("****** get_session_handle a ******")
h_a = sess.run(tf.get_session_handle(a))
print("****** get_session_handle b ******")
h_b = sess.run(tf.get_session_handle(b))

print("****** run(c) ******")
print(sess.run(c))

print("****** get_session_handle c ******")
h_c = sess.run(tf.get_session_handle(c))

sys.exit(0)

print("****** get_session_handle d ******")
h_d = sess.run(tf.get_session_handle(d))
print("****** get_session_handle e ******")
h_e = sess.run(tf.get_session_handle(e))
h_f = sess.run(tf.get_session_handle(f))

print("****** Done with get_session_handle calls ******")

# print(sess.run(c, feed_dict={a: [[2, 3], [2, 3]], b: [[20, 30], [20, 30]]}))
# print("****** run c with feed ******")
# print(sess.run(c, feed_dict={a: h_a.eval(), b: h_a.eval()}))

print("****** run d with feed ******")
print(sess.run(d, feed_dict={c: h_c.eval()}))

# print("****** run d without feed ******")
# print(sess.run(d))
