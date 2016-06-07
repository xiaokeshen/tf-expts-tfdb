import tensorflow as tf


sess = tf.Session("debug")

a = tf.constant(1.1, name="a")
b = tf.constant(2.2, name="b")

x = tf.constant(3.3, name="x")
y = tf.constant(4.4, name="y")

z = tf.mul(a, b)
result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))

print(sess.run(result))
