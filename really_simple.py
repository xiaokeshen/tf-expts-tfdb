import tensorflow as tf

sess = tf.Session("debug")
a = tf.constant(1.1, name="a")
b = tf.constant(1.1, name="b")
c = tf.add(a, b, name="c")

print(sess.run(c))
