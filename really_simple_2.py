import tensorflow as tf

sess = tf.Session()

a = tf.constant([1.1, 2.2], name="a")
b = tf.constant([1.1, -2.2], name="b")
c = tf.add(a, b, name="c")
d = tf.concat(0, [a, b], name="d")
e = tf.concat(0, [b, a], name="e")
f = tf.mul(d, e, name="f")

print(sess.run(f))
