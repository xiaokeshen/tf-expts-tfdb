import tensorflow as tf


sess = tf.InteractiveSession()

a = tf.Variable(10.0, name="a")
ia = tf.identity(a, name="ia")

b = tf.constant(5.0, name="b")

c = tf.assign_add(a, b, name="c")

sess.run(tf.initialize_all_variables())
sess.run(c)
