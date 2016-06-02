import tensorflow as tf

sess = tf.Session()

a = tf.Variable(100.0, name="a")
b = tf.Variable(200.0, name="b")
c = tf.constant(30.0, name="c")
d = tf.add(a, b, name="d")
e = tf.add(c, d, name="e")

opt = tf.train.GradientDescentOptimizer(0.01)
step = opt.minimize(e)

sess.run(tf.initialize_all_variables())
print("=====")
raw_input()

sess.run(step)

