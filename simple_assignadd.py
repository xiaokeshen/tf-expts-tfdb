import tensorflow as tf

sess = tf.Session()

x = tf.Variable([1.1, 2.2], name="x")
sess.run(tf.initialize_all_variables())

y = tf.constant([0.1, 0.2], name="y")

for i in range(10):
  aax = tf.assign_add(x, y)
  sess.run(aax)
