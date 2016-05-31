import tensorflow as tf

v1 = tf.Variable(10.0, name="v1")
v2 = tf.Variable(20.0, name="v2")
b = tf.add(v1, v2, name="b")

aa = tf.assign_add(v1, b, name="aa")

sess = tf.Session()

sess.run(tf.initialize_all_variables())
