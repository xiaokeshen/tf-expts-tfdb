import numpy as np

import tensorflow as tf

v1 = tf.Variable(np.array([[1, 2], [3, 4]]), dtype=np.float32, name="v1")
a = tf.reduce_sum(v1, name="a")

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# == This will fail because v1 is a Variable, not a Tensor == #
# h_v1 = tf.get_session_handle(v1)
# h_v1 = tf.get_session_handle(sess.run(v1))

# == This will error out, using ref(), which gives read Tensor  == #
# h_v1 = sess.run(tf.get_session_handle(v1.ref()))
# print(h_v1)
# print(type(h_v1))
# print(h_v1.eval())

# == This will error out, using ref(), which gives read Tensor  == #
h_v1 = sess.run(tf.get_session_handle(v1.value()))
print(h_v1)
print(type(h_v1))
print(h_v1.eval())



# h_a = sess.run(tf.get_session_handle(a))
# print(h_a)
# print(type(h_a))
# print(h_a.eval())
