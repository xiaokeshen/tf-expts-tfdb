import sys

import tensorflow as tf

sess = tf.Session("debug")

a = tf.Variable([1.1, 2.2], name="a")
b = tf.reduce_sum(a, name="b")

sess.run(tf.initialize_all_variables())

print("========================")
# Variable + reduce_sum = seg fault
sess.run(b)  # Seg fault
