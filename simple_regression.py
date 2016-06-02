
import numpy as np
import tensorflow as tf

N = 10
np_x = np.linspace(0.0, 10.0, N)
np_y = -2.0 * np_x + 3.0

x = tf.constant(np_x, name="x", dtype=tf.float32)
y = tf.constant(np_y, name="y", dtype=tf.float32)
k = tf.Variable(0.0, name="k")
b = tf.Variable(0.0, name="b")

y_ = k * x + b
sqe = tf.reduce_sum(tf.square(y - y_))

opt = tf.train.GradientDescentOptimizer(1e-6)
train_op = opt.minimize(sqe)

sess = tf.InteractiveSession()

sess.run(tf.initialize_all_variables())
print("=====")
raw_input()

print(sess.run(train_op))
print("=====")
raw_input()

print(sess.run(k))
print("=====")
raw_input()

print(sess.run(b))
