import sys

import numpy as np
import tensorflow as tf

sess = tf.Session()

N = 2
np_x = np.linspace(0.0, 10.0, N)
np_y = -2.0 * np_x + 3.0

np_x = np_x[:, np.newaxis]
np_x = np.concatenate([np.ones([N, 1]), np_x], axis=1)

np_y = np_y[:, np.newaxis]

x = tf.constant(np_x, name="x", dtype=tf.float32)
y = tf.constant(np_y, name="y", dtype=tf.float32)
K = tf.Variable([[0.0], [1.0]], name="K")  # [b, k]
# b = tf.Variable(0.0, name="b")

# y_ = k * x + b
y_ = tf.matmul(x, K)
# sse = tf.reduce_sum(y_)  # Seg fault
# sse = tf.reduce_sum(np_y)  # NO Seg fault
# sse = tf.reduce_sum(np_x)  # NO Seg fault
sse = tf.reduce_sum(K)

sess.run(tf.initialize_all_variables())
sess.run(sse)

sys.exit(0)

# sqe = tf.reduce_sum(tf.square(y - y_))
# sqe = tf.reduce_sum(y - y_)
sqe = tf.reduce_sum(y_)

opt = tf.train.GradientDescentOptimizer(1e-6)
train_op = opt.minimize(sqe)

# sess = tf.InteractiveSession()
sess = tf.Session()

sess.run(tf.initialize_all_variables())
print("=====")

print(sess.run(train_op))
print("=====")

print(sess.run(k))
print("=====")

print(sess.run(b))
