import numpy as np
import tensorflow as tf

true_k = -1.2

k0 = 0.0
lr = 0.1

x = tf.placeholder(tf.float32, name="lrowi_x")
y = tf.placeholder(tf.float32, name="lrowi_y")

x0 = np.linspace(0, 10, 20).astype(np.float32)
feed = {
    x: x0,
    y: true_k * x0
}

k = tf.Variable([[k0]], name="lrowi_k")
y_ = k * x
sse = tf.reduce_sum(tf.square(y_ - y))

optimizer = tf.train.AdamOptimizer(lr)
opt_step = optimizer.minimize(sse)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(100):
  sess.run(opt_step, feed_dict=feed)
  print(sess.run(sse, feed_dict=feed))
  print(sess.run(k))
