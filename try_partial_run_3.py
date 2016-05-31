import sys

import tensorflow as tf


w = tf.Variable(1.0, name="w")
init = tf.initialize_all_variables()

x = tf.placeholder(tf.float32, name="x")
slow = tf.exp(w * x, name="slow")
cost = x * slow + x * slow
# grad = tf.gradients(cost, w)[0]
opt = tf.train.GradientDescentOptimizer(0.01)
train_step = opt.minimize(cost, name="train_step")

x_feed = 3.0
feed = {x : x_feed}

if len(sys.argv) == 2 and sys.argv[1] == "--partial":
 with tf.Session() as sess:
   sess.run(init)
   h = sess.partial_run_setup([slow, cost, train_step, w], [x])
   raw_input()

   print("*** w = %g ***" % sess.partial_run(h, w, feed_dict=feed))
   raw_input()

   # Parital-run slow
   print(sess.partial_run(h, slow))
   raw_input()

   print(sess.partial_run(h, cost))
   raw_input()

   # You cannot do the following: train_step is a no-output op (None output),
   # which falls in "target_list", instead of "fetch_list", and partial_run
   # doesn't like that.
   # print(sess.partial_run(h, train_step))

   print("*** w = %g ***" % sess.partial_run(h, w))
   raw_input()
elif len(sys.argv) == 2 and sys.argv[1] == "--non-partial":
  # Note that the
  with tf.Session() as sess:
    sess.run(init)
    raw_input()

    print(sess.run(cost, feed_dict=feed))
    raw_input()

    print(sess.run(train_step, feed_dict=feed))
    raw_input()

    print(sess.run(w, feed_dict=feed))
    raw_input()
elif len(sys.argv) == 2 and sys.argv[1] == "--feed":
  with tf.Session() as sess:
    sess.run(init)
    raw_input()

    w_val, slow_val, cost_val = \
        sess.run([w, slow, cost], feed_dict=feed)
    print("w = %g" %  w_val)
    print("slow = %g" % slow_val)
    print("cost = %g" % cost_val)
    raw_input()

    grad_val = sess.run(train_step, feed_dict={
      x: x_feed,
#     w: w_val,
      slow: slow_val,
      cost: cost_val
    })
    raw_input()

    print(sess.run(w))
