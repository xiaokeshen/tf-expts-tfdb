import sys

import tensorflow as tf
from tensorflow.python.ops import session_ops

c1 = tf.constant(1.1, name="c1")
v1 = tf.Variable(1.0, name="v1")
b = tf.constant(3.0, name="b")
b1 = tf.log(b, name="b1")
c = tf.mul(v1, b1, name="c")

v2 = tf.Variable(2.0, name="v2")
d = tf.mul(c, v2, name="d")

opt = tf.train.GradientDescentOptimizer(0.01)
train_op = opt.minimize(d)

init_op = tf.initialize_all_variables()

if len(sys.argv) == 2 and sys.argv[1] == "--whole":
  with tf.Session() as sess:
    sess.run(init_op)

    print("Before: v1 = %g" % sess.run(v1))
    print("Before: v2 = %g" % sess.run(v2))

    sess.run(train_op)

    print("After: v1 = %g" % sess.run(v1))  # 0.94, 0.978028
    print("After: v2 = %g" % sess.run(v2))  # 1.97, 1.98901

elif len(sys.argv) == 2 and sys.argv[1] == "--feed-val":
  with tf.Session() as sess:
    sess.run(init_op)

    v1_val = sess.run(v1)
    b_val = sess.run(b)
    c_val = sess.run(c)
    v2_val = sess.run(v2)
    d_val = sess.run(d)

    print("v1_val = %g" % v1_val)
    print("v2_val = %g" % v2_val)

    sess.run(train_op, feed_dict={
      v1: v1_val,
      b: b_val,
      c: c_val,
      v2: v2_val,
      d: d_val
    })

elif len(sys.argv) == 2 and sys.argv[1] == "--feed-handle":
  sess = tf.Session()
  sess.run(init_op)

  b_h = sess.run(session_ops.get_session_handle(b))
  c_h = sess.run(session_ops.get_session_handle(c))
  d_h = sess.run(session_ops.get_session_handle(d))

elif len(sys.argv) == 2 and sys.argv[1] == "--feed-run":
  with tf.Session() as sess:
    sess.run(init_op)

    b_val = sess.run(b)
    b1_val = sess.run(b1)
    c_val = sess.run(c)
    d_val = sess.run(d)

    print("b_val = %g" % b_val)
    print("b1_val = %g" % b1_val)
    print("c_val = %g" % c_val)
    print("d_val = %g" % d_val)

    print("Before: v1 = %g" % sess.run(v1))
    print("Before: v2 = %g" % sess.run(v2))

    print("------")
    raw_input()
    sess.run(train_op, feed_dict={b: b_val, b1: b1_val, c: c_val, d: d_val})
    # sess.run(train_op)

    print("After: v1 = %g" % sess.run(v1))
    print("After: v2 = %g" % sess.run(v2))

  # Variables cannot be used with get_session_handle()
  # sess.run(session_ops.get_session_handle(v1))

else:
  raise ValueError("Unsupported option")
