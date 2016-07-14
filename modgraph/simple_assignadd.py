import tensorflow as tf
import debug_utils

a = tf.Variable(1.1, name="a")
b = tf.constant(2.5, name="b")
aa = tf.assign_add(a, b, name="aa")

sess = tf.InteractiveSession()

run_opts = tf.RunOptions()
debug_utils.add_tensor_watch_all(sess, run_opts)
# debug_utils.add_tensor_watch(run_opts, "a", slot=0,
#                              debug_op="DebugRefIdentity")
# debug_utils.add_tensor_watch(run_opts, "b", slot=0)
# debug_utils.add_tensor_watch(run_opts, "a/initial_value", slot=0)

sess.run(tf.initialize_all_variables())

for _ in range(2):
  print(sess.run(aa, options=run_opts))
