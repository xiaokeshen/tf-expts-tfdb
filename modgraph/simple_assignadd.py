import tensorflow as tf
import debug_utils

a = tf.Variable(1.1, name="a")
b = tf.mul(2.5, a, name="b")
aa = tf.assign_add(a, b, name="aa")

sess = tf.InteractiveSession()

run_opts = tf.RunOptions()
debug_utils.add_tensor_watch(run_opts, "a/initial_value")

# Print graph structure
debug_utils.print_all_ops(sess)

sess.run(tf.initialize_all_variables())


run_opts = debug_utils.run_opts_with_tensor_watches(
    ["a:0", "b:0", "a/read:0", "b/x:0"])

for _ in xrange(2):
  print(sess.run(aa, options=run_opts))
