
import debug_utils
import tensorflow as tf


config = tf.ConfigProto(
    graph_options=tf.GraphOptions(
        optimizer_options=tf.OptimizerOptions(do_constant_folding=False)))
sess = tf.Session(config=config)

a = tf.Variable("abc", name="str_a")
b = tf.Variable("def", name="str_b")
c = tf.string_join([a, b], name="str_c")

sess.run(tf.initialize_all_variables())

debug_utils.print_all_ops(sess)

run_opts = tf.RunOptions()
debug_utils.add_tensor_watch_all(sess, run_opts)

print(sess.run(c, options=run_opts))
