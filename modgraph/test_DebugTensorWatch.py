import tensorflow as tf

run_opts = tf.RunOptions()

watch = run_opts.debug_tensor_watch_opts.add()
watch.node_name="b"
watch.output_slot=0
watch.debug_ops.append("DebugIdentity")

watch = run_opts.debug_tensor_watch_opts.add()
watch.node_name="c"
watch.output_slot=0
watch.debug_ops.append("DebugIdentity")

sess = tf.InteractiveSession()

a = tf.Variable(2.2, name="a")
b = tf.add(a, a, name="b")
c = tf.mul(b, b, name="c")
d = tf.div(c, c, name="d")

sess.run(tf.initialize_all_variables())
print(sess.run(d, options=run_opts))
