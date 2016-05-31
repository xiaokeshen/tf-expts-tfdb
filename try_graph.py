import tensorflow as tf


# with tf.Graph().as_default() as graph:

a = tf.constant(1.1, name="a")
b = tf.constant(2.2, name="b")
c = tf.add(a, b, name="c")
d = tf.transpose(c, name="transpose")

# sess.run(d)

sess = tf.InteractiveSession()

graph = tf.get_default_graph()
ops = graph.get_operations()
for op in ops:
  print(op.name)
  for t in op.inputs:
    print("  Input: %s" % (t.name))
    print("    Eval result: %s" % repr(sess.run(t)))


#  print(graph.as_graph_def())
