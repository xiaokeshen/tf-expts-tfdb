import sys

import argparse
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--no_caching", dest="caching", action="store_false")

args = ap.parse_args()

a = tf.Variable(10.0, name="a")
b = tf.Variable(20.0, name="b")
c = tf.mul(a, b, name="c")

d = tf.Variable(30.0, name="d")
e = tf.Variable(40.0, name="e")
f = tf.div(d, e, name="f")

g = tf.add(c, f, name="g")

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# Get graph elements
g = sess.graph
ops = g.get_operations()
for op in ops:
  print(op.name)

a_read = g.as_graph_element("a/read:0")
b_read = g.as_graph_element("b/read:0")

# Compute a and b
h_a = sess.run(tf.get_session_handle(a_read))
h_b = sess.run(tf.get_session_handle(b_read))

feed = {}
feed[a] = h_a.eval()
feed[b] = h_b.eval()

print("--- About to compute c ---")
raw_input()

h_c = sess.run(tf.get_session_handle(c), feed_dict=feed)
val_c = h_c.eval()
print("c value = %s" % repr(val_c))

# Compute d and e
d_read = g.as_graph_element("d/read:0")
e_read = g.as_graph_element("e/read:0")

h_d = sess.run(tf.get_session_handle(d_read))
h_e = sess.run(tf.get_session_handle(e_read))

feed = {}
feed[c] = h_c.eval()
feed[e] = h_e.eval()

print("--- About to compute f ---")
raw_input()

h_f = sess.run(tf.get_session_handle(f), feed_dict=feed)
val_f = h_f.eval()
print("f value = %s" % repr(val_f))

# Compute g, using immediately available feeds
g_output = g.as_graph_element("g:0")


if args.caching:
  feed = {}
  feed[c] = val_c
  feed[f] = val_f

  print("--- About to compute f (From feed) ---")
  raw_input()

  h_g = sess.run(tf.get_session_handle(g_output), feed_dict=feed)
  val_g = h_g.eval()
  print("g value = %s" % repr(val_g))

  sys.exit(0)

else:
  print("--- About to compute f (Without feed) ---")
  raw_input()

  h_g = sess.run(tf.get_session_handle(g_output))
  val_g = h_g.eval()
  print("g value = %s" % repr(val_g))
