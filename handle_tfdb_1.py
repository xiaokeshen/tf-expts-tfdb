import tensorflow as tf

a = tf.Variable(1.1, name="a")
b = tf.constant(2.2, name="b")
c = tf.mul(a, b, name="c")
d = tf.div(a, b, name="d")
e = tf.add(c, d, name="e")

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# Modify the value of a; keep the value of b; run c
h_b = tf.get_session_handle(b).eval()  # TensorHandle

new_a = 1.2
print("** Modified a; kept b: %s **" %
      repr(sess.run(c, feed_dict={a: new_a, b: h_b.eval()})))

# Keep the value of a; modify the value of b; run c
h_a = tf.get_session_handle(a.value()).eval()
new_b = 3.0
print("** Kept a; modified b: %s **" %
      repr(sess.run(c, feed_dict={a: h_a.eval(), b: new_b})))

# Obtain the value of c, with the same input as above
h_c = sess.run(tf.get_session_handle(c), feed_dict={a: h_a.eval(), b: new_b})
print("** Value from handle to c: %s **" % h_c.eval())

# Obtain the value of d, with the same input as above
h_d = sess.run(tf.get_session_handle(d), feed_dict={a: h_a.eval(), b: new_b})
print("** Value from handle to d: %s **" % h_d.eval())

# Continue to node "e", using the handles to c and
print("")
print("")
raw_input()
h_e = sess.run(tf.get_session_handle(e),
               feed_dict={c: h_c.eval(), d: h_d.eval()})
print("** Value from handle to e: %s **" % h_d.eval())
