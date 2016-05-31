import tensorflow as tf

a = tf.Variable(1.1, name="a")
b = tf.Variable(2.2, name="b")
aaa = tf.assign_add(a, 0.1, name="aaa")

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# Note that handles can be obtained in batches
h_a, h_b = sess.run([tf.get_session_handle(a.value()),
                     tf.get_session_handle(b.value())])
print("** Before aaa: h_a.eval() = %s **" % repr(h_a.eval()))

raw_input()
print(sess.run(aaa))

# Node that the same handle gives the updated value
print("** After aaa: h_a.eval() = %s **" % repr(h_a.eval()))
