import tensorflow as tf

sess = tf.InteractiveSession()
a = tf.constant(1.1, name="a")
b = tf.constant(1.1, name="b")
c = tf.add(a, b, name="c")
d = tf.mul(a, b, name="d")
e = tf.div(c, d, name="e")

x = tf.Variable(11.0, name="x")
y = tf.log(x, name="y")

r = tf.add(e, y, name="r")

sess.run(tf.initialize_all_variables())
print("")
raw_input()

print(sess.run(r))
