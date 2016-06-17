import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("session_target", "", "Session target string")

FLAGS = flags.FLAGS

sess = tf.Session(FLAGS.session_target)
a = tf.constant(1.1, name="a")
b = tf.constant(1.1, name="b")
c = tf.add(a, b, name="c")

print(sess.run(c))
