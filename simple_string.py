import tensorflow as tf

sess = tf.InteractiveSession()

a = tf.constant("abc", name="a")

print(sess.run(a))
