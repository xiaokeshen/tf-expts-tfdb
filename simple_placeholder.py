import tensorflow as tf

sess = tf.Session("debug")

a = tf.placeholder(tf.float32, name="3")
b = a * a

print(sess.run(b, feed_dict={a: 1.1}))
