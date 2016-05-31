import tensorflow as tf

a = tf.constant(1.1, name="a")
a1 = tf.constant(2.3, name="a1")
b = tf.constant(2.2, name="b")

def f_true():
  print("<")
  return tf.mul(a1, a, name="true_res")

def f_false():
  print(">")
  return tf.div(a1, b, name="false_res")

c = tf.cond(tf.less(a, b), f_true, f_false)

sess = tf.Session()
print(sess.run(c))
