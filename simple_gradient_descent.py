import tensorflow as tf

sess = tf.Session()

a = tf.Variable(100.0, name="a")
opt = tf.train.GradientDescentOptimizer(0.01)
step = opt.minimize(a)

sess.run(tf.initialize_all_variables())
print("=====")
raw_input()

for i in range(100):
  sess.run(step)

print("=====")
raw_input()

sess.run(a)

