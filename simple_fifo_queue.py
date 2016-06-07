import tensorflow as tf

sess = tf.Session("debug")

q = tf.FIFOQueue(3, "float")
init = q.enqueue_many(([1.1, 2.2, 3.3],))

x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])

sess.run(init)

for i in xrange(1000):
  sess.run(q_inc)
