import numpy as np
import tensorflow as tf

def numpy_to_handle(val):
  dtype = val.dtype
  ph = tf.placeholder(dtype)
  handle_op = tf.get_session_handle(ph)
  sess = tf.get_default_session()
  handle_obj = sess.run(handle_op, feed_dict={ph: val})
  return handle_obj


def as_tensorflow(input):
  dtype = input._dtype
  ph, tensor = tf.get_session_tensor(dtype)
  return tensor, ph, input.handle


sess = tf.InteractiveSession()

x0 = np.array([[1, 2], [3, 4]], dtype=np.float32)
x0_handle = numpy_to_handle(x0)
print("x0_handle.eval() = %s" % repr(x0_handle.eval()))

x1 = np.array([[10, 20], [30, 40]], dtype=np.float32)
x1_handle = numpy_to_handle(x1)
print("x1_handle.eval() = %s" % repr(x1_handle.eval()))
print("x1_handle.handle: %s" % x1_handle.handle)

feed = {}

x0_tensor, x0_ph, x0_handle_handle = as_tensorflow(x0_handle)
print("x0_tensor = %s" % repr(x0_tensor))
print("x0_ph = %s" % repr(x0_ph))
print("x0_handle_handle = %s" % repr(x0_handle_handle))

feed[x0_ph] = x0_handle_handle

x1_tensor, x1_ph, x1_handle_handle = as_tensorflow(x1_handle)
print("x1_tensor = %s" % repr(x1_tensor))
print("x1_ph = %s" % repr(x1_ph))
print("x1_handle_handle = %s (%s)" %
      (repr(x1_handle_handle), repr(type(x1_handle_handle))))

feed[x1_ph] = x1_handle_handle
# feed[x1_ph] = np.array([[0, 0], [0, 1]], dtype=np.float32)

c = tf.add(x0_tensor, x1_tensor)

assert(isinstance(c, tf.Tensor))

c_deferred_handle = tf.get_session_handle(c)
print("c_deferred_handle = %s" % repr(c_deferred_handle))

c_concrete_handle = sess.run(c_deferred_handle, feed_dict=feed)
print("c_concrete_handle = %s" % repr(c_concrete_handle))

print("c_concrete_handle.eval() = %s" % repr(c_concrete_handle.eval()))
