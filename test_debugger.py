import sys
import threading
import time

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class DebuggerTest(test_util.TensorFlowTestCase):

  def setUp(self):
    # TODO(cais): Proper mutex locking to push down to
    self._init_delay_sec = 0.1
    self._delay_sec = 0.02

    self._debug_round_res = None

    self._debug_sess = tf.Session("debug")
    self._non_debug_sess = tf.Session()

  def tearDown(self):
    self._debug_sess.close()

  def debugMainThread(self, node, feed=None):
    def func():
      self._debug_sess.run(node, feed_dict=feed)

    return func

  def startDebugMainThread(self, node, feed=None):
    thr = threading.Thread(target=self.debugMainThread(node, feed=feed))
    thr.start()

    time.sleep(self._init_delay_sec)
    return thr

  def step(self):
    output = self._debug_sess.debug("step")
    return output

  def where(self):
    output = self._debug_sess.debug("where")
    return output

  def inspect_value(self, node_name):
    output = self._debug_sess.debug("print %s" % node_name)
    node_value = output["node_value"]

    if node_value is not None:
      self._debug_round_res = node_value

    return node_value

  def inject_value(self, node_name, new_value):
    self._debug_sess.debug("inject_value %s" % node_name,
                           feed={node_name: new_value})

  def auto_step(self, do_inspect=True, val_replace=None):
    while True:
      self.step()

      time.sleep(self._delay_sec)
      debugger_output = self.where()

      node_just_completed = debugger_output["completed_nodes"][-1]
      print("Node just completed: %s" % node_just_completed)

      if do_inspect:
        node_value = self.inspect_value(node_just_completed)

        if node_value is not None:
          print("  Value: %s" % repr(node_value))

        # Perform value replacement
        if val_replace is not None and node_just_completed in val_replace:
          # For variable, we match "${NODE_NAME}/read"
          print("*** Python calling inject_value ***")

          mapping_func = val_replace[node_just_completed]
          new_node_value = mapping_func(node_value)
          print("new_node_value =", new_node_value)

          self.inject_value(node_just_completed, new_node_value)

      if debugger_output["is_completed"]:
        self.step()
        break

  def testPlaceHolderContext(self):
    with tf.Session("debug") as sess:
      A = tf.constant(2.2, name="phc_A")
      B = tf.constant(3.3, name="phc_B")

  def testPlaceHolderNonDebugSession(self):
    A = tf.placeholder(tf.float32, shape=(2, 2), name="nds_phA")
    B = tf.placeholder(tf.float32, shape=(2, 2), name="nds_phB")
    y = tf.matmul(tf.transpose(A), B) * 2.0

    feed = {
        A: np.array([[1, 2], [3, 4]]).astype(np.float32),
        B: np.array([[1, 0], [0, -1]]).astype(np.float32)
    }

    output = self._non_debug_sess.run(y, feed_dict=feed)
    self.assertAllClose(np.array([[2, -6], [4, -8]]), output)

  def testPlaceHolder(self):
    A = tf.placeholder(tf.float32, name="phA")
    B = tf.placeholder(tf.float32, name="phB")
    y = tf.matmul(tf.transpose(A), B) * 2

    feed = {
        A: np.array([[1, 2, 3]]),
        B: np.array([[-1, 0, 1]])
    }

    t1 = self.startDebugMainThread(y, feed=feed)

    while True:
      self.step()

      time.sleep(self._delay_sec)

      debugger_output = self.where()
      node_just_completed = debugger_output["completed_nodes"][-1]
      print("Node just completed: %s" % node_just_completed)

      node_value = self.inspect_value(node_just_completed)

      if node_value is not None:
        print("  Value: %s" % repr(node_value))

      if node_just_completed == "phA" or node_just_completed == "_recv_phA_0":
          new_node_value = 2.0 * node_value
          print("new_node_value =", new_node_value)

          self.inject_value(node_just_completed, new_node_value)

      if debugger_output["is_completed"]:
        self.step()
        break

    t1.join()

    expected_res = np.array([[-2, 0, 2],
                             [-4, 0, 4],
                             [-6, 0, 6]]) * 2
    self.assertAllClose(expected_res, self._debug_round_res)

  def testStringNoInjection(self):
    str1 = tf.constant("123", name="strni_str1")
    num1 = tf.string_to_number(str1, name="strni_num1")

    t1 = self.startDebugMainThread(num1)
    self.auto_step()
    t1.join()

    self.assertAllClose(123.0, self._debug_round_res)

  def disable_testStringWithInjection(self):
    # TODO(cais): Make injecting str values work
    str1 = tf.constant("123", name="strwi_str1")
    num1 = tf.string_to_number(str1, name="strwi_num1")

    def replace_string(x):
      return np.array("456")

    val_replace = {
      "strwi_str1": replace_string
    }

    t1 = self.startDebugMainThread(num1)
    self.auto_step(val_replace=val_replace)
    t1.join()

    self.assertAllClose(123.0, self._debug_round_res)

  def testVariablesNoInjection(self):
    A0 = 10.0
    B0 = 20.0
    lr = 0.01

    A = tf.Variable(A0, dtype=tf.float32, name="tni_A")
    B = tf.Variable(B0, dtype=tf.float32, name="tni_B")
    p = tf.mul(A, B)

    optimizer = tf.train.GradientDescentOptimizer(lr)
    opt_step = optimizer.minimize(p)

    init_op = tf.initialize_all_variables()

    # Initialize variables
    t1 = self.startDebugMainThread(init_op)
    self.auto_step(do_inspect=False)
    t1.join()

    # Perform calculation
    t2 = self.startDebugMainThread(opt_step)
    self.auto_step()
    t2.join()

    # Get the final variable value
    t3 = self.startDebugMainThread(A)
    self.auto_step()
    t3.join()

    self.assertAllClose(A0 - lr * B0, self._debug_round_res)

  def testVariablesWithInjection(self):
    A0 = 10.0
    B0 = 20.0
    lr = 0.01

    A = tf.Variable(A0, dtype=tf.float32, name="twi_A")
    B = tf.Variable(B0, dtype=tf.float32, name="twi_B")
    p = tf.mul(A, B)

    optimizer = tf.train.GradientDescentOptimizer(lr)
    opt_step = optimizer.minimize(p)

    init_op = tf.initialize_all_variables()

    # Initialize variables
    t1 = self.startDebugMainThread(init_op)
    self.auto_step(do_inspect=False)
    t1.join()

    # Perform calculation
    t2 = self.startDebugMainThread(opt_step)

    def multiply_by_10(x):
      return np.array(10.0 * x).astype(np.float32)

    val_replace = {
      "twi_A": multiply_by_10,
      "twi_B": multiply_by_10
    }

    self.auto_step(val_replace=val_replace)
    t2.join()

    # Get the final variable value
    t3 = self.startDebugMainThread(A)
    self.auto_step()
    t3.join()

    self.assertAllClose(multiply_by_10(A0) - lr * multiply_by_10(B0),
                        self._debug_round_res)

  def testLinRegressOptimWithInjection(self):
    true_k = -1.2

    k0 = 1.0
    lr = 1e-2

    x = tf.placeholder(tf.float32, name="lrowi_x")
    y = tf.placeholder(tf.float32, name="lrowi_y")

    x0 = np.linspace(0, 10, 20).astype(np.float32)
    feed = {
      x: x0,
      y: true_k * x0
    }

    k = tf.Variable(k0, name="lrowi_k")
    y_ = k * x
    sse = tf.reduce_sum(tf.square(y_ - y))

    optimizer = tf.train.AdamOptimizer(lr)
    opt_step = optimizer.minimize(sse)

    init_op = tf.initialize_all_variables()

    # Initialize variables
    t1 = self.startDebugMainThread(init_op)
    self.auto_step(do_inspect=False)
    t1.join()

    one_step_k = None

    # Perform calculation
    # TODO(cais): Repeated injection (inject more than once) doesn't work yet.
    # This is probably related some unintended address overwriting.
    # Tensor::CopyFromInternal merely copies the reference
    for i in range(4):
      t2 = self.startDebugMainThread(opt_step, feed=feed)
      self.auto_step()
      t2.join()

      # Get the final variable value
      t3 = self.startDebugMainThread(k)
      self.auto_step()
      t3.join()

      if one_step_k is None:
        self.assertAllClose(0.99, self._debug_round_res)
        one_step_k = self._debug_round_res
      else:
        # Assert that running the optimization step after dialing back the
        # value leads to the same result as before.
        self.assertAllClose(one_step_k, self._debug_round_res)

      # Get the final variable value
      def set_k_back_to_k0(x):
        return np.array(k0).astype(np.float32).copy()

      # Force set the value of k to initial (k0)
      t3 = self.startDebugMainThread(k)
      self.auto_step(val_replace={"lrowi_k": set_k_back_to_k0})

      t3.join()
      # raw_input()

      # Verify that the injection has taken effect
      t3 = self.startDebugMainThread(k)
      self.auto_step()
      t3.join()

      self.assertAllClose(k0, self._debug_round_res)


if __name__ == "__main__":
  googletest.main()
