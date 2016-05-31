import argparse
import fileinput
from six.moves import xrange
import sys
# import thread
import threading
import time

import numpy as np
import tensorflow as tf

if __name__ == "__main__":
  A = tf.constant([[np.nan, -2.4], [1.1, 2.2], [1.0, -2.0]], name="A")
  A_trans = tf.transpose(A, name="A_trans")

  sess = tf.Session("debug")

  def start_debugging():
    print(sess.run(A_trans))

  t1 = threading.Thread(target=start_debugging)
  t1.start()

  while True:
    debug_cmd = raw_input("tfdb> ")
    if debug_cmd == "exit":
      break

    debugger_output = sess.debug(debug_cmd)
    print(debugger_output)
