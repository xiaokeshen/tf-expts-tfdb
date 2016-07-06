import sys

import numpy as np
from tensorflow.python import pywrap_tensorflow as tf_session


if len(sys.argv) != 2:
  print("Usage: %s <TENSOR_PROTO_FILENAME>" % sys.argv[0])
  sys.exit(1)

filename = sys.argv[1]
np_array = tf_session.TF_LoadTensor(filename)[0]

print(type(np_array))
print(np_array)
