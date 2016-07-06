import sys

import tensorflow as tf
import tensorflow.core.framework.tensor_pb2 as tensor_pb2

filename = sys.argv[1]
print(filename)

data = open(filename, "rb").read()

tp = tensor_pb2.TensorProto()
tp.ParseFromString(data)

# Use tensor_util.MakeNdarray to convert TensorProto to
