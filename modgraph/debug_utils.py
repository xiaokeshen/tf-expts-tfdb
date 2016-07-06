import os
import pickle as pkl
import re

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow as tf_session


OP_BLACKLIST = ["HistogramAccumulatorSummary",
                "HistogramSummary",
                "ImageSummary",
                "AudioSummary",
                "MergeSummary",
                "ScalarSummary"]


def print_all_ops(sess):
  g = sess.graph
  ops = g.get_operations()
  for op in ops:
    print(op.name)
    for inp in op.inputs:
      print("  %s" % inp.name)


def add_tensor_watch(run_options, name, slot=0, debug_op="DebugIdentity"):
  watch_opts = run_options.debug_tensor_watch_opts

  watch = watch_opts.add()
  watch.node_name = name
  watch.output_slot = slot
  watch.debug_ops.append(debug_op)


def add_tensor_watch_all(sess, run_options, debug_op="DebugIdentity",
                         op_blacklist=OP_BLACKLIST,
                         op_re_blacklist=[]):
  g = sess.graph
  ops = g.get_operations()
  for op in ops:
    for inp in op.inputs:
      if inp.name.count(":") != 1:
        continue

      node_name = inp.name.split(":")[0]

      node_base_name = node_name.split("/")[-1].split("_")[0]
      if op_blacklist.count(node_base_name) != 0:
        print("Skipping blacklisted op: %s" % node_name)
        continue

      node_name_items = node_name.split("/")
      if len(node_name_items) > 1:
        node_base_name = node_name_items[-2].split("_")[0]
        if op_blacklist.count(node_base_name) != 0:
          print("Skipping blacklisted op: %s" % node_name)
          continue

      skip_for_regex = False
      for regex in op_re_blacklist:
        if re.match(regex, node_name):
          skip_for_regex = True
          break
      if skip_for_regex:
        print("Skipping regex-blacklisted op: %s" % node_name)
        continue

      output_slot = int(inp.name.split(":")[1])
      add_tensor_watch(run_options, node_name, slot=output_slot)
      print("Added watch: %s : %d" % (node_name, output_slot))


def run_opts_with_tensor_watches(tensor_names):
  run_opts = tf.RunOptions()

  for tensor_name in tensor_names:
    node_name = tensor_name.split(":")[0]
    output_slot = int(tensor_name.split(":")[1])
    add_tensor_watch(run_opts, node_name, slot=output_slot)

  return run_opts


def walk_dump_dir(dump_dir, pickle_out_path=None, interactive=False):
  if pickle_out_path:
    data = {}

  for root, dirs, files in os.walk(dump_dir):
    if not dirs:
      for f in files:
        full_path = os.path.join(root, f)
        tensor_name = full_path.replace(dump_dir, "")
        if tensor_name[0] == "/":
          tensor_name = tensor_name[1:]

        if interactive:
          raw_input("Press Enter to load %s (tensor name: %s): " %
                    (full_path, tensor_name))
        else:
          print("Loading from file %s (tensor name: %s): " %
                (full_path, tensor_name))

        np_array = tf_session.TF_LoadTensor(full_path)[0]
        if pickle_out_path:
          data[tensor_name] = np_array

        print(np_array)
        print(np_array.shape)
        print("")

  if pickle_out_path:
    pkl.dump(data, open(pickle_out_path, "wb"))
