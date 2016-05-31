# import tensorflow as tf
# from tensorflow.models.image.cifar10 import cifar10

# # CIFAR10 data input pipeline
# cifar10.maybe_download_and_extract()
# images, labels = cifar10.distorted_inputs()

# with tf.Session() as sess:
#   logits = cifar10.inference(images)
#   top_k_op = tf.nn.in_top_k(logits, labels, 1)

#   print("Running logits")
#   sess.run(top_k_op)

from datetime import datetime
import os.path
import time
import threading

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.image.cifar10 import cifar10
import tfdb_cli

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    sess = tf.Session("debug")

    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    # sess.run(init)
    print("Calling tfdb_cli")
    tfdb_cli.CommandLineDebugger(sess, init)

    # Start the queue runners.
    def start_queue():
      tf.train.start_queue_runners(sess=sess)

    start_queue_thr = threading.Thread(target=start_queue)
    start_queue_thr.start()

    start_time = time.time()
    sess.run(train_op)
    duration = time.time() - start_time

    num_examples_per_step = FLAGS.batch_size
    examples_per_sec = num_examples_per_step / duration
    sec_per_batch = float(duration)

    loss_value = sess.run(loss)

    format_str = ('%s: step, loss = %.2f (%.1f examples/sec; %.3f '
                  'sec/batch)')
    print (format_str % (datetime.now(), loss_value,
                         examples_per_sec, sec_per_batch))


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
