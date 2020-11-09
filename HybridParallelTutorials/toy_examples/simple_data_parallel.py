"""Simple DNN algorithm of data parallelism with whale."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import whale as wh


def read_mock_data():
  """Read batch from mock data"""
  num_samples = 1000
  num_x = np.random.randint(0, 10, (num_samples, 20)).astype(dtype=np.float32)
  num_y = np.random.randint(0, 10, num_samples).astype(dtype=np.int64)
  dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
                           .batch(10).repeat(1)
  iterator = dataset.make_initializable_iterator()
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
  return iterator


def main(_):
  wh.init()
  with wh.cluster():
    with wh.replica():
      iterator = read_mock_data()
      x, labels = iterator.get_next()

      logits = tf.layers.dense(x, 2)
      logits = tf.layers.dense(logits, 10)
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  global_step = tf.train.get_or_create_global_step()
  optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
  train_op = optimizer.minimize(loss, global_step=global_step)

  hooks = [tf.train.StopAtStepHook(last_step=10)]
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=None,
      hooks=hooks) as sess:
    while not sess.should_stop():
      train_loss, _, step = sess.run([loss, train_op, global_step])
      print("Iteration %s , Loss: %s ." % (step, train_loss))
  print("Train Finished.")


if __name__ == '__main__':
  tf.app.run()
