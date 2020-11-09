# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
"""Startup script for classification task.

See the README for more information.
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import tensorflow as tf
import whale as wh

import resnet_model
from shared_params import SHARED_FLAGS


tf.app.flags.DEFINE_integer("class_num", 10000, "")
FLAGS = tf.app.flags.FLAGS

batch_size = SHARED_FLAGS.batch_size
class_num = FLAGS.class_num


def get_mock_iterator():
  dataset = tf.data.Dataset.from_tensor_slices((
      tf.random.uniform([batch_size * 2, 224, 224, 3],
                        minval=0.0, maxval=255.0, dtype=tf.float32),
      tf.random.uniform([batch_size * 2, ],
                        minval=0, maxval=class_num, dtype=tf.int64)))
  dataset = dataset.repeat(200)
  dataset = dataset.batch(batch_size).prefetch(buffer_size=batch_size * 2)
  mock_iterate_op = dataset.make_initializable_iterator()
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                       mock_iterate_op.initializer)
  return mock_iterate_op


def run_model(cluster, config_proto):

  with wh.replica(devices=cluster.slices[0]):
    iterator = get_mock_iterator()
    images, labels = iterator.get_next()
    feature_extract_fn = resnet_model.imagenet_resnet_v2(50)
    features = feature_extract_fn(images, is_training=True)

  with wh.split(devices=cluster.slices[0]):
    logits = tf.layers.dense(features, class_num)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  global_step = tf.train.get_or_create_global_step()
  optimizer = tf.train.AdamOptimizer(learning_rate=0.9)
  train_op = optimizer.minimize(loss, global_step=global_step)

  hooks = []
  hooks = [tf.train.StopAtStepHook(last_step=200)]
  with tf.train.MonitoredTrainingSession(
      config=config_proto, hooks=hooks) as sess:
    while not sess.should_stop():
      starttime = time.time()
      train_loss, _, step = sess.run([loss, train_op, global_step])
      endtime = time.time()
      print("[Iteration {} ], Loss: {:.6} , Time: {:.4} ." \
          .format(step, train_loss, endtime-starttime))
  print("[Finished]")
