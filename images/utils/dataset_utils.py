# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
"""Contains utilities for converting datasets."""
from __future__ import division
from __future__ import print_function

from flags import FLAGS
import tensorflow as tf
from tensorflow.contrib.data.python.ops import threadpool


def _create_mock_iterator(train_image_size):
  with tf.device('/cpu:0'):
    dataset = tf.data.Dataset.from_tensor_slices((
      tf.ones(shape=[FLAGS.batch_size * 10 * 8, train_image_size, train_image_size, 3], dtype=tf.float16),
      tf.ones(shape=[FLAGS.batch_size * 10 * 8, ], dtype=tf.int64)))
    dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.batch_size).prefetch(buffer_size=FLAGS.batch_size * 2)
    mock_iterate_op = dataset.make_one_shot_iterator()

    return mock_iterate_op


def _experimental_data_namespace():
  return tf.data.experimental if hasattr(tf.data, "experimental") else tf.contrib.data


def _create_dataset_iterator(data_sources, parse_fn, reader=None):
  with tf.device("/cpu:0"):
    experimental_data_namespace = _experimental_data_namespace()
    files = tf.data.Dataset.from_tensor_slices(data_sources)
    dataset = files.apply(experimental_data_namespace.parallel_interleave(
      tf.data.TFRecordDataset,
      cycle_length=1,
      buffer_output_elements=FLAGS.batch_size * 8,
      prefetch_input_elements=FLAGS.batch_size * 8))
    if FLAGS.datasets_use_caching:
      dataset = dataset.cache()
    dataset = dataset.apply(experimental_data_namespace.shuffle_and_repeat(
      buffer_size=FLAGS.shuffle_buffer_size, count=FLAGS.num_epochs))
    dataset = dataset.apply(experimental_data_namespace.map_and_batch(
      map_func=parse_fn, batch_size=FLAGS.batch_size, num_parallel_batches=FLAGS.num_parallel_batches))

    dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)

    dataset = threadpool.override_threadpool(
      dataset,
      threadpool.PrivateThreadPool(
        FLAGS.num_preprocessing_threads, display_name='input_pipeline_thread_pool'))

    ds_iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, ds_iterator.initializer)

    return ds_iterator
