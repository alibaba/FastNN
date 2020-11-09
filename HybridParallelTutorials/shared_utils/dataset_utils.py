# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
"""Contains utilities for converting datasets."""
from __future__ import division
from __future__ import print_function

import fnmatch
import os
from distutils.version import LooseVersion as Version
import tensorflow as tf
from tensorflow import logging
from tensorflow.contrib.data.python.ops import threadpool
from tensorflow.python.framework.versions import __version__

from shared_params import SHARED_FLAGS


def fetch_files(dataset_dir, file_pattern=None):
  dataset_dir = os.path.join(dataset_dir)
  files = tf.gfile.ListDirectory(dataset_dir)
  files = [os.path.join(dataset_dir, file_obj) for file_obj in files]
  if not file_pattern:
    logging.info(
        "[WhaleModelZoo]Regex on files with file_pattern {}.".format(file_pattern))
    files = [file_name for file_name in files \
             if fnmatch.fnmatch(file_name, file_pattern)]
  return files


def get_tfrecord_files(dataset_dir, train_or_eval_files=None, file_pattern="*"):
  """Get all dataset files.

  Args:
      dataset_dir: Path to dataset files.
      train_or_eval_files: Name of train or eval files.
      file_pattern: The file pattern to use for matching the dataset source files.

  Returns:
      A file list.

  Raises:
      ValueError: If the dataset is unknown.
  """
  with tf.device("/cpu:0"):
    if SHARED_FLAGS.dataset_name == 'mock':
      return []
    all_tfrecord_files = []
    if dataset_dir is None:
      raise ValueError("Need to specify dataset, mock or real.")

    if train_or_eval_files is not None:
      files_list = train_or_eval_files.split(',')
      for file_name in files_list:
        all_tfrecord_files.append(os.path.join(dataset_dir, file_name))
    else:
      all_tfrecord_files = fetch_files(dataset_dir, file_pattern)
    all_tfrecord_files.sort()
    logging.info("[WhaleModelZoo]All dataset files:{}".format(all_tfrecord_files))
    return all_tfrecord_files


def _experimental_data_namespace():
  return tf.data.experimental if hasattr(tf.data, "experimental") else tf.contrib.data


def _create_dataset_iterator(data_sources, batch_size, parse_fn, reader=None, is_training=True):
  with tf.device("/cpu:0"):
    experimental_data_namespace = _experimental_data_namespace()
    files = tf.data.Dataset.from_tensor_slices(data_sources)
    dataset = files.apply(experimental_data_namespace.parallel_interleave(
      tf.data.TFRecordDataset,
      cycle_length=1,
      buffer_output_elements=batch_size * 8,
      prefetch_input_elements=batch_size * 8))
    if SHARED_FLAGS.datasets_use_caching:
      dataset = dataset.cache()
    if is_training:
      dataset = dataset.apply(experimental_data_namespace.shuffle_and_repeat(
        buffer_size=SHARED_FLAGS.shuffle_buffer_size, count=SHARED_FLAGS.num_epochs))
    dataset = dataset.apply(experimental_data_namespace.map_and_batch(
      map_func=parse_fn, batch_size=batch_size, num_parallel_batches=SHARED_FLAGS.num_parallel_batches))

    dataset = dataset.prefetch(buffer_size=SHARED_FLAGS.prefetch_buffer_size)

    dataset = threadpool.override_threadpool(
      dataset,
      threadpool.PrivateThreadPool(
        SHARED_FLAGS.num_preprocessing_threads, display_name="input_pipeline_thread_pool"))

    if Version(__version__) >= Version("1.12.0") and \
        Version(__version__) < Version("1.14.0"):
      ds_iterator = dataset.make_initializable_iterator()
    elif Version(__version__) < Version("2.0"):
      ds_iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    else:
      raise RuntimeError("Version {} not supported.".format(Version(__version__)))
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, ds_iterator.initializer)

    return ds_iterator
