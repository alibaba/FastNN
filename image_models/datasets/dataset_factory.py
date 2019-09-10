# Copyright (C) 2016 The TensorFlow Authors.
# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
"""A factory-pattern map which returns classification dataset iterator."""
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cifar10
import flowers
import mnist
from flags import FLAGS
from utils import dataset_utils

datasets_map = {
    'cifar10': cifar10,
    'flowers': flowers,
    'mnist': mnist,
}


def get_dataset_iterator(dataset_name, train_image_size, preprocessing_fn=None, data_sources=None, reader=None):
  with tf.device("/cpu:0"):
    if not dataset_name:
      raise ValueError('expect dataset_name not None.')
    if dataset_name not in datasets_map:
      raise ValueError('Name of network unknown %s' % dataset_name)
    if dataset_name == 'mock':
      return dataset_utils._create_mock_iterator(train_image_size)

    def parse_fn(example):
      with tf.device("/cpu:0"):
        image, label = datasets_map[dataset_name].parse_fn(example)
        if preprocessing_fn is not None:
          image = preprocessing_fn(image, train_image_size, train_image_size)
        if FLAGS.use_fp16:
          image = tf.cast(image, tf.float16)
        label -= FLAGS.labels_offset
        return image, label
    return dataset_utils._create_dataset_iterator(data_sources, parse_fn, reader)
