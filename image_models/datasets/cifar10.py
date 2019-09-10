# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
"""Parse example for the Cifar10 dataset.

The dataset scripts used to create the dataset can be found at:
datasets/download_and_convert_cifar10.py
"""
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def parse_fn(example):
  with tf.device("/cpu:0"):
    features = tf.parse_single_example(
      example,
      features={
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      }
    )
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    label = features['image/class/label']
    return image, label


