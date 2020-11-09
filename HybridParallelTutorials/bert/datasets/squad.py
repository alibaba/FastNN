# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
"""Parse example for the squad dataset."""
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from bert.exclusive_params import EXCLUSIVE_FLAGS


def parse_fn(example):
  with tf.device("/cpu:0"):
    features = tf.parse_single_example(
      example,
      features={
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([EXCLUSIVE_FLAGS.max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([EXCLUSIVE_FLAGS.max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([EXCLUSIVE_FLAGS.max_seq_length], tf.int64),
        "start_positions": tf.FixedLenFeature([], tf.int64),
        "end_positions": tf.FixedLenFeature([], tf.int64)
      }
    )
    return features