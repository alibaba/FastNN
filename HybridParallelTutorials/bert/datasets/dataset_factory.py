# Copyright (C) 2016 The TensorFlow Authors.
# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
"""A factory-pattern map which returns classification dataset iterator."""
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from bert.datasets import squad
from shared_utils import dataset_utils

datasets_map = {
    'squad': squad,
}


def get_dataset_iterator(dataset_name, batch_size, data_sources=None, reader=None):
  with tf.device("/cpu:0"):
    if not dataset_name:
      raise ValueError('Expect dataset_name not None.')
    if dataset_name not in datasets_map:
      raise ValueError('Name of network unknown %s for bert.' % dataset_name)
    if not data_sources:
      raise ValueError('Expect train files list not None.')

    return dataset_utils._create_dataset_iterator(data_sources, batch_size, datasets_map[dataset_name].parse_fn, reader)
