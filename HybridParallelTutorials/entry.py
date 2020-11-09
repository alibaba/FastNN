# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
"""Startup script for TensorFlow.

See the README for more information.
"""
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import logging

from algo_with_split import large_scale_classification
from bert import train_bert_models
from bert import pipelined_bert_models_3_staged
from shared_params import SHARED_FLAGS


runner_map = {
    "bert": train_bert_models, # Bert + Data Parallel
    "pipelined_bert_stage_3": pipelined_bert_models_3_staged, # Bert + Pipeline Parallel
    "large_scale_classification": large_scale_classification # Resnet50 + Operator Sharding
}


def main(_):
  assert SHARED_FLAGS.runner_name in runner_map, \
    "Wrong runner_name for {}, default images".format(SHARED_FLAGS.runner_name)
  ####################
  # Kicks off Whale. #
  ####################
  import whale as wh
  if SHARED_FLAGS.enable_whale:
    wh.init()
  layout = {"average": SHARED_FLAGS.average}
  if SHARED_FLAGS.runner_name == "large_scale_classification":
    layout = "all"
  cluster = wh.cluster(worker_hosts=SHARED_FLAGS.worker_hosts,
                       ps_hosts=SHARED_FLAGS.ps_hosts,
                       job_name=SHARED_FLAGS.job_name,
                       rank=SHARED_FLAGS.task_index,
                       layout=layout)
  config_proto = tf.ConfigProto(
      log_device_placement=False,
      allow_soft_placement=True,
      gpu_options=tf.GPUOptions(
          force_gpu_compatible=True,
          allow_growth=True))

  if SHARED_FLAGS.task_type in ['pretrain', 'finetune']:
    with cluster:
      runner_map[SHARED_FLAGS.runner_name].run_model(cluster, config_proto)
  else:
    raise ValueError('task_type [%s] was not recognized' % SHARED_FLAGS.task_type)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.app.run()
