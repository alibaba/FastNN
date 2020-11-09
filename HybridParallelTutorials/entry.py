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

from dp_pipeline import train_bert_model
from dp_split import large_scale_classification


FLAGS = tf.app.flags.FLAGS


runner_map = {
    "dp_pipeline": train_bert_model,  # Bert + Pipeline Parallel
    "dp_split": large_scale_classification  # Resnet50 + Operator Sharding
}


def main(_):
  assert FLAGS.runner_name in runner_map, \
    "Wrong runner_name for {}, default images".format(FLAGS.runner_name)
  ####################
  # Kicks off Whale. #
  ####################
  import whale as wh
  if FLAGS.enable_whale:
    wh.init()
  layout = {"average": FLAGS.average}
  if FLAGS.runner_name == "large_scale_classification":
    layout = "all"
  cluster = wh.cluster(worker_hosts=FLAGS.worker_hosts,
                       ps_hosts=FLAGS.ps_hosts,
                       job_name=FLAGS.job_name,
                       rank=FLAGS.task_index,
                       layout=layout)
  config_proto = tf.ConfigProto(
      log_device_placement=False,
      allow_soft_placement=True,
      gpu_options=tf.GPUOptions(
          force_gpu_compatible=True,
          allow_growth=True))

  if FLAGS.task_type in ['pretrain', 'finetune']:
    with cluster:
      runner_map[FLAGS.runner_name].run_model(cluster, config_proto)
  else:
    raise ValueError('task_type [%s] was not recognized' % FLAGS.task_type)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.app.run()
