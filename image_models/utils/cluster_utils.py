# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from flags import FLAGS


def create_config_proto():
  """Returns session config proto."""
  config = tf.ConfigProto(
    log_device_placement=FLAGS.log_device_placement,
    inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
    intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads,
    allow_soft_placement=True,
    gpu_options=tf.GPUOptions(
      force_gpu_compatible=True,
      allow_growth=True))
  return config


def get_cluster_manager():
  """Returns the cluster manager to be used."""
  return GrpcClusterManager(create_config_proto())


class BaseClusterManager(object):
  """The manager for the cluster of servers running the fast-nn."""

  def __init__(self):
    assert FLAGS.job_name in ['worker'], 'job_name must be worker'
    if FLAGS.job_name and FLAGS.worker_hosts:
      cluster_dict = {'worker': FLAGS.worker_hosts.split(',')}
    else:
      cluster_dict = {'worker': ['127.0.0.1:0']}

    self._num_workers = len(cluster_dict['worker'])
    self._cluster_spec = tf.train.ClusterSpec(cluster_dict)
    self._device_exp = tf.train.replica_device_setter(
      worker_device="/job:worker/task:%d/" % FLAGS.task_index,
      cluster=self._cluster_spec)

  def get_target(self):
    """Returns a target to be passed to tf.Session()."""
    raise NotImplementedError('get_target must be implemented by subclass')

  def get_cluster_spec(self):
    return self._cluster_spec

  def num_workers(self):
    return self._num_workers

  def device_exp(self):
    return self._device_exp


class GrpcClusterManager(BaseClusterManager):
  """A cluster manager for a cluster networked with gRPC."""

  def __init__(self, config_proto):
    super(GrpcClusterManager, self).__init__()
    self._server = tf.train.Server(self._cluster_spec,
                                   job_name=FLAGS.job_name,
                                   task_index=FLAGS.task_index,
                                   config=config_proto,
                                   protocol=FLAGS.protocol)
    self._target = self._server.target

  def get_target(self):
    return self._target
