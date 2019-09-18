# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
"""Startup script for TensorFlow.

See the README for more information.
"""
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow import logging
from models import model_factory
from datasets import dataset_factory
from preprocessing import preprocessing_factory
from flags import FLAGS
from utils import cluster_utils, optimizer_utils, hooks_utils
from utils import misc_utils as utils


_DATASET_TRAIN_FILES = {
  'mock': '',
  'cifar10': 'cifar10_train.tfrecord',
  'mnist': 'mnist_train.tfrecord',
  'flowers': ','.join(['flowers_train_%s-of-00005.tfrecord' % (str(i).rjust(5, '0')) for i in xrange(5)]),
}

_DATASET_EVAL_FILES = {
  'mock': '',
  'cifar10': 'cifar10_test.tfrecord',
  'mnist': 'mnist_test.tfrecord',
  'flowers': ','.join(['flowers_validation_%s-of-00005.tfrecord' % (str(i).rjust(5, '0')) for i in xrange(5)]),
}


def log_trainable_or_optimizer_vars_info():
  if FLAGS.log_trainable_vars_statistics:
    utils.print_model_statistics()
  if FLAGS.log_optimizer_vars_statistics:
    utils.optimizer_statistics()


def get_hooks(params):
  """Specify hooks used for training or evaluation. See 'utils/hooks_utils.py' for more information."""
  return hooks_utils.get_train_hooks(params)


def get_tfrecord_files(train_or_eval_files, num_workers=1):
  """Split dataset by worker.

  Args:
      num_workers: String, the name of the dataset.
      file_pattern: The file pattern to use for matching the dataset source files.

  Returns:
      A file list.

  Raises:
      ValueError: If the dataset is unknown.
  """

  if FLAGS.dataset_name == 'mock':
    return []
  ret = []
  all_tfrecord_files = []
  dataset_dir = FLAGS.dataset_dir
  if dataset_dir is None:
    raise ValueError('Need to specify dataset, mock or real.')

  assert train_or_eval_files is not None
  files_list = train_or_eval_files.split(',')
  for file_name in files_list:
    all_tfrecord_files.append(os.path.join(dataset_dir, file_name))
  if (len(all_tfrecord_files) // num_workers) <= 0:
    raise ValueError('Require num_training_files_per_worker > 0 with num_training_files({}) < num_workers({}).'\
                     .format(len(all_tfrecord_files), num_workers))
  if len(all_tfrecord_files) % num_workers > 0:
    logging.warning(
      "{} files can not be distributed equally between {} workers.".format(len(all_tfrecord_files), num_workers))
  all_tfrecord_files.sort()
  for i in range(len(all_tfrecord_files)):
    if i % num_workers == FLAGS.task_index:
      ret.append(all_tfrecord_files[i])
  logging.info('Worker Host {} handles {} files including {}.'.format(FLAGS.task_index, len(ret), ret))
  return ret


def run_model(target, num_workers, global_step):
  ##########################
  #  Config learning_rate  #
  ##########################
  learning_rate = optimizer_utils.configure_learning_rate(FLAGS.num_sample_per_epoch, global_step)

  ##########################################################
  #  Config optimizer and Wrapper optimizer with PAI-Soar  #
  ##########################################################
  samples_per_step = FLAGS.batch_size
  optimizer = optimizer_utils.configure_optimizer(learning_rate)
  if FLAGS.enable_paisoar:
    import paisoar
    optimizer = paisoar.ReplicatedVarsOptimizer(optimizer, clip_norm=FLAGS.max_gradient_norm)
    ctx = paisoar.Config.get()
    samples_per_step *= len(ctx.device_indices) * num_workers

  #######################
  #  Config model func  #
  #######################
  model_fn = model_factory.get_model_fn(FLAGS.model_name,
                                        num_classes=FLAGS.num_classes,
                                        weight_decay=FLAGS.weight_decay,
                                        is_training=True)

  #############################
  #  Config dataset iterator  #
  #############################
  with tf.device('/cpu:0'):
    train_image_size = model_fn.default_image_size

    # split dataset by worker
    data_sources = get_tfrecord_files(_DATASET_TRAIN_FILES[FLAGS.dataset_name] or FLAGS.train_files, num_workers)

    # select the preprocessing func
    preprocessing_fn = preprocessing_factory.get_preprocessing(
      FLAGS.preprocessing_name or FLAGS.model_name,
      is_training=True) if (FLAGS.preprocessing_name or FLAGS.model_name) else None

    dataset_iterator = dataset_factory.get_dataset_iterator(FLAGS.dataset_name,
                                                            train_image_size,
                                                            preprocessing_fn,
                                                            data_sources,
                                                            FLAGS.reader)
  ###############################################
  #  Config loss_func and Wrapper with PAI-Soar #
  ###############################################
  accuracy = []
  def loss_fn():
    with tf.device('/cpu:0'):
      images, labels = dataset_iterator.get_next()
    logits, end_points = model_fn(images)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                  logits=tf.cast(logits, tf.float32), weights=1.0)
    if 'AuxLogits' in end_points:
      loss += tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                     logits=tf.cast(end_points['AuxLogits'], tf.float32),
                                                     weights=0.4)
    per_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), labels), tf.float32))
    accuracy.append(per_accuracy)
    return loss

  # wrapper loss_fn with PAI-Soar 2.0
  loss = optimizer.compute_loss(loss_fn, loss_scale=FLAGS.loss_scale) if FLAGS.enable_paisoar \
    else loss_fn()

  ########################
  #  Config train tensor #
  ########################
  train_op = optimizer.minimize(loss, global_step=global_step)

  ###############################################
  #  Log trainable or optimizer variables info, #
  #  including name and size.                   #
  ###############################################
  log_trainable_or_optimizer_vars_info()

  ################
  # Restore ckpt #
  ################
  if FLAGS.model_dir and FLAGS.task_type == 'finetune':
    utils.load_checkpoint()

  #########################
  # Config training hooks #
  #########################
  params = dict()
  if FLAGS.log_loss_every_n_iters > 0:
    tensors_to_log = {'loss': loss if isinstance(loss, tf.Tensor) else loss.replicas[0],
                      'accuracy': tf.reduce_mean(accuracy),
                      'lrate': learning_rate}
    params['tensors_to_log'] = tensors_to_log
    params['samples_per_step'] = samples_per_step
  hooks = get_hooks(params=params)

  ###########################
  # Kicks off the training. #
  ###########################
  logging.info('training starts.')

  with tf.train.MonitoredTrainingSession(
      target,
      is_chief=(FLAGS.task_index == 0),
      hooks=hooks) as sess:
    try:
      while not sess.should_stop():
        sess.run(train_op)
    except tf.errors.OutOfRangeError:
      print('All threads done.')
    except Exception as e:
      import sys
      import traceback
      logging.error(e.message)
      traceback.print_exc(file=sys.stdout)
  logging.info('training ends.')


def main(_):
  ##############################
  # Kicks off PAI-Soar or not. #
  ##############################
  if FLAGS.enable_paisoar:
    import paisoar
    paisoar.enable_replicated_vars(tensor_fusion_policy=FLAGS.tensor_fusion_policy,
                                   communication_policy=FLAGS.communication_policy,
                                   tensor_fusion_max_bytes=FLAGS.tensor_fusion_max_bytes)

  cluster_manager = cluster_utils.get_cluster_manager()
  with tf.device(cluster_manager.device_exp()):
    global_step = tf.train.get_or_create_global_step()

    if FLAGS.task_type in ['pretrain', 'finetune']:
      run_model(cluster_manager.get_target(), cluster_manager.num_workers(), global_step)

    else:
      raise ValueError('task_type [%s] was not recognized' % FLAGS.task_type)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.app.run()

