# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
"""Startup script for TensorFlow.

See the README for more information.
"""
from __future__ import division
from __future__ import print_function

import os
os.environ["WHALE_COMMUNICATION_SPARSE_AS_DENSE"]="True"
os.environ["WHALE_COMMUNICATION_NUM_COMMUNICATORS"]="2"


import tensorflow as tf

from bert.datasets import dataset_factory
from bert.exclusive_params import EXCLUSIVE_FLAGS
from bert.models.bert_finetune import BertFinetune
from shared_params import SHARED_FLAGS
from shared_utils import dataset_utils, optimizer_utils, hooks_utils, misc_utils as utils

import whale as wh


def run_model(cluster, config_proto):

  with wh.replica():
    #############################
    #  Config training files  #
    #############################
    # Fetch all dataset files
    data_sources = None if SHARED_FLAGS.dataset_name == 'mock' \
      else dataset_utils.get_tfrecord_files(SHARED_FLAGS.dataset_dir,
                                            file_pattern=SHARED_FLAGS.file_pattern)

    ########################
    #  Config train tensor #
    ########################
    global_step = tf.train.get_or_create_global_step()

    # Set dataset iterator
    dataset_iterator = dataset_factory.get_dataset_iterator(SHARED_FLAGS.dataset_name,
                                                            SHARED_FLAGS.batch_size,
                                                            data_sources,
                                                            SHARED_FLAGS.reader)

    # Set learning_rate
    learning_rate = optimizer_utils.configure_learning_rate(SHARED_FLAGS.num_sample_per_epoch, global_step)

    # Set optimizer
    samples_per_step = SHARED_FLAGS.batch_size
    optimizer = optimizer_utils.configure_optimizer(learning_rate)

    next_batch = dataset_iterator.get_next()
    kwargs = dict()
    kwargs['start_positions'] = next_batch['start_positions']
    kwargs['end_positions'] = next_batch['end_positions']
    kwargs['unique_ids'] = next_batch['unique_ids']
    import os
    bert_config_file = os.path.join(SHARED_FLAGS.model_dir, EXCLUSIVE_FLAGS.model_config_file_name)
    model = BertFinetune(bert_config_file=bert_config_file,
                         max_seq_length=EXCLUSIVE_FLAGS.max_seq_length,
                         is_training=True,
                         input_ids=next_batch['input_ids'],
                         input_mask=next_batch['input_mask'],
                         segment_ids=next_batch['segment_ids'],
                         labels=None,
                         use_one_hot_embeddings=False,
                         model_type=EXCLUSIVE_FLAGS.model_type,
                         kwargs=kwargs)
    loss = model.loss

  # Minimize loss
  train_op = optimizer.minimize(loss, global_step=global_step)


  #########################
  # Config training hooks #
  #########################
  params = dict()
  if SHARED_FLAGS.log_loss_every_n_iters > 0:
    tensors_to_log = {'loss': loss if isinstance(loss, tf.Tensor) else loss.replicas[0],
                      'lrate': learning_rate}
    params['tensors_to_log'] = tensors_to_log
    params['samples_per_step'] = samples_per_step
  hooks = hooks_utils.get_train_hooks(params=params)

  ###############################################
  #  Log trainable or optimizer variables info, #
  #  including name and size.                   #
  ###############################################
  utils.log_trainable_or_optimizer_vars_info(optimizer)

  ################
  # Restore ckpt #
  ################
  if SHARED_FLAGS.model_dir and SHARED_FLAGS.task_type == 'finetune':
    utils.load_checkpoint()

  ###########################
  # Kicks off the training. #
  ###########################
  with tf.train.MonitoredTrainingSession(
      config=config_proto,
      checkpoint_dir=SHARED_FLAGS.checkpointDir,
      hooks=hooks) as sess:
    #sess.run([tf.local_variables_initializer()])
    try:
      while not sess.should_stop():
        sess.run([train_op])
    except tf.errors.OutOfRangeError:
      print('All threads done.')
    except Exception as e:
      import sys
      import traceback
      tf.logging.error(e.message)
      traceback.print_exc(file=sys.stdout)
  tf.logging.info('training ends.')

