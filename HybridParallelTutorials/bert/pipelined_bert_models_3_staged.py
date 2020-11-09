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
from bert.models import modeling
from shared_params import SHARED_FLAGS
from shared_utils import dataset_utils, optimizer_utils, hooks_utils, misc_utils as utils

import whale as wh


def build_output_layer_squad(final_hidden, input_ids, kwargs):

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
    "cls/squad/output_weights", [2, hidden_size],
    initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
    "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  # compute loss
  seq_length = modeling.get_shape_list(input_ids)[1]

  def compute_loss(logits, positions):
    one_hot_positions = tf.one_hot(
      positions, depth=seq_length, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits)
    loss = -tf.reduce_mean(
      tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
    return loss

  def def_loss():
    start_positions = kwargs["start_positions"]
    end_positions = kwargs["end_positions"]

    start_loss = compute_loss(start_logits, start_positions)
    end_loss = compute_loss(end_logits, end_positions)

    loss = (start_loss + end_loss) / 2.0
    return loss

  loss = def_loss()
  return loss


def run_model(cluster, config_proto, is_training=True):

  with wh.replica():
    with wh.pipeline(num_micro_batch=SHARED_FLAGS.num_micro_batch):
      with wh.stage():
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
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        if EXCLUSIVE_FLAGS.max_seq_length > bert_config.max_position_embeddings:
          raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (EXCLUSIVE_FLAGS.max_seq_length, bert_config.max_position_embeddings))

        if not is_training:
          bert_config.hidden_dropout_prob = 0.0
          bert_config.attention_probs_dropout_prob = 0.0

        input_shape = modeling.get_shape_list(next_batch['input_ids'], expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if next_batch['input_mask'] is None:
          input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
        else:
          input_mask = next_batch['input_mask']

        if next_batch['segment_ids'] is None:
          token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
        else:
          token_type_ids = next_batch['segment_ids']

        #with tf.variable_scope("bert", None):
        with tf.variable_scope("embeddings"):
          # Perform embedding lookup on the word ids.
          (embedding_output, embedding_table) = modeling.embedding_lookup(
            input_ids=next_batch['input_ids'],
            vocab_size=bert_config.vocab_size,
            embedding_size=bert_config.hidden_size,
            initializer_range=bert_config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=False)

          # Add positional embeddings and token type embeddings, then layer
          # normalize and perform dropout.
          embedding_output = modeling.embedding_postprocessor(
            input_tensor=embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=bert_config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=bert_config.initializer_range,
            max_position_embeddings=bert_config.max_position_embeddings,
            dropout_prob=bert_config.hidden_dropout_prob)

          input_tensor = embedding_output
          if SHARED_FLAGS.use_fp16:
            input_tensor = tf.cast(input_tensor, tf.float16)

        with tf.variable_scope("encoder1"):
          # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
          # mask of shape [batch_size, seq_length, seq_length] which is used
          # for the attention scores.
          attention_mask = modeling.create_attention_mask_from_input_mask(
            next_batch['input_ids'], input_mask)
          all_encoder_layers = modeling.transformer_model(
            input_tensor=input_tensor,
            attention_mask=attention_mask,
            hidden_size=bert_config.hidden_size,
            num_hidden_layers=SHARED_FLAGS.cut_layer1,
            num_attention_heads=bert_config.num_attention_heads,
            intermediate_size=bert_config.intermediate_size,
            intermediate_act_fn=modeling.get_activation(bert_config.hidden_act),
            hidden_dropout_prob=bert_config.hidden_dropout_prob,
            attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
            initializer_range=bert_config.initializer_range,
            do_return_all_layers=True)

      with wh.stage():
        with tf.variable_scope("encoder2"):
          all_encoder_layers = modeling.transformer_model(
            input_tensor=all_encoder_layers[-1],
            attention_mask=attention_mask,
            hidden_size=bert_config.hidden_size,
            num_hidden_layers=SHARED_FLAGS.cut_layer2,
            num_attention_heads=bert_config.num_attention_heads,
            intermediate_size=bert_config.intermediate_size,
            intermediate_act_fn=modeling.get_activation(bert_config.hidden_act),
            hidden_dropout_prob=bert_config.hidden_dropout_prob,
            attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
            initializer_range=bert_config.initializer_range,
            do_return_all_layers=True,
            start_layer_idx=SHARED_FLAGS.cut_layer1)

      with wh.stage():
        with tf.variable_scope("encoder"):
          all_encoder_layers = modeling.transformer_model(
            input_tensor=all_encoder_layers[-1],
            attention_mask=attention_mask,
            hidden_size=bert_config.hidden_size,
            num_hidden_layers=bert_config.num_hidden_layers - SHARED_FLAGS.cut_layer1 - SHARED_FLAGS.cut_layer2,
            num_attention_heads=bert_config.num_attention_heads,
            intermediate_size=bert_config.intermediate_size,
            intermediate_act_fn=modeling.get_activation(bert_config.hidden_act),
            hidden_dropout_prob=bert_config.hidden_dropout_prob,
            attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
            initializer_range=bert_config.initializer_range,
            do_return_all_layers=True,
            start_layer_idx=SHARED_FLAGS.cut_layer1 + SHARED_FLAGS.cut_layer2)

        with tf.variable_scope("cal_loss"):
          sequence_output = all_encoder_layers[-1]
          if SHARED_FLAGS.use_fp16:
            sequence_output = tf.cast(sequence_output, tf.float32)
          # The "pooler" converts the encoded sequence tensor of shape
          # [batch_size, seq_length, hidden_size] to a tensor of shape
          # [batch_size, hidden_size]. This is necessary for segment-level
          # (or segment-pair-level) classification tasks where we need a fixed
          # dimensional representation of the segment.
          with tf.variable_scope("pooler"):
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token. We assume that this has been pre-trained
            first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
            pooled_output = tf.layers.dense(
              first_token_tensor,
              bert_config.hidden_size,
              activation=tf.tanh,
              kernel_initializer=modeling.create_initializer(bert_config.initializer_range))

          input_ids = next_batch['input_ids']

          if EXCLUSIVE_FLAGS.model_type == 'mrc':
            loss = build_output_layer_squad(sequence_output, input_ids, kwargs)
          else:
            raise ValueError("model_type should be one of ['classification', "
                             "'regression', pretrain', 'mrc'].")

          saver = tf.train.Saver(
            var_list=tf.global_variables(),
            max_to_keep=2)

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