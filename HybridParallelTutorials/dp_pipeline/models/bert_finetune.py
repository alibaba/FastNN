from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import logging

from bert.models import modeling


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


class BertFinetune(object):
    """
    Fintune Method based on Bert.
    """

    def __init__(self, bert_config_file, max_seq_length, is_training,
                 input_ids, input_mask, segment_ids, labels, use_one_hot_embeddings,
                 model_type='classification', kwargs=None):

        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        if max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (max_seq_length, bert_config.max_position_embeddings))

        self.model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        self.bert_config = bert_config
        self.kwargs = kwargs
        self.labels = labels
        self.input_ids = input_ids

        if model_type == 'classification':
            self.build_output_layer_classification()
        elif model_type == 'regression':
            self.build_output_layer_regression()
        elif model_type == 'mrc':
            self.build_output_layer_squad()
        elif model_type == 'pretrain':
            self.build_pretrain()
        else:
            raise ValueError("model_type should be one of ['classification', "
                             "'regression', pretrain', 'mrc'].")

        self.saver = tf.train.Saver(
            var_list=tf.global_variables(),
            max_to_keep=2)

    def restore(self, saver_directory, sess):
        checkpoint = tf.train.latest_checkpoint(saver_directory)
        if not checkpoint:
            logging.info("Couldn't find trained model at %s." % saver_directory)
        else:
            logging.info('restore from {}'.format(checkpoint))
            self.saver.restore(sess, checkpoint)

    def save(self, saver_directory, sess, step=None):
        logging.info("Save to %s." % saver_directory)
        if step is not None:
            self.saver.save(sess, saver_directory, global_step=step)
        else:
            self.saver.save(sess, saver_directory)

    def build_pretrain(self):
        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = self.get_masked_lm_output(
            self.bert_config,
            self.model.get_sequence_output(),
            self.model.get_embedding_table(),
            self.kwargs['masked_lm_positions'],
            self.kwargs['masked_lm_ids'],
            self.kwargs['masked_lm_weights'])

        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = self.get_next_sentence_output(
            self.bert_config,
            self.model.get_pooled_output(),
            self.kwargs['next_sentence_labels'])

        self.loss = masked_lm_loss + next_sentence_loss


        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(self.kwargs['masked_lm_ids'], [-1])
        masked_lm_weights = tf.reshape(self.kwargs['masked_lm_weights'], [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(self.kwargs['next_sentence_labels'], [-1])
        next_sentence_accuracy = tf.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_mean_loss = tf.metrics.mean(
            values=next_sentence_example_loss)

        self.eval_metric = {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

    def get_masked_lm_output(self, bert_config, input_tensor, output_weights, positions,
                             label_ids, label_weights):
        """Get loss and log probs for the masked LM."""
        input_tensor = gather_indexes(input_tensor, positions)

        with tf.variable_scope("cls/predictions"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        bert_config.initializer_range))
                input_tensor = modeling.layer_norm(input_tensor)

            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.get_variable(
                "output_bias",
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())
            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            # log_probs = tf.nn.log_softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits)

            label_ids = tf.reshape(label_ids, [-1])
            label_weights = tf.reshape(label_weights, [-1])

            one_hot_labels = tf.one_hot(
                label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            numerator = tf.reduce_sum(label_weights * per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            loss = numerator / denominator

        return (loss, per_example_loss, log_probs)

    def get_next_sentence_output(self, bert_config, input_tensor, labels):
        """Get loss and log probs for the next sentence prediction."""

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.
        with tf.variable_scope("cls/seq_relationship"):
            output_weights = tf.get_variable(
                "output_weights",
                shape=[2, bert_config.hidden_size],
                initializer=modeling.create_initializer(bert_config.initializer_range))
            output_bias = tf.get_variable(
                "output_bias", shape=[2], initializer=tf.zeros_initializer())

            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            # log_probs = tf.nn.log_softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits)
            labels = tf.reshape(labels, [-1])
            one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return (loss, per_example_loss, log_probs)

    def build_output_layer_regression(self):
        with tf.variable_scope("src-output-layer"):
            self.src_estimation = tf.contrib.layers.fully_connected(
                inputs=self.model.get_pooled_output(),
                num_outputs=1,
                activation_fn=None, #tf.nn.sigmoid
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

        self.src_prediction = self.src_estimation
        self.src_pred_cost = tf.add(
            tf.reduce_mean(tf.pow(self.src_prediction - self.labels, 2)),
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            name="src_cost")

        self.loss = self.src_pred_cost
        self.logits = self.src_estimation
        self.predictions = self.src_prediction
        self.accuracy = tf.metrics.accuracy(self.labels, self.predictions)
        print('loss', self.loss)
        print('logits', self.logits)
        print('predictions', self.predictions)
        print('accuracy', self.accuracy)

    def build_output_layer_classification(self):
        with tf.variable_scope("src-output-layer"):
            self.src_estimation = tf.contrib.layers.fully_connected(
                inputs=self.model.get_pooled_output(),
                num_outputs=2,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

        self.src_prediction = tf.contrib.layers.softmax(self.src_estimation)[:, 1]
        self.src_pred_cost = tf.add(
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.src_estimation, labels=self.labels)),
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            name="src_cost")

        self.loss = self.src_pred_cost
        self.logits = self.src_estimation
        self.predictions = self.src_prediction
        self.accuracy = tf.metrics.accuracy(self.labels, self.predictions)
        print('logits', self.logits)
        print('predictions', self.predictions)
        print('accuracy', self.accuracy)

    def build_output_layer_squad(self, is_training=False):
        final_hidden = self.model.get_sequence_output()

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
        seq_length = modeling.get_shape_list(self.input_ids)[1]

        def compute_loss(logits, positions):
            one_hot_positions = tf.one_hot(
                positions, depth=seq_length, dtype=tf.float32)
            log_probs = tf.nn.log_softmax(logits)
            loss = -tf.reduce_mean(
                tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
            return loss

        def def_loss():
            start_positions = self.kwargs["start_positions"]
            end_positions = self.kwargs["end_positions"]

            start_loss = compute_loss(start_logits, start_positions)
            end_loss = compute_loss(end_logits, end_positions)

            loss = (start_loss + end_loss) / 2.0
            return loss

        self.loss = def_loss()


    def build_output_layer(self, is_training):
        output_layer = self.model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [2, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable(
            "output_bias", [2], initializer=tf.zeros_initializer())

        print('output_layer', output_layer.shape)
        print('output_weights', output_weights.shape)
        print('output_bias', output_bias.shape)

        with tf.variable_scope("loss"):
            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            self.logits = logits
            log_probs = tf.nn.log_softmax(self.logits)
            print('logits', logits.shape)

            one_hot_labels = tf.one_hot(self.labels, depth=2,
                                        dtype=tf.float32)

            self.per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            self.loss = tf.reduce_mean(self.per_example_loss)

            self.predictions = tf.argmax(self.logits, axis=-1, output_type=tf.int32)
            self.accuracy = tf.metrics.accuracy(self.labels, self.predictions)

