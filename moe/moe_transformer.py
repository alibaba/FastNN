# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Transformer with MoE(Mixture of Experts) module."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range  # pylint: disable=redefined-builtin

import tensorflow.compat.v1 as tf

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry
try:
  from tensor2tensor.models.transformer import Transformer
except:  # pylint: disable=bare-except
  pass

import moe_ffn

# Alias some commonly reused layers, here and elsewhere.
transformer_ffn_layer = moe_ffn.moe_ffn_layer


def moe_transformer_encoder_layer(layer,
                                  x,
                                  encoder_self_attention_bias,
                                  hparams,
                                  attention_dropout_broadcast_dims,
                                  save_weights_to=None,
                                  make_image_summary=True):
  """A single transformer encoder layer with MoE module."""
  with tf.variable_scope("self_attention"):
    if layer < hparams.get("num_area_layers", 0):
      max_area_width = hparams.get("max_area_width", 1)
      max_area_height = hparams.get("max_area_height", 1)
      memory_height = hparams.get("memory_height", 1)
    else:
      max_area_width = 1
      max_area_height = 1
      memory_height = 1
    y = common_attention.multihead_attention(
        common_layers.layer_preprocess(x, hparams),
        None,
        encoder_self_attention_bias,
        hparams.attention_key_channels or hparams.hidden_size,
        hparams.attention_value_channels or hparams.hidden_size,
        hparams.hidden_size,
        hparams.num_heads,
        hparams.attention_dropout,
        attention_type=hparams.self_attention_type,
        max_relative_position=hparams.max_relative_position,
        heads_share_relative_embedding=(
            hparams.heads_share_relative_embedding),
        add_relative_to_values=hparams.add_relative_to_values,
        save_weights_to=save_weights_to,
        make_image_summary=make_image_summary,
        dropout_broadcast_dims=attention_dropout_broadcast_dims,
        max_length=hparams.get("max_length"),
        vars_3d=hparams.get("attention_variables_3d"),
        activation_dtype=hparams.get("activation_dtype", "float32"),
        weight_dtype=hparams.get("weight_dtype", "float32"),
        hard_attention_k=hparams.get("hard_attention_k", 0),
        gumbel_noise_weight=hparams.get("gumbel_noise_weight", 0.0),
        max_area_width=max_area_width,
        max_area_height=max_area_height,
        memory_height=memory_height,
        area_key_mode=hparams.get("area_key_mode", "none"),
        area_value_mode=hparams.get("area_value_mode", "none"),
        training=(hparams.get("mode", tf.estimator.ModeKeys.TRAIN)
                  == tf.estimator.ModeKeys.TRAIN))
    x = common_layers.layer_postprocess(x, y, hparams)
  with tf.variable_scope("moe-ffn"):
    y, _ = transformer_ffn_layer(
        common_layers.layer_preprocess(
            x, hparams),
        hparams)
    x = common_layers.layer_postprocess(x, y, hparams)
    tf.logging.info("moe transformer encoder layer {}".format(layer))

  return x


def moe_transformer_encoder(encoder_input,
                            encoder_self_attention_bias,
                            hparams,
                            name="moe-encoder",
                            nonpadding=None,
                            save_weights_to=None,
                            make_image_summary=True,
                            attn_bias_for_padding=None):
  """A stack of transformer moe layers.
  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This must either be
      passed in, which we do for "packed" datasets, or inferred from
      encoder_self_attention_bias.  The knowledge about padding is used
      for pad_remover(efficiency) and to mask out padding in convolutional
      layers.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses
    attn_bias_for_padding: Padded attention bias in case a unidirectional
      encoder is being used where future attention is masked.
  Returns:
    y: a Tensors
  """
  x = encoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_encoder_layers or hparams.num_hidden_layers)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
      value=hparams.attention_dropout)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
      value={
          "use_bias": "false",
          "num_heads": hparams.num_heads,
          "hidden_size": hparams.hidden_size
      })

  with tf.variable_scope(name):
    if nonpadding is not None:
      padding = 1.0 - nonpadding
    else:
      attention_bias = encoder_self_attention_bias
      if attn_bias_for_padding is not None:
        attention_bias = attn_bias_for_padding
      padding = common_attention.attention_bias_to_padding(attention_bias)
      nonpadding = 1.0 - padding
    for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        x = moe_transformer_encoder_layer(
            layer,
            x,
            encoder_self_attention_bias,
            hparams,
            attention_dropout_broadcast_dims,
            save_weights_to,
            make_image_summary)
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(x, hparams)


def moe_transformer_decoder_layer(decoder_input,
                                  decoder_self_attention_bias,
                                  layer_idx,
                                  hparams,
                                  encoder_output=None,
                                  encoder_decoder_attention_bias=None,
                                  cache=None,
                                  decode_loop_step=None,
                                  save_weights_to=None,
                                  make_image_summary=False,
                                  layer_collection=None,
                                  recurrent_memory_by_layer=None,
                                  chunk_number=None):
  """A single transformer decoder layer with MoE module."""
  x, _ = transformer.transformer_self_attention_layer(
      decoder_input=decoder_input,
      decoder_self_attention_bias=decoder_self_attention_bias,
      layer_idx=layer_idx,
      hparams=hparams,
      encoder_output=encoder_output,
      encoder_decoder_attention_bias=encoder_decoder_attention_bias,
      cache=cache,
      decode_loop_step=decode_loop_step,
      save_weights_to=save_weights_to,
      make_image_summary=make_image_summary,
      layer_collection=layer_collection,
      recurrent_memory_by_layer=recurrent_memory_by_layer,
      chunk_number=chunk_number)

  layer = layer_idx
  layer_name = "layer_%d" % layer
  with tf.variable_scope(layer_name):
    with tf.variable_scope("ffn"):
      y, _ = transformer_ffn_layer(
          common_layers.layer_preprocess(
              x, hparams),
          hparams)
      x = common_layers.layer_postprocess(x, y, hparams)
      return x


def moe_transformer_decoder(decoder_input,
                            encoder_output,
                            decoder_self_attention_bias,
                            encoder_decoder_attention_bias,
                            hparams,
                            cache=None,
                            decode_loop_step=None,
                            name="decoder",
                            save_weights_to=None,
                            make_image_summary=True,
                            layer_collection=None,
                            recurrent_memory_by_layer=None,
                            chunk_number=None):
  """A stack of transformer layers.
  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention (see
      common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
      attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop. Only used
      for inference on TPU.
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used to mask out
      padding in convolutional layers.  We generally only need this mask for
      "packed" datasets, because for ordinary datasets, no padding is ever
      followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights for
      visualization; the weights tensor will be appended there under a string
      key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the KFAC
      optimizer. Default is None.
    recurrent_memory_by_layer: Optional dict, mapping layer names to instances
      of transformer_memory.RecurrentMemory. Default is None.
    chunk_number: an optional integer Tensor with shape [batch] used to operate
      the recurrent_memory.
  Returns:
    y: a Tensors
  """
  x = decoder_input

  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_decoder_layers or hparams.num_hidden_layers)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
      value=hparams.attention_dropout)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
      value={
          "use_bias": "false",
          "num_heads": hparams.num_heads,
          "hidden_size": hparams.hidden_size
      })

  with tf.variable_scope(name):
    for layer_idx in range(hparams.num_decoder_layers or
                           hparams.num_hidden_layers):
      x = moe_transformer_decoder_layer(
          x,
          decoder_self_attention_bias,
          layer_idx,
          hparams,
          encoder_decoder_attention_bias=encoder_decoder_attention_bias,
          encoder_output=encoder_output,
          cache=cache,
          decode_loop_step=decode_loop_step,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          layer_collection=layer_collection,
          recurrent_memory_by_layer=recurrent_memory_by_layer,
          chunk_number=chunk_number
          )

    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(
        x, hparams)


@registry.register_model
class MoeTransformer(Transformer):  # pylint: disable=abstract-method
  """Transformer net based on MoE(Mixture of Experts.).  See file docstring."""

  def __init__(self, *args, **kwargs):
    super(MoeTransformer, self).__init__(*args, **kwargs)  # pylint: disable=super-with-arguments
    self._encoder_function = moe_transformer_encoder
    self._decoder_function = moe_transformer_decoder
