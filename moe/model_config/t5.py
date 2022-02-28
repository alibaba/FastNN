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
"""Hparams registion for models."""


import moe_transformer  # pylint: disable=unused-import
from tensor2tensor.models.transformer import transformer_base_v1
from tensor2tensor.utils import registry


@registry.register_hparams
def t5_large():
  """Hparams for T5-large."""
  hparams = transformer_base_v1()
  hparams.num_hidden_layers = 24
  hparams.num_encoder_layers = 24
  hparams.num_decoder_layers = 24
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.num_heads = 16
  hparams.attention_key_channels = 1024
  hparams.attention_value_channels = 1024
  hparams.ffn_layer = "dense_relu_dense"
  hparams.parameter_attention_key_channels = 0
  hparams.symbol_modality_num_shards = 1

  hparams.attention_dropout = 0.1
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_preprocess_sequence = "n"
  hparams.optimizer_adam_beta2 = 0.997
  hparams.use_fixed_batch_size = False
  hparams.batch_size = 1024
  hparams.max_length = 512
  hparams.clip_grad_norm = 0.0  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_schedule = "constant*linear_warmup*rsqrt_decay"
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.2
  hparams.learning_rate_constant = 0.1
  hparams.learning_rate_warmup_steps = 16000
  hparams.initializer_gain = 1.0
  hparams.relu_dropout = 0.1

  hparams.optimizer = "TrueAdam"

  return hparams


@registry.register_hparams
def t5_small():
  """Hparams for T5-small."""
  hparams = transformer_base_v1()
  hparams.num_hidden_layers = 4
  hparams.num_encoder_layers = 4
  hparams.num_decoder_layers = 4
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.num_heads = 16
  hparams.attention_key_channels = 1024
  hparams.attention_value_channels = 1024
  hparams.ffn_layer = "dense_relu_dense"
  hparams.parameter_attention_key_channels = 0
  hparams.symbol_modality_num_shards = 1

  hparams.attention_dropout = 0.1
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_preprocess_sequence = "n"
  hparams.optimizer_adam_beta2 = 0.997
  hparams.use_fixed_batch_size = False
  hparams.batch_size = 1024
  hparams.max_length = 512
  hparams.clip_grad_norm = 0.0  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_schedule = "constant*linear_warmup*rsqrt_decay"
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.2
  hparams.learning_rate_constant = 0.1
  hparams.learning_rate_warmup_steps = 16000
  hparams.initializer_gain = 1.0
  hparams.relu_dropout = 0.1

  hparams.optimizer = "TrueAdam"

  return hparams


@registry.register_hparams
def moe_t5_small():
  """Hparams for T5-small based on MoE(Mixture of Experts.)."""
  hparams = transformer_base_v1()
  hparams.num_hidden_layers = 4
  hparams.num_encoder_layers = 4
  hparams.num_decoder_layers = 4
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.num_heads = 16
  hparams.attention_key_channels = 1024
  hparams.attention_value_channels = 1024
  hparams.parameter_attention_key_channels = 0
  hparams.symbol_modality_num_shards = 1

  hparams.attention_dropout = 0.1
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_preprocess_sequence = "n"
  hparams.optimizer_adam_beta2 = 0.997
  hparams.use_fixed_batch_size = True
  hparams.batch_size = 2
  hparams.max_length = 512
  hparams.clip_grad_norm = 0.0  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_schedule = "constant*linear_warmup*rsqrt_decay"
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.2
  hparams.learning_rate_constant = 0.1
  hparams.learning_rate_warmup_steps = 16000
  hparams.initializer_gain = 1.0
  hparams.relu_dropout = 0.1

  hparams.optimizer = "TrueAdam"


  # MoE configuration
  hparams.add_hparam("num_experts", 8)
  hparams.add_hparam("local_dispatch", False)
  hparams.add_hparam("num_local_groups", 2)
  hparams.add_hparam("initializer_range", 0.02)
  hparams.add_hparam("moe_gating", "top_2")
  hparams.add_hparam("expert_capacity_dim", 8)
  hparams.add_hparam("min_expert_capacity", 4)
  hparams.add_hparam("second_expert_policy", "all")
  hparams.add_hparam("second_expert_threshold", 0.5)
  hparams.add_hparam("switch_policy_train", "argmax")
  hparams.add_hparam("switch_policy_eval", "argmax")
  hparams.add_hparam("switch_dropout", 0.1)
  hparams.add_hparam("capacity_factor_train", 1.25)
  hparams.add_hparam("capacity_factor_eval", 2.0)
  hparams.add_hparam("loss_coef", 1e-2)
  hparams.add_hparam("activation_fn", "relu")
  hparams.add_hparam("is_training", True)

  return hparams
