# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
import tensorflow as tf

# Model base parameters
tf.app.flags.DEFINE_string("model_type", "mrc",
                           "Model types could be one of [classification, regression, pretrain, mrc], "
                           "default is pretrain.")
tf.app.flags.DEFINE_integer("max_seq_length", 384,
                            "The maximum total input sequence length after WordPiece tokenization. "
                            "Sequences longer than this will be truncated, and sequences shorter "
                            "than this will be padded.")
tf.app.flags.DEFINE_string("model_config_file_name", None,
                           "The config json file corresponding to the pre-trained model. "
                           "This specifies the model architecture.")


EXCLUSIVE_FLAGS = tf.app.flags.FLAGS

