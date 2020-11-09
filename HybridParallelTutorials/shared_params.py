# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
import tensorflow as tf

# Machine parameter
tf.app.flags.DEFINE_integer('task_index', 0, 'task_index')
tf.app.flags.DEFINE_string('taskId', None, '')
tf.app.flags.DEFINE_string('worker_hosts', None, 'worker_hosts')
tf.app.flags.DEFINE_string('ps_hosts', None, '')
tf.app.flags.DEFINE_integer('worker_count', None, '')
tf.app.flags.DEFINE_integer('ps_count', None, '')
tf.app.flags.DEFINE_string('job_name', None, 'job_name')


# Model base parameters
tf.app.flags.DEFINE_string('runner_name', 'images', 'images or bert or nmt or xlnet or opennmt')
tf.app.flags.DEFINE_string('task_type', 'pretrain', 'pretrain or finetune, default pretrain.')
tf.app.flags.DEFINE_bool('do_predict', False, 'whether do prediction or not.')
tf.app.flags.DEFINE_string("model_dir", "", "The path corresponding to the pre-trained model.")
tf.app.flags.DEFINE_string("ckpt_file_name", None, "Initial checkpoint (pre-trained model: model_dir + model.ckpt).")
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size for training.')
tf.app.flags.DEFINE_integer("eval_batch_size", 32, "batch size for prediction")


# dataset
tf.app.flags.DEFINE_string('dataset_name', 'mock', 'default mock data, or flowers/mnist/cifar10/imagenet.')
tf.app.flags.DEFINE_string('eval_dataset_name', None, 'default None.')
tf.app.flags.DEFINE_string('dataset_dir', None, '')
tf.app.flags.DEFINE_string('eval_dataset_dir', None, '')
tf.app.flags.DEFINE_integer('num_epochs', 100, '')
tf.app.flags.DEFINE_string('file_pattern', '*', 'file pattern for dataset, eg. cifar10_%s.tfrecord for cifar10')
tf.app.flags.DEFINE_string('eval_file_pattern', '*', 'file pattern for dataset, eg. cifar10_%s.tfrecord for cifar10')
tf.app.flags.DEFINE_string('reader', None, '')
tf.app.flags.DEFINE_integer('num_sample_per_epoch', 1000000, '')
tf.app.flags.DEFINE_string('train_files', None, '')
tf.app.flags.DEFINE_string('eval_files', None, '')


# preprocessing
tf.app.flags.DEFINE_integer('num_preprocessing_threads', 16, 'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer('shuffle_buffer_size', 1024, '')
tf.app.flags.DEFINE_integer('prefetch_buffer_size', 32, '')
tf.app.flags.DEFINE_integer('num_parallel_batches', 8, '')
tf.app.flags.DEFINE_bool('datasets_use_caching', False,
                         'Cache the compressed input data in memory. This improves '
                         'the data input performance, at the cost of additional '
                         'memory.')


# fp16 parameters, if use_fp16=False, no other fp16 parameters apply.
tf.app.flags.DEFINE_bool('use_fp16', False,
                         'Use 16-bit floats for certain tensors instead of 32-bit floats')
tf.app.flags.DEFINE_float('loss_scale', 1.0,
                          'If fp16 is enabled, the loss is multiplied by this amount '
                          'right before gradients are computed, then each gradient '
                          'is divided by this amount. Mathematically, this has no '
                          'effect, but it helps avoid fp16 underflow. Set to 1 to '
                          'effectively disable. Ignored during eval.')


# Learning rate tuning parameters
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential',
                           'Specifies how the learning rate is decayed. One of "fixed", "exponential", or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float('end_learning_rate', 0.0001,
                          'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float("min_lr_ratio", 0.0, "min lr ratio for cos decay.")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float("lr_layer_decay_rate", 0.75, "Top layer: lr[L] = FLAGS.learning_rate."
                                                       "Lower layers: lr[l-1] = lr[l] * lr_layer_decay_rate.")
tf.app.flags.DEFINE_integer('warmup_steps', 0, 'how many steps we inverse-decay learning.')
tf.app.flags.DEFINE_string('warmup_scheme', 't2t',
                           'how to warmup learning rates. Options include:'
                           't2t: Tensor2Tensor way, start with lr 100 times smaller,'
                           'then exponentiate until the specified lr.')
tf.app.flags.DEFINE_string('decay_scheme', '',
                           'How we decay learning rate. Options include:'
                           'luong234: after 2/3 num train steps, we start halving the learning rate'
                           'for 4 times before finishing.'
                           'luong5: after 1/2 num train steps, we start halving the learning rate'
                           'for 5 times before finishing.'
                           'luong10: after 1/2 num train steps, we start halving the learning rate'
                           'for 10 times before finishing.')


# Optimizer parameters
tf.app.flags.DEFINE_string('optimizer', 'rmsprop',
                           'The name of the optimizer, one of "adadelta", "adagrad", "adam", "ftrl", "momentum", "sgd" or "rmsprop".')

# Optimizer parameters specifc to adadelta
tf.app.flags.DEFINE_float('adadelta_rho', 0.95, 'The decay rate for adadelta.')

# Optimizer parameters specifc to adagrad
tf.app.flags.DEFINE_float('adagrad_initial_accumulator_value', 0.1, 'Starting value for the AdaGrad accumulators.')

# Optimizer parameters specifc to adam
tf.app.flags.DEFINE_float('adam_beta1', 0.9, 'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, 'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

# Optimizer parameters specifc to ftrl
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5, 'The learning rate power.')
tf.app.flags.DEFINE_float('ftrl_initial_accumulator_value', 0.1, 'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float('ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float('ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

# Optimizer parameters specifc to momentum
tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

# Optimizer parameters specifc to rmsprop_momentum
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')


# Logging parameters
tf.app.flags.DEFINE_integer('stop_at_step', 100, 'stop at stop')
tf.app.flags.DEFINE_integer('eval_every_n_iters', 100, '')
tf.app.flags.DEFINE_integer('log_loss_every_n_iters', 10, 'log_loss_every_n_iters')
tf.app.flags.DEFINE_integer('profile_every_n_iters', 0, 'profile_every_n_iters')
tf.app.flags.DEFINE_integer('profile_at_task', 0, 'profile_at_task')
tf.app.flags.DEFINE_bool('log_device_placement', False, 'Whether or not to log device placement.')
tf.app.flags.DEFINE_bool('log_trainable_vars_statistics', False, '')
tf.app.flags.DEFINE_bool('log_optimizer_vars_statistics', False, '')
tf.app.flags.DEFINE_string('hooks', 'StopAtStepHook,ProfilerHook,LoggingTensorHook',
                           'specify hooks for training.')


# Performance tuning parameters
tf.app.flags.DEFINE_string('protocol', 'grpc', 'default grpc. if rdma cluster, use grpc+verbs instead.')
tf.app.flags.DEFINE_integer('inter_op_parallelism_threads', 256, 'Compute pool size')
tf.app.flags.DEFINE_integer('intra_op_parallelism_threads', 96, 'Eigen pool size')
tf.app.flags.DEFINE_bool('colocate_gradients_with_ops', True, 'whether try colocating gradients with corresponding op')
tf.app.flags.DEFINE_float('max_gradient_norm', None, 'clip gradients to this norm.')


# Input/Output parameter
tf.app.flags.DEFINE_string('tables', None, '')
tf.app.flags.DEFINE_string('buckets', None, '')
tf.app.flags.DEFINE_string('volumes', None, '')
tf.app.flags.DEFINE_string('output_dir', '.', '')
tf.app.flags.DEFINE_string('checkpointDir', None, 'The output directory where the model checkpoints will be written.')
tf.app.flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")
tf.app.flags.DEFINE_integer("max_save", 5, "Max number of checkpoints to save. Use 0 to save all.")
tf.app.flags.DEFINE_string('summaryDir', None, '')
tf.app.flags.DEFINE_string('outputs', None, '')


# Whale Parameters
tf.app.flags.DEFINE_bool("enable_whale", True, "")
tf.app.flags.DEFINE_integer('num_gpus_for_pq', 1, '')
tf.app.flags.DEFINE_integer('cut_layer', 24, '')
tf.app.flags.DEFINE_integer('cut_layer1', 8, '')
tf.app.flags.DEFINE_integer('cut_layer2', 8, '')
tf.app.flags.DEFINE_integer('num_micro_batch', 3, '')
tf.app.flags.DEFINE_integer('average', 1, '')
tf.app.flags.DEFINE_bool('show_memory', False, '')


SHARED_FLAGS = tf.app.flags.FLAGS