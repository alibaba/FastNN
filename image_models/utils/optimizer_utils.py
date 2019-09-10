# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from flags import FLAGS


def _get_learning_rate_warmup(global_step):
  """Get learning rate warmup."""
  warmup_steps = FLAGS.warmup_steps
  warmup_scheme = FLAGS.warmup_scheme
  tf.logging.info("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
                  (FLAGS.learning_rate, warmup_steps, warmup_scheme))

  # Apply inverse decay if global steps less than warmup steps.
  # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
  # When step < warmup_steps,
  #   learing_rate *= warmup_factor ** (warmup_steps - step)
  if warmup_scheme != "t2t":
    return FLAGS.learning_rate
  else:
    # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
    warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
    inv_decay = warmup_factor ** (
      tf.to_float(warmup_steps - global_step))
    return tf.cond(
      global_step < warmup_steps,
      lambda: inv_decay * FLAGS.learning_rate,
      lambda: FLAGS.learning_rate,
      name="learning_rate_warmup_cond")


def _get_decay_info():
  """Return decay info based on decay_scheme."""
  decay_factor = FLAGS.learning_rate_decay_factor
  if FLAGS.decay_scheme in ["luong5", "luong10", "luong234"]:
    if FLAGS.decay_scheme == "luong5":
      start_decay_step = int(FLAGS.stop_at_step / 2)
      decay_times = 5
    elif FLAGS.decay_scheme == "luong10":
      start_decay_step = int(FLAGS.stop_at_step / 2)
      decay_times = 10
    elif FLAGS.decay_scheme == "luong234":
      start_decay_step = int(FLAGS.stop_at_step * 2 / 3)
      decay_times = 4
    remain_steps = FLAGS.stop_at_step - start_decay_step
    decay_steps = int(remain_steps / decay_times)
  elif not FLAGS.decay_scheme:  # no decay
    start_decay_step = FLAGS.stop_at_step
    decay_steps = 0
  elif FLAGS.decay_scheme:
    raise ValueError("Unknown decay scheme %s" % FLAGS.decay_scheme)
  return start_decay_step, decay_steps, decay_factor


def configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

  Returns:
      A `Tensor` representing the learning rate.

  Raises:
      ValueError: if FLAGS.learning_rate_decay_type was not recognized
  """
  start_decay_step, decay_steps, decay_factor = _get_decay_info()
  tf.logging.info("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                  "decay_factor %g" % (FLAGS.decay_scheme,
                                       start_decay_step,
                                       decay_steps,
                                       decay_factor))
  learning_rate = _get_learning_rate_warmup(global_step)

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.cond(
      global_step < start_decay_step,
      lambda: learning_rate,
      lambda: tf.train.exponential_decay(
        learning_rate,
        (global_step - start_decay_step),
        decay_steps, decay_factor, staircase=True),
      name="exponential_learning_rate_decay_cond")
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.cond(
      global_step < start_decay_step,
      lambda: learning_rate,
      lambda: tf.train.polynomial_decay(
        learning_rate,
        (global_step - start_decay_step),
        decay_steps,
        FLAGS.end_learning_rate,
        power=1.0,
        cycle=False),
      name="polynomial_learning_rate_decay_cond")

  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                     FLAGS.learning_rate_decay_type)


def configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
      learning_rate: A scalar or `Tensor` learning rate.

  Returns:
      An instance of an optimizer.

  Raises:
      ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
      learning_rate,
      rho=FLAGS.adadelta_rho,
      epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
      learning_rate,
      initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
      learning_rate,
      beta1=FLAGS.adam_beta1,
      beta2=FLAGS.adam_beta2,
      epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
      learning_rate,
      learning_rate_power=FLAGS.ftrl_learning_rate_power,
      initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
      l1_regularization_strength=FLAGS.ftrl_l1,
      l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
      learning_rate,
      momentum=FLAGS.momentum,
      name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
      learning_rate,
      decay=FLAGS.rmsprop_decay,
      momentum=FLAGS.rmsprop_momentum,
      epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif FLAGS.optimizer == "adamweightdecay":
    optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=FLAGS.adam_beta1,
      beta_2=FLAGS.adam_beta2,
      epsilon=FLAGS.opt_epsilon,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  else:
    raise ValueError('Optimizer [%s] was not recognized' % FLAGS.optimizer)
  return optimizer


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.m_and_v = []

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
     if grad is None or param is None:
       continue

     with tf.control_dependencies([grad]):

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
        name=param_name + "/adam_m",
        shape=param.shape.as_list(),
        dtype=tf.float32,
        trainable=False,
        initializer=tf.zeros_initializer())
      v = tf.get_variable(
        name=param_name + "/adam_v",
        shape=param.shape.as_list(),
        dtype=tf.float32,
        trainable=False,
        initializer=tf.zeros_initializer())
      self.m_and_v.append(m)
      self.m_and_v.append(v)
      tf.add_to_collection('OPT_VARS', m)
      tf.add_to_collection('OPT_VARS', v)

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, tf.cast(grad, tf.float32)))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(tf.cast(grad, tf.float32))))

      update = tf.cast(next_m / (tf.sqrt(next_v) + self.epsilon), grad.dtype)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = tf.cast(self.learning_rate, grad.dtype) * update

      next_param = param - update_with_lr

      assignments.extend(
        [param.assign(next_param),
         m.assign(next_m),
         v.assign(next_v)])

    if global_step is not None:
      increase_step = tf.assign_add(global_step, 1)
      assignments.append(increase_step)

    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
