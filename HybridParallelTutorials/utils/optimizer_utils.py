# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


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
  elif FLAGS.learning_rate_decay_type == "cosine":
    return tf.train.cosine_decay(
        learning_rate,
        global_step=global_step - FLAGS.warmup_steps,
        decay_steps=FLAGS.stop_at_step - FLAGS.warmup_steps,
        alpha=FLAGS.min_lr_ratio)
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

  def _prepare(self):
    self.learning_rate_t = tf.convert_to_tensor(
      self.learning_rate, name='learning_rate')
    self.weight_decay_rate_t = tf.convert_to_tensor(
      self.weight_decay_rate, name='weight_decay_rate')
    self.beta_1_t = tf.convert_to_tensor(self.beta_1, name='beta_1')
    self.beta_2_t = tf.convert_to_tensor(self.beta_2, name='beta_2')
    self.epsilon_t = tf.convert_to_tensor(self.epsilon, name='epsilon')

  def _create_slots(self, var_list):
    for v in var_list:
      self._zeros_slot(v, 'm', self._name)
      self._zeros_slot(v, 'v', self._name)

  def _apply_dense(self, grad, var):
    with tf.control_dependencies([grad]):
      return self._apply_dense_impl(grad, var)

  def _apply_dense_impl(self, grad, var):
    learning_rate_t = tf.cast(
      self.learning_rate_t, var.dtype.base_dtype)
    beta_1_t = tf.cast(self.beta_1_t, var.dtype.base_dtype)
    beta_2_t = tf.cast(self.beta_2_t, var.dtype.base_dtype)
    epsilon_t = tf.cast(self.epsilon_t, var.dtype.base_dtype)
    weight_decay_rate_t = tf.cast(
      self.weight_decay_rate_t, var.dtype.base_dtype)

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    # Standard Adam update.
    next_m = (
      tf.multiply(beta_1_t, m) +
      tf.multiply(1.0 - beta_1_t, grad))
    next_v = (
      tf.multiply(beta_2_t, v) + tf.multiply(1.0 - beta_2_t,
                                             tf.square(grad)))

    update = next_m / (tf.sqrt(next_v) + epsilon_t)

    if self._do_use_weight_decay(self._get_variable_name(var.name)):
      update += weight_decay_rate_t * var

    update_with_lr = learning_rate_t * update

    next_param = var - update_with_lr

    return tf.group(*[var.assign(next_param),
                      m.assign(next_m),
                      v.assign(next_v)])

  def _resource_apply_dense(self, grad, var):
    with tf.control_dependencies([grad]):
      return self._resource_apply_dense_impl(grad, var)

  def _resource_apply_dense_impl(self, grad, var):
    learning_rate_t = tf.cast(
      self.learning_rate_t, var.dtype.base_dtype)
    beta_1_t = tf.cast(self.beta_1_t, var.dtype.base_dtype)
    beta_2_t = tf.cast(self.beta_2_t, var.dtype.base_dtype)
    epsilon_t = tf.cast(self.epsilon_t, var.dtype.base_dtype)
    weight_decay_rate_t = tf.cast(
      self.weight_decay_rate_t, var.dtype.base_dtype)

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    # Standard Adam update.
    next_m = (
      tf.multiply(beta_1_t, m) +
      tf.multiply(1.0 - beta_1_t, grad))
    next_v = (
      tf.multiply(beta_2_t, v) + tf.multiply(1.0 - beta_2_t,
                                             tf.square(grad)))

    update = next_m / (tf.sqrt(next_v) + epsilon_t)

    if self._do_use_weight_decay(var.name):
      update += weight_decay_rate_t * var

    update_with_lr = learning_rate_t * update

    next_param = var - update_with_lr

    return tf.group(*[var.assign(next_param),
                      m.assign(next_m),
                      v.assign(next_v)])

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    with tf.control_dependencies([grad]):
      return self._apply_sparse_shared_impl(grad, var, indices, scatter_add)

  def _apply_sparse_shared_impl(self, grad, var, indices, scatter_add):
    learning_rate_t = tf.cast(
      self.learning_rate_t, var.dtype.base_dtype)
    beta_1_t = tf.cast(self.beta_1_t, var.dtype.base_dtype)
    beta_2_t = tf.cast(self.beta_2_t, var.dtype.base_dtype)
    epsilon_t = tf.cast(self.epsilon_t, var.dtype.base_dtype)
    weight_decay_rate_t = tf.cast(
      self.weight_decay_rate_t, var.dtype.base_dtype)

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    m_t = tf.assign(m, m * beta_1_t,
                    use_locking=self._use_locking)

    m_scaled_g_values = grad * (1 - beta_1_t)
    with tf.control_dependencies([m_t]):
      m_t = scatter_add(m, indices, m_scaled_g_values)

    v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
    v_t = tf.assign(v, v * beta_2_t, use_locking=self._use_locking)
    with tf.control_dependencies([v_t]):
      v_t = scatter_add(v, indices, v_scaled_g_values)

    update = m_t / (tf.math.sqrt(v_t) + epsilon_t)

    if self._do_use_weight_decay(var.name):
      update += weight_decay_rate_t * var

    update_with_lr = learning_rate_t * update

    var_update = tf.assign_sub(var,
                               update_with_lr,
                               use_locking=self._use_locking)
    return tf.group(*[var_update, m_t, v_t])

  def _apply_sparse(self, grad, var):
    return self._apply_sparse_shared(
      grad.values, var, grad.indices,
      lambda x, i, v: tf.scatter_add(  # pylint: disable=g-long-lambda
        x, i, v, use_locking=self._use_locking))

  def _resource_scatter_add(self, x, i, v):
    with tf.control_dependencies([tf.scatter_add(x, i, v)]):
      return x.value()

  def _resource_apply_sparse(self, grad, var, indices):
    return self._apply_sparse_shared(
      grad, var, indices, self._resource_scatter_add)

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
