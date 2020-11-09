# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


class LoggingTensorHook(tf.train.SessionRunHook):
  """Self-defined Hook for logging."""

  def __init__(self, tensors, samples_per_step=1, every_n_iters=100):
    self._tensors = tensors
    self._samples_per_step = samples_per_step
    self._timer = tf.train.SecondOrStepTimer(every_steps=every_n_iters)

  def begin(self):
    self._timer.reset()
    self._tensors['global_step'] = tf.train.get_or_create_global_step()

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(self._tensors)

  def after_run(self, run_context, run_values):
    _ = run_context
    tensor_values = run_values.results
    stale_global_step = tensor_values['global_step']
    if self._timer.should_trigger_for_step(stale_global_step + 1):
      global_step = run_context.session.run(self._tensors['global_step'])
      if self._timer.should_trigger_for_step(global_step):
        elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
        if elapsed_time is not None:
          secs_per_step = '{0:.4f}'.format(elapsed_time / elapsed_steps)
          samples_per_sec = '{0:.4f}'.format(self._samples_per_step * elapsed_steps / elapsed_time)
          tf.logging.info("INFO:tensorflow:[%s secs/step,\t%s samples/sec]\t%s" % (
            secs_per_step, samples_per_sec,
            ',\t'.join(['%s = %s' % (tag, tensor_values[tag]) for tag in tensor_values])))


def get_train_hooks(params, **kwargs):
  """Factory for getting a list of TensorFlow hooks for training by name.

  Args:
      name_list: a list of strings to name desired hook classes. Allowed:
      StopAtStepHook, ProfilerHook, LoggingTensorHook, which are defined
      as keys in HOOKS
      tensors: dict of tensor names to be logged.
      **kwargs: a dictionary of arguments to the hooks.

    Returns:
      list of instantiated hooks, ready to be used in a classifier.train call.

    Raises:
      ValueError: if an unrecognized name is passed.
    """
  if not FLAGS.hooks:
    return []

  name_list = FLAGS.hooks.split(',')
  train_hooks = []
  for name in name_list:
    hook_name = HOOKS.get(name.strip().lower())
    if hook_name is None:
      raise ValueError('Unrecognized training hook requested: {}'.format(name))
    else:
      res = hook_name(params, **kwargs)
      if res:
        train_hooks.append(res)

  return train_hooks


def get_logging_tensor_hook(params, **kwargs):
  """Function to get LoggingTensorHook.
  Args:
      tensors: dict of tensor names.
      samples_per_step:num of samples that machines process at one step.
      every_n_iter: `int`, print the values of `tensors` once every N local steps taken on the current worker.
      **kwargs: a dictionary of arguments to LoggingTensorHook.

    Returns:
      Returns a LoggingTensorHook with a standard set of tensors that will be
      printed to stdout or Null.
  """
  if FLAGS.log_loss_every_n_iters > 0:
    return LoggingTensorHook(
      tensors=params['tensors_to_log'],
      samples_per_step=params['samples_per_step'],
      every_n_iters=FLAGS.log_loss_every_n_iters)
  else:
    pass


def get_profiler_hook(params, **kwargs):
  """Function to get ProfilerHook.
  Args:
      model_dir: The directory to save the profile traces to.
      save_steps: `int`, print profile traces every N steps.
      **kwargs: a dictionary of arguments to ProfilerHook.

  Returns:
      Returns a ProfilerHook that writes out timelines that can be loaded into
      profiling tools like chrome://tracing or Null.
  """
  if FLAGS.profile_every_n_iters > 0 and FLAGS.task_index == FLAGS.profile_at_task:
    return tf.train.ProfilerHook(
      show_memory=FLAGS.show_memory,
      save_steps=FLAGS.profile_every_n_iters,
      output_dir=FLAGS.output_dir)
  else:
    pass


def get_stop_at_step_hook(params, **kwargs):
  """Function to get StopAtStepHook.

  Args:
      last_step: `int`, training stops at N steps.
      **kwargs: a dictionary of arguments to StopAtStepHook.

  Returns:
      Returns a StopAtStepHook.
  """
  return tf.train.StopAtStepHook(last_step=FLAGS.stop_at_step)


def get_checkpoint_saver_hook(params, **kwargs):
  """Function to get CheckpointSaverHook.

  Args:
      last_step: `int`, training stops at N steps.
      **kwargs: a dictionary of arguments to StopAtStepHook.

  Returns:
      Returns a CheckpointSaverHook.
  """
  if FLAGS.task_index == 0 and FLAGS.checkpointDir is not None \
      and FLAGS.save_checkpoints_steps > 0:
    import os
    import time
    local_time_ticket = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    model_output_path = os.path.join(FLAGS.checkpointDir, local_time_ticket)
    model_saver = tf.train.Saver(max_to_keep=FLAGS.max_save) if FLAGS.max_save is not None else None
    if not os.path.exists(model_output_path):
      tf.logging.info('Creating output path:%s' % (model_output_path))
      os.makedirs(model_output_path)
    return tf.train.CheckpointSaverHook(model_output_path,
                                        save_steps=FLAGS.save_checkpoints_steps,
                                        saver=model_saver)
  else:
    pass


# A dictionary to map one hook name and its corresponding function
HOOKS = {
  'loggingtensorhook': get_logging_tensor_hook,
  'profilerhook': get_profiler_hook,
  'stopatstephook': get_stop_at_step_hook,
  'checkpointsaverhook': get_checkpoint_saver_hook,
}
