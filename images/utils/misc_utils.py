# Copyright (C) 2016 The TensorFlow Authors.
# Copyright (C) 2019 Alibaba Group Holding Limited.
# All Rights Reserved.
# ==============================================================================
from __future__ import division
from __future__ import print_function

import six
import sys
import tensorflow as tf
from flags import FLAGS
from tensorflow import logging


def print_out(s, f=None, new_line=True):
  """Similar to print but with support to flush and output to a file."""
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  if six.PY2:
    sys.stdout.write(s.encode("utf-8"))
  else:
    sys.stdout.buffer.write(s.encode("utf-8"))

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()


def get_assigment_map_from_checkpoint(tvars, init_checkpoint, scope_map=None):
  """Compute the union of the current variables and checkpoint variables."""

  import collections
  import re

  initialized_variable_names = {}

  # current model variables
  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  # ckpt variables
  init_vars = tf.train.list_variables(init_checkpoint)
  # logging.info('init_vars from init_checkpoint')
  # for var in init_vars:
  #    logging.info(var)

  # logging.info('')
  # if scope_map is not None:
  #    logging.info('assignment_map from scope {} to {}'.format(scope_map[0], scope_map[1]))

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name in name_to_variable:
      assignment_map[name] = name
      initialized_variable_names[name] = 1
      initialized_variable_names[name + ":0"] = 1

    if scope_map:
      try:
        idx = name.index(scope_map[0])
        new_name = scope_map[1] + name[idx + len(scope_map[0]):]
        if new_name in name_to_variable:
          assignment_map[name] = new_name
          initialized_variable_names[new_name] = 1
          initialized_variable_names[new_name + ":0"] = 1
      except:
        continue

  return (assignment_map, initialized_variable_names)


def load_checkpoint():
    tvars = tf.trainable_variables()
    import os
    init_checkpoint = os.path.join(FLAGS.model_dir, FLAGS.ckpt_file_name)
    print('start loading checkpoint with path: ', init_checkpoint)
    (assignment_map, initialized_variable_names) = \
        get_assigment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        else:
            # un_init_variables.add(var)
            pass
        logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)


def optimizer_statistics(optimizer):
  size = lambda v: reduce(lambda x, y: x * y, v.get_shape().as_list())
  for v in optimizer.variables():
    logging.info('%s, %s, %s' % (v.name, v.device, size(v)))
  logging.info('Total optimizer size: %s ' % sum(size(v) for v in optimizer.variables()))


def print_model_statistics():
  """Print trainable variables info, including name and size."""
  print_out("# Trainable variables")
  print_out("Format: <name>, <shape>, <(soft) device placement>")
  size = lambda v: reduce(lambda x, y: x * y, v.get_shape().as_list())
  params = tf.trainable_variables()
  for param in params:
    print_out("  %s, %s, %s, %d" % (param.name, str(param.get_shape()),
                                      param.op.device, size(param)))
  print("Total model size:", sum(size(param) for param in params))