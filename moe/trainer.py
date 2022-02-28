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
"""Training entry."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags

import tensorflow.compat.v1 as tf
import epl

from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
try:
  from tensor2tensor import models  # pylint: disable=unused-import
except:  # pylint: disable=bare-except
  pass
from tensor2tensor.utils import flags as t2t_flags  # pylint: disable=unused-import
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import optimize
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import trainer_lib
from model_config import t5_large  # pylint: disable=unused-import


FLAGS = flags.FLAGS

flags.DEFINE_integer("random_seed", None, "Random seed.")
flags.DEFINE_integer("iterations_per_loop", 100,
                     "Number of iterations in a TPU training loop.")
flags.DEFINE_bool("generate_data", False, "Generate data before training?")
flags.DEFINE_string("tmp_dir", "/tmp/t2t_datagen",
                    "Temporary storage directory, used if --generate_data.")
flags.DEFINE_integer("inter_op_parallelism_threads", 0,
                     "Number of inter_op_parallelism_threads to use for CPU. "
                     "See TensorFlow config.proto for details.")
flags.DEFINE_integer("intra_op_parallelism_threads", 0,
                     "Number of intra_op_parallelism_threads to use for CPU. "
                     "See TensorFlow config.proto for details.")

# To maintain compatibility with some internal libs, we guard against these flag
# definitions possibly erroring. Apologies for the ugliness.
try:
  flags.DEFINE_string("master", "", "Address of TensorFlow master.")
  flags.DEFINE_string("output_dir", "", "Base output directory for run.")
  flags.DEFINE_string("schedule", "train_and_eval",
                      "Method of Experiment to run.")
  flags.DEFINE_integer("eval_steps", 100,
                       "Number of steps in evaluation. By default, eval will "
                       "stop after eval_steps or when it runs through the eval "
                       "dataset once in full, whichever comes first, so this "
                       "can be a very large number.")
except:  # pylint: disable=bare-except
  pass


# Note than in open-source TensorFlow, the dash gets converted to an underscore,
# so access is FLAGS.job_dir.
flags.DEFINE_integer("log_step_count_steps", 100,
                     "Number of local steps after which progress is printed "
                     "out")

flags.DEFINE_integer("op_split", 0, "whether to shard moe layer or not.")
flags.DEFINE_bool("enable_fp16", False, "")


def set_hparams_from_args(args):
  """Set hparams overrides from unparsed args list."""
  if not args:
    return

  hp_prefix = "--hp_"
  tf.logging.info("Found unparsed command-line arguments. Checking if any "
                  "start with %s and interpreting those as hparams "
                  "settings.", hp_prefix)

  pairs = []
  i = 0
  while i < len(args):
    arg = args[i]
    if arg.startswith(hp_prefix):
      pairs.append((arg[len(hp_prefix):], args[i+1]))
      i += 2
    else:
      tf.logging.warn("Found unknown flag: %s", arg)
      i += 1

  as_hparams = ",".join(["%s=%s" % (key, val) for key, val in pairs])
  if FLAGS.hparams:
    as_hparams = "," + as_hparams
  FLAGS.hparams += as_hparams


def create_hparams():
  """Create hparams."""
  hparams_path = os.path.join(FLAGS.output_dir, "hparams.json")
  return trainer_lib.create_hparams(FLAGS.hparams_set, FLAGS.hparams,
                                    hparams_path=hparams_path)


def create_run_config():
  """Create a run config.

  Returns:
    a run config
  """
  session_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=FLAGS.log_device_placement,
      inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
      intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads)
  run_config = tf.estimator.RunConfig(save_checkpoints_steps=5,
                                      model_dir=FLAGS.output_dir,
                                      session_config=session_config)
  run_config.use_tpu = False
  return run_config


def generate_data():
  # Generate data if requested.
  data_dir = os.path.expanduser(FLAGS.data_dir)
  tmp_dir = os.path.expanduser(FLAGS.tmp_dir)
  tf.gfile.MakeDirs(data_dir)
  tf.gfile.MakeDirs(tmp_dir)

  problem_name = FLAGS.problem
  tf.logging.info("Generating data for %s" % problem_name)
  registry.problem(problem_name).generate_data(data_dir, tmp_dir)


def is_chief():
  schedules = ["train", "train_and_eval", "continuous_train_and_eval"]
  return FLAGS.worker_id == 0 and FLAGS.schedule in schedules


def save_metadata(hparams):
  """Saves FLAGS and hparams to output_dir."""
  output_dir = os.path.expanduser(FLAGS.output_dir)
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  # Save hparams as hparams.json
  hparams_fname = os.path.join(output_dir, "hparams.json")
  with tf.gfile.Open(hparams_fname, "w") as f:
    f.write(hparams.to_json(indent=0, sort_keys=True))
  tf.logging.info("Write hparams.json to {}".format(output_dir))


def main(argv):
  config = epl.Config({"cluster.colocate_split_and_replicate": True})
  epl.init(config)
  FLAGS.worker_id = epl.Env.get().cluster.worker_index
  FLAGS.worker_gpu = epl.Env.get().cluster.total_gpu_num
  epl.set_default_strategy(epl.replicate(FLAGS.worker_gpu))

  # Create HParams.
  if argv:
    set_hparams_from_args(argv[1:])
  if FLAGS.schedule != "run_std_server":
    hparams = create_hparams()

  if FLAGS.schedule == "train":
    mlperf_log.transformer_print(key=mlperf_log.RUN_START)
  else:
    raise RuntimeError("Support training tasks only for now, you can define tasks in other modes.")
  trainer_lib.set_random_seed(FLAGS.random_seed)

  hparams.add_hparam("data_dir", FLAGS.data_dir)
  hparams.add_hparam("schedule", FLAGS.schedule)
  hparams.add_hparam("train_steps", FLAGS.train_steps)
  hparams.add_hparam("warm_start_from", None)
  trainer_lib.add_problem_hparams(hparams, FLAGS.problem)

  # Dataset generation.
  if FLAGS.generate_data:
    generate_data()

  def model_fn_replicate(features, labels, mode):
    model_fn = t2t_model.T2TModel.make_estimator_model_fn(FLAGS.model, hparams)
    return model_fn(features, labels, mode)

  if is_chief():
    save_metadata(hparams)

  estimator = tf.estimator.Estimator(model_fn=model_fn_replicate, config=create_run_config())
  hooks = []
  hooks.append(tf.train.StepCounterHook(every_n_steps=FLAGS.log_step_count_steps))

  optimize.log_variable_sizes(verbose=True)

  problem = hparams.problem
  train_input_fn = problem.make_estimator_input_fn(tf.estimator.ModeKeys.TRAIN,
                                                   hparams)

  estimator.train(train_input_fn, max_steps=hparams.train_steps, hooks=hooks)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
