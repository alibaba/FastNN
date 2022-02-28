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
"""ResNet split+dp example."""
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import epl


tf.app.flags.DEFINE_integer("class_num", 10000, "")
FLAGS = tf.app.flags.FLAGS

batch_size = 32
class_num = FLAGS.class_num


def get_mock_iterator():
  """Get mock dataset."""
  dataset = tf.data.Dataset.from_tensor_slices((
      tf.random.uniform([batch_size * 2, 224, 224, 3],
                        minval=0.0, maxval=255.0, dtype=tf.float32),
      tf.random.uniform([batch_size * 2, ],
                        minval=0, maxval=class_num, dtype=tf.int64)))
  dataset = dataset.repeat(200)
  dataset = dataset.batch(batch_size).prefetch(buffer_size=batch_size * 2)
  mock_iterate_op = dataset.make_initializable_iterator()
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                       mock_iterate_op.initializer)
  return mock_iterate_op


def run_model(total_gpu_num):
  """Train model."""
  with epl.replicate(total_gpu_num):
    iterator = get_mock_iterator()
    images, labels = iterator.get_next()
    features = resnet_v1.resnet_v1_50(images, num_classes=None, is_training=True)[0]
    features = tf.squeeze(features, [1, 2])

  with epl.split(total_gpu_num):
    logits = tf.layers.dense(features, class_num)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  global_step = tf.train.get_or_create_global_step()
  optimizer = tf.train.AdamOptimizer(learning_rate=0.9)
  train_op = optimizer.minimize(loss, global_step=global_step)

  hooks = [tf.train.StopAtStepHook(last_step=20)]
  with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
    while not sess.should_stop():
      starttime = time.time()
      _, _, step = sess.run([loss, train_op, global_step])
      endtime = time.time()
      tf.logging.info("[Iteration {} ], Time: {:.4} .".format(step, endtime-starttime))
  tf.logging.info("[Finished]")


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  config = epl.Config({"cluster.colocate_split_and_replicate": True})
  epl.init(config)
  _total_gpu_num = epl.Env.get().cluster.total_gpu_num
  run_model(_total_gpu_num)
