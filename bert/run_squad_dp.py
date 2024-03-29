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
"""EPL resnet dp example."""
import tensorflow as tf
from run_squad import main, FLAGS # pylint: disable=unused-import
import epl


if __name__ == "__main__":
  config_json = {}
  if FLAGS.gc:
    config_json["gradient_checkpoint.type"] = "auto"
  if FLAGS.amp:
    config_json["amp.level"] = "o1"
    config_json["amp.loss_scale"] = 128
    config_json["amp.debug_log"] = True
  if FLAGS.zero:
    config_json["zero.level"] = "v1"
  epl.init(epl.Config(config_json))
  epl.set_default_strategy(epl.replicate(device_count=1))
  tf.app.run()
