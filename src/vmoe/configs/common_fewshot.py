# Copyright 2024 Google LLC.
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

"""Most common few-shot eval configuration."""
from typing import Optional
import ml_collections


def get_fewshot(
    *,
    batch_size: int,
    resize_resolution: int = 256,
    target_resolution: int = 224,
    every_steps: Optional[int] = 25_000,
    seeds_per_step: int = 1,
) -> ml_collections.ConfigDict:
  """Returns the standard configuration for few-shot evaluation."""
  config = ml_collections.ConfigDict()
  # Datasets to evaluate.
  config.datasets = {
      'birds': ('caltech_birds2011', 'train', 'test'),
      'caltech': ('caltech101', 'train', 'test'),
      'cars': ('cars196:2.1.0', 'train', 'test'),
      'cifar100': ('cifar100', 'train', 'test'),
      'col_hist': ('colorectal_histology', 'train[:2000]', 'train[2000:]'),
      'dtd': ('dtd', 'train', 'test'),
      'imagenet': ('imagenet2012_subset/10pct', 'train', 'validation'),
      'pets': ('oxford_iiit_pet', 'train', 'test'),
      'uc_merced': ('uc_merced', 'train[:1000]', 'train[1000:]'),
  }
  # Dataset processing options.
  config.batch_size = batch_size
  config.process = (f'decode|resize({resize_resolution})|'
                    f'central_crop({target_resolution})|value_range(-1, 1)|'
                    'keep("image", "label")')
  config.cache = 'loaded'
  # FewShotEvaluator options.
  config.shots = [1, 5, 10, 25]
  config.l2_regs = [2.**i for i in range(-10, 20)]
  # The accuracy of this task is shown as '0/fewshot/imagenet/10shot'.
  config.main_task = ('imagenet', 10)
  # Override num_classes argument passed to the model constructor, so that
  # few-shot representations are the pre-logit activations.
  config.model_overrides = ml_collections.ConfigDict()
  config.model_overrides.num_classes = None
  # FewShotPeriodicAction options.
  if every_steps:
    config.every_steps = every_steps
  if seeds_per_step:
    config.seeds_per_step = seeds_per_step
  return config
