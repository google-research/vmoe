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

"""Most common configuration parameters, used in many of the V-MoE paper experiments.

TL;DR: IF YOU ARE THINKING ABOUT CHANGING THIS FILE: DON'T.

This file is to help simplify the config files related to the V-MoE paper.
You might think: "Oh! I could change this line and then re-use that function
from common.py in my own config. That would make my life so easy!".
Unfortunately, that might have as well unexpected consequences
(i.e. non-reproducible results). So, unless you are able to prove that your
changes have no effect in the paper experiments, we will not accept them.
"""
import math
import re
from typing import Optional

import ml_collections
from vmoe.configs import common_fewshot

DESCRIPTIONS_REGEX = re.compile(
    r'^ViT-(?P<variant>.)/(?P<patch>[0-9]+), '
    r'E=(?P<num_experts>[0-9]+), '
    r'K=(?P<k>[0-9]+), '
    r'(?P<where>Every|Last) (?P<where_num>[0-9]+), '
    r'(?P<epochs>[0-9]+) Epochs$')


get_fewshot = common_fewshot.get_fewshot


def flatten_dict(config, prefix=''):
  if isinstance(config, ml_collections.ConfigDict):
    config = config.to_dict()
  flat_dict = {}
  for k, v in config.items():
    if isinstance(v, dict):
      flat_dict.update(flatten_dict(v, prefix=f'{prefix}{k}.'))
    else:
      flat_dict[f'{prefix}{k}'] = v
  return flat_dict


def get_base_config() -> ml_collections.ConfigDict:
  """Returns the base config with options for saving checkpoints, profiling, etc."""
  config = ml_collections.ConfigDict()
  # Write checkpoints every 1000 steps.
  config.save_checkpoint = ml_collections.ConfigDict()
  config.save_checkpoint.every_steps = 1_000
  config.save_checkpoint.keep_last = 1
  config.save_checkpoint.wait_seconds = 300
  # Report training progress every minute.
  config.report_progress = ml_collections.ConfigDict()
  config.report_progress.every_secs = None
  config.report_progress.every_steps = 100
  # Evaluate on the validation set every 1000 steps.
  config.evaluate = ml_collections.ConfigDict()
  config.evaluate.every_steps = 1_000
  # Run device profiling on process_index = 0, for 5 steps, starting at step 10.
  # Then repeat profiling every hour.
  config.profile = ml_collections.ConfigDict()
  config.profile.all_processes = False
  config.profile.num_profile_steps = 5
  config.profile.first_profile = 10
  config.profile.every_secs = 3600.0
  # Seed for generating random numbers.
  config.seed = 0
  return config


def get_data_config(
    name: str,
    split: str,
    process: str,
    batch_size: int,
    shuffle_buffer: Optional[int] = None,
    cache: Optional[str] = None,
    data_dir: Optional[str] = None) -> ml_collections.ConfigDict:
  """Returns dataset parameters."""
  config = ml_collections.ConfigDict(type_safe=False)
  config.name = name
  config.split = split
  config.process = process
  config.batch_size = batch_size
  config.prefetch = 'autotune'
  config.prefetch_device = 2
  if shuffle_buffer:
    config.shuffle_buffer = shuffle_buffer
  if cache:
    config.cache = cache
  if data_dir:
    config.data_dir = data_dir
  return config


def get_mixup_concentration(aug: str) -> float:
  return {
      'light0': 0.0,
      'light1': 0.2,
      'medium1': 0.2,
      'medium2': 0.5,
      'strong1': 0.5,
      'strong2': 0.8,
      'extreme1': 0.5,
      'extreme2': 0.8,
  }[aug]


def get_mixup_config(aug: str) -> ml_collections.ConfigDict():
  config = ml_collections.ConfigDict()
  config.granularity = 'device'
  config.size = 2
  config.concentration = get_mixup_concentration(aug)
  return config


def get_num_epochs(description: str) -> int:
  match = re.match(DESCRIPTIONS_REGEX, description)
  if not match:
    raise ValueError(f"Description {description!r} doesn't match the regex.")
  return int(match.group('epochs'))


def get_optimizer_config(description: str) -> ml_collections.ConfigDict:
  """Returns optimizer parameters for different canonical architectures."""
  match = re.match(DESCRIPTIONS_REGEX, description)
  if not match:
    raise ValueError(f"Description {description!r} doesn't match the regex.")

  variant = match.group('variant')
  patch_size = int(match.group('patch'))

  config = ml_collections.ConfigDict(type_safe=False)
  config.name = 'adam'
  config.b1 = 0.9
  config.b2 = 0.999
  config.mu_dtype = 'float32'  # Optionally, use bfloat16 to save memory.
  config.weight_decay = 0.1

  # Parameters of the learning rate schedule.
  config.learning_rate = ml_collections.ConfigDict()
  config.learning_rate.schedule = 'warmup_linear_decay'
  config.learning_rate.peak_value = {
      ('S', 32): 1e-3,
      ('B', 32): 8e-4,
      ('B', 16): 8e-4,
      ('L', 32): 6e-4,
      ('L', 16): 4e-4,
      ('H', 14): 3e-4,
  }[(variant, patch_size)]
  config.learning_rate.end_value = 1e-5
  config.learning_rate.warmup_steps = 10_000
  # Gradient clipping is only used for VMoE-H/* models.
  config.gradient_clip = ml_collections.ConfigDict({
      'global_norm': 10.0 if variant == 'H' else None
  })
  return config


def get_randaug(aug: str) -> str:
  """Returns a string representing the RandAugment op to use during preprocessing."""
  l, m = {
      'light0': (2, 0),
      'light1': (2, 10),
      'medium1': (2, 15),
      'medium2': (2, 15),
      'strong1': (2, 20),
      'strong2': (2, 20),
      'extreme1': (4, 15),
      'extreme2': (4, 20),
  }[aug]
  return f'randaug({l}, {m})'


def get_vmoe_config(description: str, image_size: int,
                    num_classes: int) -> ml_collections.ConfigDict:
  """Returns transformer parameters for different canonical architectures."""
  match = re.match(DESCRIPTIONS_REGEX, description)
  if not match:
    raise ValueError(f"Description {description!r} doesn't match the regex.")

  variant = match.group('variant')
  variant_idx = ['S', 'B', 'L', 'H'].index(variant)
  patch_size = int(match.group('patch'))
  num_total_experts = int(match.group('num_experts'))
  num_selected_experts = int(match.group('k'))
  moe_where = match.group('where')
  moe_where_num = int(match.group('where_num'))
  # For efficiency reasons, the tokens are divided in several groups of the
  # following size. The routing is performed independently on each group.
  # Group size must be a divisor of the number of tokens per device.
  # We assume here that the smallest batch size per device (images/device) is 8,
  # and any other batch size per device will be a multiple of this.
  min_batch_size_per_device = 8
  num_patches = (image_size // patch_size) * (image_size // patch_size) + 1
  group_size = min_batch_size_per_device * num_patches

  config = ml_collections.ConfigDict()
  config.name = 'VisionTransformerMoe'
  config.num_classes = num_classes
  config.patch_size = (patch_size, patch_size)
  config.hidden_size = [512, 768, 1024, 1280][variant_idx]
  config.classifier = 'token'
  config.representation_size = config.hidden_size
  config.head_bias_init = -math.log(num_classes)
  config.encoder = ml_collections.ConfigDict()
  config.encoder.num_layers = [8, 12, 24, 32][variant_idx]
  config.encoder.mlp_dim = [2048, 3072, 4096, 5120][variant_idx]
  config.encoder.num_heads = [8, 12, 16, 16][variant_idx]
  config.encoder.dropout_rate = 0.0
  config.encoder.attention_dropout_rate = 0.0
  config.encoder.moe = ml_collections.ConfigDict()
  config.encoder.moe.num_experts = num_total_experts
  # Position of MoE layers.
  if moe_where == 'Every':
    config.encoder.moe.layers = tuple(
        range(moe_where_num - 1, config.encoder.num_layers, moe_where_num))
  elif moe_where == 'Last':
    config.encoder.moe.layers = tuple(
        range(1, config.encoder.num_layers, 2))[-moe_where_num:]
  else:
    raise ValueError(
        f'Unknown position for expert layers: {moe_where} {moe_where_num}')
  config.encoder.moe.dropout_rate = 0.0
  config.encoder.moe.split_rngs = False  # All experts share initialization.
  config.encoder.moe.group_size = group_size
  config.encoder.moe.router = ml_collections.ConfigDict()
  config.encoder.moe.router.num_selected_experts = num_selected_experts
  config.encoder.moe.router.noise_std = 1.0  # Actually, it's 1.0 / num_experts.
  config.encoder.moe.router.importance_loss_weight = 0.005
  config.encoder.moe.router.load_loss_weight = 0.005
  # We support both 'einsum' and 'indices' dispatcher. However, 'indices' is
  # currently not well supported by pjit.
  config.encoder.moe.router.dispatcher = ml_collections.ConfigDict()
  config.encoder.moe.router.dispatcher.name = 'einsum'
  config.encoder.moe.router.dispatcher.bfloat16 = True
  config.encoder.moe.router.dispatcher.capacity_factor = 1.05
  # This is used to hint pjit about how data is distributed at the input/output
  # of each MoE layer.
  config.encoder.moe.router.dispatcher.partition_spec = (('expert', 'replica'),)
  # By default we don't use batch priority for training the model.
  config.encoder.moe.router.dispatcher.batch_priority = False

  return config
