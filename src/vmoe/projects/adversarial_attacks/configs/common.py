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

"""Common config utils for the paper On the Adversarial Robustness of Mixture of Experts."""
import math
import re

import ml_collections
from vmoe.configs.vmoe_paper import common

IMAGENET_BATCH_SIZE = 1024
IMAGENET_NUM_CLASSES = 1000
IMAGENET_IMAGE_SIZE = 384


flatten_dict = common.flatten_dict


def get_num_epochs(description: str) -> int:
  match = re.match(r'.*, ([0-9]+) Epochs$', description)
  if not match:
    raise ValueError(
        f"The number of epochs can't be parsed from {description!r}.")
  return int(match.group(1))


def get_base_finetune_config(
    batch_size: int = IMAGENET_BATCH_SIZE,
    image_size: int = IMAGENET_IMAGE_SIZE,
    num_classes: int = IMAGENET_NUM_CLASSES,
) -> ml_collections.ConfigDict:
  """Base config used for fine-tuning ViT/V-MoE models."""
  config = common.get_base_config()
  # Update options from the base config.
  config.evaluate.every_steps = 100       # Evaluate every 100 steps.

  config.dataset = ml_collections.ConfigDict()
  pp_common = f'value_range(-1,1)|onehot({num_classes}, inkey="label", outkey="labels")|keep("image", "labels")'
  # Dataset variation used for training.
  config.dataset.train = common.get_data_config(
      name='imagenet2012', split='train[:99%]', batch_size=batch_size,
      process=f'decode_jpeg_and_inception_crop({image_size})|flip_lr|{pp_common}',
      shuffle_buffer=50_000, cache=None)
  # Dataset variation used for validation.
  config.dataset.val = common.get_data_config(
      name='imagenet2012', split='train[99%:]', batch_size=batch_size,
      process=f'decode|resize({image_size})|{pp_common}',
      shuffle_buffer=None, cache='batched')
  # Dataset variation used for test.
  config.dataset.test = common.get_data_config(
      name='imagenet2012', split='validation', batch_size=batch_size,
      process=f'decode|resize({image_size})|{pp_common}',
      shuffle_buffer=None, cache='batched')
  # Dataset variation used for test with "real" labels.
  config.dataset.test_real = common.get_data_config(
      name='imagenet2012_real', split='validation', batch_size=batch_size,
      process=(f'decode|resize({image_size})|value_range(-1,1)|'
               f'onehot({num_classes}, inkey="real_label", outkey="labels")|'
               f'keep("image", "labels")'),
      shuffle_buffer=None, cache='batched')
  # Loss used to train the model.
  config.loss = ml_collections.ConfigDict()
  config.loss.name = 'softmax_xent'
  config.train_steps = 10_000
  config.optimizer = ml_collections.ConfigDict({
      'name': 'sgd',
      'momentum': 0.9,
      'accumulator_dtype': 'float32',
      'learning_rate': {
          'schedule': 'warmup_cosine_decay',
          'peak_value': 0.03,
          'end_value': 1e-5,
          'warmup_steps': 500,
      },
      'gradient_clip': {'global_norm': 10.0},
  })
  return config


def get_vit_config(description: str,
                   num_classes: int) -> ml_collections.ConfigDict:
  """Returns the config for the ViT models used in the paper."""
  match = re.match(r'^ViT-(?P<variant>[^/]+)/(?P<patch>[0-9]+)', description)
  if not match:
    raise ValueError(f"Description {description!r} doesn't match the regex.")
  variant = match.group('variant')
  # Note: the B+, B++, and B+++ interpolate the hparams between B and L.
  # B++ has roughly the same performance as V-MoE B/32 (but its more expensive).
  idx = ['S', 'B', 'B+', 'B++', 'B+++', 'L', 'H'].index(variant)
  patch_size = int(match.group('patch'))
  config = ml_collections.ConfigDict()
  # Note: This uses the official VIT implementation from:
  # https://github.com/google-research/vision_transformer
  config.name = 'VisionTransformer'
  config.num_classes = num_classes
  config.patches = ml_collections.ConfigDict({'size': (patch_size, patch_size)})
  config.hidden_size = [512, 768, 832, 896, 960, 1024, 1280][idx]
  config.classifier = 'token'
  config.representation_size = config.hidden_size
  config.head_bias_init = -math.log(num_classes)
  config.transformer = ml_collections.ConfigDict()
  config.transformer.num_layers = [8, 12, 15, 18, 21, 24, 32][idx]
  config.transformer.mlp_dim = [2048, 3072, 3328, 3584, 3840, 4096, 5120][idx]
  config.transformer.num_heads = [8, 12, 13, 14, 15, 16, 16][idx]
  config.transformer.dropout_rate = 0.0
  config.transformer.attention_dropout_rate = 0.0
  return config


def get_vmoe_config(description: str, num_classes: int,
                    image_size: int) -> ml_collections.ConfigDict:
  """Returns the config for the V-MoE models used in the paper."""
  config = common.get_vmoe_config(
      description, image_size=image_size, num_classes=num_classes)
  # The load loss used in the original V-MoE paper can result in NaNs when the
  # total_number of experts is 2. In such cases, we use the auxiliary loss from
  # the GShard paper.
  num_experts = config.encoder.moe.num_experts
  if num_experts > 2:
    config.encoder.moe.router.importance_loss_weight = 0.005
    config.encoder.moe.router.load_loss_weight = 0.005
    config.encoder.moe.router.gshard_loss_weight = 0.0
  else:
    config.encoder.moe.router.importance_loss_weight = 0.0
    config.encoder.moe.router.load_loss_weight = 0.0
    config.encoder.moe.router.gshard_loss_weight = 0.01
  return config
