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

"""Common configurations used in the Soft router experiments."""
import math
from typing import Optional

import ml_collections
from ml_collections import config_dict
from vmoe.configs import common_fewshot

get_fewshot_config = common_fewshot.get_fewshot


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
  # Report training progress every minute to avoid hitting maximum RPC/s quota.
  config.report_progress = ml_collections.ConfigDict()
  config.report_progress.every_secs = 60.0
  config.report_progress.every_steps = 250
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
    data_dir: Optional[str] = None,
) -> ml_collections.ConfigDict:
  """Returns dataset parameters."""
  config = ml_collections.ConfigDict(type_safe=False)
  config.name = name
  config.split = split
  config.process = process
  config.batch_size = batch_size
  config.prefetch = 'autotune'
  config.prefetch_device = 2
  if shuffle_buffer:
    config.shuffle_buffer = shuffle_buffer or config_dict.placeholder(int)
  if cache:
    config.cache = cache or config_dict.placeholder(str)
  if data_dir:
    config.data_dir = data_dir or config_dict.placeholder(str)
  return config


def get_adam_config() -> ml_collections.ConfigDict:
  config = ml_collections.ConfigDict(type_safe=False)
  config.name = 'adam'
  config.b1 = 0.9
  config.b2 = 0.999
  config.mu_dtype = 'float32'  # Optionally, use bfloat16 to save memory.
  config.weight_decay = (
      ('head/kernel', 3.0),
      ('.*/kernel', 0.03),
  )
  config.gradient_clip = ml_collections.ConfigDict({'global_norm': 1.0})
  return config


def get_optimizer_linear_config() -> ml_collections.ConfigDict:
  """Returns optimizer parameters as in the "Scaling Vision Transformers" paper with linear LR decay."""
  config = get_adam_config()
  # Parameters of the learning rate schedule.
  config.learning_rate = ml_collections.ConfigDict()
  config.learning_rate.schedule = 'warmup_linear_decay'
  config.learning_rate.peak_value = 8e-4
  config.learning_rate.end_value = 0.
  config.learning_rate.warmup_steps = 10_000
  return config


def get_optimizer_rsqrt_config() -> ml_collections.ConfigDict:
  """Returns optimizer parameters as in the ViT 22b paper."""
  config = get_adam_config()
  # Parameters of the learning rate schedule.
  config.learning_rate = ml_collections.ConfigDict()
  config.learning_rate.schedule = 'big_vision_rsqrt'
  config.learning_rate.peak_value = 1e-3
  config.learning_rate.warmup_steps = 10_000
  config.learning_rate.cooldown_steps = 50_000
  config.learning_rate.timescale = 10_000
  return config


def get_imagenet_config(
    batch_size: int,
    resize_hi: int = 256,
    resize_lo: int = 224,
    randaug: str = '',
    data_dir: Optional[str] = None,
) -> ml_collections.ConfigDict:
  """Returns configuration for training/evaluating on ImageNet."""
  randaug = f'|{randaug}' if randaug and randaug[0] != '|' else randaug
  # pylint: disable=line-too-long
  pp_common_fn = lambda inkey: f'value_range(-1,1)|onehot(1000, inkey="{inkey}", outkey="labels")|keep("image", "labels")'
  pp_train = f'decode_jpeg_and_inception_crop({resize_lo})|flip_lr{randaug}|{pp_common_fn("label")}'
  pp_eval1 = f'decode|resize_small({resize_hi})|central_crop({resize_lo})|{pp_common_fn("label")}'
  pp_eval2 = f'decode|resize_small({resize_hi})|central_crop({resize_lo})|ignore_no_labels(labels_key="real_label")|{pp_common_fn("real_label")}'
  # pylint: enable=line-too-long
  return ml_collections.ConfigDict({
      'train': {
          'name': 'imagenet2012',
          'split': 'train[:99%]',
          'process': pp_train,
          'batch_size': batch_size,
          'data_dir': data_dir,
          'cache': 'loaded',
          'shuffle_buffer': 250_000,
      },
      'val': {
          'name': 'imagenet2012',
          'split': 'train[99%:]',
          'process': pp_eval1,
          'batch_size': batch_size,
          'data_dir': data_dir,
          'cache': 'batched',
      },
      'test': {
          'name': 'imagenet2012',
          'split': 'validation',
          'process': pp_eval1,
          'batch_size': batch_size,
          'data_dir': data_dir,
          'cache': 'batched',
      },
      'v2': {
          'name': 'imagenet_v2',
          'split': 'test',
          'process': pp_eval1,
          'batch_size': batch_size,
          'data_dir': data_dir,
          'cache': 'batched',
      },
      'real': {
          'name': 'imagenet2012_real',
          'split': 'validation',
          'process': pp_eval2,
          'batch_size': batch_size,
          'data_dir': data_dir,
          'cache': 'batched',
      },
  })


def get_vit_config(
    variant: str, patch_size: int, num_classes: Optional[int],
) -> ml_collections.ConfigDict:
  """Returns transformer parameters for different canonical architectures."""
  variant_idx = ['Ti', 'S', 'B', 'L', 'H'].index(variant)
  return ml_collections.ConfigDict({
      'name': 'VisionTransformerMoe',
      'num_classes': num_classes,
      'patch_size': (patch_size, patch_size),
      'hidden_size': [192, 384, 768, 1024, 1280][variant_idx],
      'classifier': 'gap',
      'head_bias_init': -math.log(num_classes) if num_classes else 0.0,
      'encoder': {
          'num_layers': [12, 12, 12, 24, 32][variant_idx],
          'mlp_dim': [768, 1536, 3072, 4096, 5120][variant_idx],
          'num_heads': [3, 6, 12, 16, 16][variant_idx],
          'dropout_rate': 0.0,
          'attention_dropout_rate': 0.0,
          'attention_qk_norm': True,
          'moe': {'layers': ()},
      },
  }, type_safe=False)


def get_vmoe_experts_choose_config(
    variant: str, patch_size: int, num_classes: Optional[int], *,
    image_size: int, num_experts: int, last_n: int,
    capacity_factor: float = 1.0,
) -> ml_collections.ConfigDict:
  """Returns a ViT model with MoE layers using the ExpertsChoose router."""
  config = get_vit_config(variant, patch_size, num_classes)
  config.encoder.moe = ml_collections.ConfigDict({
      'layers': tuple(range(config.encoder.num_layers))[-last_n:],
      'num_experts': num_experts,
      'group_size': (image_size // patch_size)**2,
      'split_rngs': False,
      'router': {
          'name': 'NoisyTopItemsPerExpertRouter',
          'noise_std': 1.0,
          'dispatcher': {
              'name': 'einsum',
              'bfloat16': True,
              'capacity_factor': capacity_factor,
              # Note: this is what it's used in the soft router, so we change
              # the defaults for a fair comparison. Otherwise, the actual
              # capacity_factor can be significantly bigger.
              'capacity_ceil_or_round': 'round',
              'capacity_multiple_of': 1,
              'partition_spec': (('expert', 'replica'),),
          },
      }
  })
  return config


def get_vmoe_soft_router_config(
    variant: str, patch_size: int, num_classes: Optional[int], *,
    image_size: int, num_experts: int, last_n: int,
    capacity_factor: Optional[float] = 1.0, num_slots: Optional[int] = None):
  """Returns a ViT model with MoE layers using the Soft router."""
  config = get_vit_config(variant, patch_size, num_classes)
  config.encoder.moe = ml_collections.ConfigDict({
      'layers': tuple(range(config.encoder.num_layers))[-last_n:],
      'num_experts': num_experts,
      'group_size': (image_size // patch_size)**2,
      'split_rngs': False,
      'router': {
          'name': 'SoftRouter',
          'capacity_factor': capacity_factor,
          'num_slots': num_slots,
          'partition_spec': (('expert', 'replica'),),
          'compute_similarity_metrics': True,
      }
  })
  return config
