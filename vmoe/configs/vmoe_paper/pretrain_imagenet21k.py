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

# pylint: disable=line-too-long
r"""Train ViT model with MoE layers on ImageNet-21k.

This is the config for pre-training the model that was later fine-tuned on
ILSVRC 2012 and CIFAR 10. See the corresponding fine-tuning configs:
  - vmoe_b16_imagenet21k_randaug_strong_ft_cifar10.py
  - vmoe_b16_imagenet21k_randaug_strong_ft_ilsvrc2012.py

"""
# pylint: enable=line-too-long
import re

import ml_collections

DESCRIPTIONS_REGEX = re.compile(
    r'^ViT-(?P<variant>.)/(?P<patch>[0-9]+), '
    r'E=(?P<num_experts>[0-9]+), '
    r'K=(?P<k>[0-9]+), '
    r'(?P<where>Every|Last) (?P<where_num>[0-9]+), '
    r'(?P<epochs>[0-9]+) Epochs$')
# Number of ImageNet21k classes.
NUM_CLASSES = 21_843


def get_config():
  """Config to train V-MoE S/32, B/32, B/16, L/32, L/16 & H/14."""
  config = ml_collections.ConfigDict()

  config.dataset = ml_collections.ConfigDict()
  pp_common = f'value_range(-1,1)|onehot({NUM_CLASSES})|keep("image", "labels")'
  # Dataset variation used for training.
  config.dataset.train = ml_collections.ConfigDict()
  config.dataset.train.name = 'imagenet21k'
  config.dataset.train.split = 'full[102400:]'
  config.dataset.train.process = (
      f'decode_jpeg_and_inception_crop(224)|flip_lr|randaug(2,20)|{pp_common}'
  )
  config.dataset.train.shuffle_buffer = 250_000
  config.dataset.train.batch_size = 4096
  config.dataset.train.prefetch = 'autotune'
  config.dataset.train.prefetch_device = 2
  # Dataset variation used for evaluation.
  config.dataset.val = ml_collections.ConfigDict()
  config.dataset.val.name = 'imagenet21k'
  config.dataset.val.split = 'full[:102400]'
  config.dataset.val.process = (
      f'decode|resize_small(256)|central_crop(224)|{pp_common}'
  )
  config.dataset.val.batch_size = 4096
  config.dataset.val.cache = 'batched'
  config.dataset.val.prefetch = 'autotune'
  # Loss used to train the model.
  config.loss = ml_collections.ConfigDict()
  config.loss.name = 'sigmoid_xent'
  # Model and optimizer parameters depend on the model type.
  config.description = 'ViT-B/16, E=8, K=2, Every 2, 300 Epochs'
  config.model = get_vmoe_params(config.description)
  config.optimizer = get_optimizer_params(config.description)
  config.train_epochs = get_num_epochs(config.description)
  config.mixup = ml_collections.ConfigDict()
  config.mixup.concentration = 0.5
  config.mixup.mixup_size = 2

  # These control how the model parameters are partitioned across the device
  # mesh for running the models efficiently.
  config.num_expert_partitions = config.model.encoder.moe.num_experts
  config.params_axis_resources = [('Moe/Mlp/.*', ('expert',))]
  config.extra_rng_keys = ('dropout', 'gating', 'mixup')
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

  config.seed = 0

  return config


def get_vmoe_params(description: str,
                    image_size: int = 224) -> ml_collections.ConfigDict:
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
  # Group size must be a divisor of the number of tokens per device.
  # We assume here that the smallest batch size per device (images/device) is 8,
  # and any other batch size per device will be a multiple of this.
  min_batch_size_per_device = 8
  num_patches = (image_size // patch_size) * (image_size // patch_size) + 1
  group_size = min_batch_size_per_device * num_patches

  config = ml_collections.ConfigDict()
  config.name = 'VisionTransformerMoe'
  config.num_classes = NUM_CLASSES
  config.patch_size = (patch_size, patch_size)
  config.hidden_size = [512, 768, 1024, 1280][variant_idx]
  config.classifier = 'token'
  config.representation_size = None
  config.head_bias_init = -10.0
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


def get_optimizer_params(description: str) -> ml_collections.ConfigDict:
  """Returns optimizer parameters for different canonical architectures."""
  match = re.match(DESCRIPTIONS_REGEX, description)
  if not match:
    raise ValueError(f"Description {description!r} doesn't match the regex.")

  variant = match.group('variant')
  patch_size = int(match.group('patch'))

  config = ml_collections.ConfigDict()
  config.name = 'adam'
  config.b1 = 0.9
  config.b2 = 0.999
  config.mu_dtype = 'float32'  # Optionally, use bfloat16 to save memory.
  # config.weight_decay = 0.1  # Weight decay is applied to all parameters.
  config.weight_decay = [('.*/kernel', 0.1)]

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
  config.gradient_clip = ml_collections.ConfigDict()
  config.gradient_clip.global_norm = 1.0
  return config


def get_num_epochs(description) -> int:
  match = re.match(DESCRIPTIONS_REGEX, description)
  if not match:
    raise ValueError(f"Description {description!r} doesn't match the regex.")
  return int(match.group('epochs'))


def get_hyper(hyper):
  # Adjust this to train with multiple seed or adjust other hyperparameters.
  return hyper.product([])
