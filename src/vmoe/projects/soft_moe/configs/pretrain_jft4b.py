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
r"""Train different models used in the Soft MoE paper.

This includes the configs used for some of the "Long training runs" results.

Notice that we pretrain on JFT-4B, a proprietary dataset by Google, which is not
available externally. Feel free to use this config file as a template and adapt
it to train on your favorite dataset.

"""
# pylint: enable=line-too-long
import ml_collections
from vmoe.projects.soft_moe.configs import common

BATCH_SIZE = 4096
DATASET = 'jft4b'  # This is a proprietary dataset by Google.
NUM_CLASSES = 29_593
PP_COMMON = f'value_range(-1,1)|onehot({NUM_CLASSES})|keep("image", "labels")'


def get_default_moe_num_experts_and_last_n(variant, patch_size):
  """Default number of experts and MoE layers for Sparse and Soft MoEs."""
  num_experts = {
      ('S', 16): 128,
      ('S', 14): 256,
      ('B', 16): 128,
      ('L', 16): 128,
      ('H', 14): 256,
  }[(variant, patch_size)]
  last_n = {
      'S': 6,
      'B': 6,
      'L': 12,
      'H': 16,
  }[variant]
  return num_experts, last_n


def get_config(model='soft-s16') -> ml_collections.ConfigDict:
  """Config to train different models used in the Soft MoE paper."""
  # Parse model argument.
  model_type, model_backbone = model.split('-')
  patch_size = int(model_backbone[1:])
  variant = model_backbone[0].upper()

  config = common.get_base_config()
  config.dataset = ml_collections.ConfigDict()
  config.dataset.train = common.get_data_config(
      name=DATASET,
      split='full[16384:]',
      batch_size=BATCH_SIZE,
      process=f'decode_jpeg_and_inception_crop(224)|flip_lr|{PP_COMMON}',
  )
  config.dataset.val = common.get_data_config(
      name=DATASET,
      split='full[:16384]',
      batch_size=BATCH_SIZE,
      process=f'decode|resize_small(256)|central_crop(224)|{PP_COMMON}',
      cache='batched',
  )
  config.fewshot = common.get_fewshot_config(
      batch_size=BATCH_SIZE, resize_resolution=256, target_resolution=224)
  config.loss = ml_collections.ConfigDict({'name': 'sigmoid_xent'})
  config.optimizer = common.get_optimizer_rsqrt_config()
  # Model hyperparameters depend on the model type.
  if model_type == 'vit':
    config.model = common.get_vit_config(variant, patch_size, NUM_CLASSES)
  elif model_type == 'ec':
    num_experts, last_n = get_default_moe_num_experts_and_last_n(
        variant, patch_size)
    config.model = common.get_vmoe_experts_choose_config(
        variant, patch_size, NUM_CLASSES, image_size=224,
        num_experts=num_experts, last_n=last_n, capacity_factor=1.0)
  elif model_type == 'soft':
    num_experts, last_n = get_default_moe_num_experts_and_last_n(
        variant, patch_size)
    config.model = common.get_vmoe_soft_router_config(
        variant, patch_size, NUM_CLASSES, image_size=224,
        num_experts=num_experts, last_n=last_n, capacity_factor=None,
        num_slots=1)
    config.model.encoder.moe.router.compute_similarity_metrics = False
    config.optimizer.weight_decay = config.optimizer.weight_decay + (
        ('.*/Moe/Router/scale', 0.03),  # SoftMoE doesn't have a kernel param.
    )
  else:
    raise ValueError(f'Unknown model type: {model_type!r}')
  if variant == 'H':
    config.train_steps = 2_000_000
  else:
    config.train_steps = 4_000_000
  # These control how the train state is partitioned across the device mesh.
  if model_type == 'vit':
    config.num_expert_partitions = 1
    config.params_axis_resources = []
  else:
    config.num_expert_partitions = config.model.encoder.moe.num_experts
    config.params_axis_resources = [('Moe/Mlp/.*', ('expert',))]
  config.extra_rng_keys = ('dropout', 'gating')
  # Plot summary of different arrays.
  config.summarize_arrays = ml_collections.ConfigDict({
      'rules': [
          'opt_state/.*/hyperparams/learning_rate',  # Learning rate.
          'params/.*/Moe/Router/scale',              # Soft MoE scale.
      ],
      # Maximum values reported per rule and array.
      # If you are reporting individual values for every expert parameter,
      # increase this accordingly.
      'max_summary_values': 1,
  })
  # Keep checkpoints every 50k steps, useful to do intermediate cooldowns.
  config.save_checkpoint.keep_last = 2
  config.save_checkpoint.keep_steps_multiple_of = 50_000
  return config


def get_hyper(hyper, model='soft-s16'):
  del model
  return hyper.product([])
