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
r"""Train different models used in the Soft MoE paper on the LAION dataset.

"""
# pylint: enable=line-too-long
import ml_collections
from vmoe.projects.soft_moe.configs import common

BATCH_SIZE = 16_384
DATASET = 'laion400m'


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


# pylint: disable=line-too-long
def tokenize(inkey: str, outkey: str = 'text') -> str:
  return f'tokenize(max_len=16, model="c4_en", eos="sticky", inkey="{inkey}", outkey="{outkey}", pad_value=1)'
# pylint: enable=line-too-long


def get_config(model='soft-s16') -> ml_collections.ConfigDict:
  """Config to train different models used in the Soft MoE paper."""
  # Parse model argument.
  model_type, model_backbone = model.split('-')
  patch_size = int(model_backbone[1:])
  variant = model_backbone[0].upper()

  # SoftMoEs highly benefit from data augmentation, while ViTs and MoEs with
  # Experts Choice routing actually do worse. See Figure 15 in the paper.
  #
  if model_type in ('vit', 'ec'):
    process_str = 'decode|resize(256)'
  else:
    process_str = 'decode_jpeg_and_inception_crop(256)'

  config = common.get_base_config()
  config.dataset = ml_collections.ConfigDict()
  config.dataset.train = common.get_data_config(
      name=DATASET,
      split='full[16384:]',
      batch_size=BATCH_SIZE,
      process=(
          f'{process_str}|value_range(-1,1)|flatten|'
          f'{tokenize("text")}|keep("image", "text")'
      ),
      shuffle_buffer=250_000,
  )
  config.fewshot = common.get_fewshot_config(
      batch_size=1_024, resize_resolution=292, target_resolution=256,
      every_steps=10_000, seeds_per_step=3)
  config.fewshot.model_overrides = ml_collections.ConfigDict()
  config.retrieval = ml_collections.ConfigDict({
      'batch_size': 1_024,
      'every_steps': 10_000,
      'datasets': {
          'coco': {
              'dataset': 'coco_captions',
              'txt_name': ('captions', 'text'),
              'pp_img': 'resize(256)|value_range(-1, 1)',
              'pp_txt': f'{tokenize(inkey="texts", outkey="labels")}',
          },
          'flickr': {
              'dataset': 'argus:flickr30k/captions',
              'txt_name': 'texts',
              'pp_img': 'resize(256)|value_range(-1, 1)',
              'pp_txt': f'{tokenize(inkey="texts", outkey="labels")}',
          },
      }
  })
  config.zeroshot = ml_collections.ConfigDict({
      'batch_size': 1_024,
      'every_steps': 10_000,
      'pp_img': 'resize(256)|value_range(-1, 1)',
      'pp_txt': f'{tokenize(inkey="texts", outkey="labels")}',
      'datasets': {
          'cifar100': {},
          'imagenet2012': {'class_names': 'clip', 'split': 'validation'},
          'oxford_iiit_pet': {},
      },
  })

  # Optimizer configuration.
  config.optimizer = common.get_optimizer_rsqrt_config()
  config.optimizer.weight_decay = (('.*/kernel', 0.1),)
  config.optimizer.learning_rate.warmup_steps = 20_000
  config.optimizer.learning_rate.cooldown_steps = 20_000
  config.train_steps = 750_000

  config.model = ml_collections.ConfigDict({
      'name': 'vmoe.projects.contrastive.models.TwoTower',
      'bias_init': -10.0,
      'scale_init': 10.0,
  })

  # Image encoder hyperparameters depend on the model type.
  if model_type == 'vit':
    config.model.image = common.get_vit_config(variant, patch_size, None)
  elif model_type == 'ec':
    num_experts, last_n = get_default_moe_num_experts_and_last_n(
        variant, patch_size)
    config.model.image = common.get_vmoe_experts_choose_config(
        variant, patch_size, None, image_size=256,
        num_experts=num_experts, last_n=last_n, capacity_factor=1.0)
  elif model_type == 'soft':
    num_experts, last_n = get_default_moe_num_experts_and_last_n(
        variant, patch_size)
    config.model.image = common.get_vmoe_soft_router_config(
        variant, patch_size, None, image_size=256,
        num_experts=num_experts, last_n=last_n, capacity_factor=None,
        num_slots=1)
    config.model.image.encoder.moe.router.compute_similarity_metrics = False
  else:
    raise ValueError(f'Unknown model type: {model_type!r}')

  # Text encoder is a B size model.
  config.model.text = ml_collections.ConfigDict({
      'vocab_size': 32_000,
      'num_classes': config.model.image.hidden_size,
      'hidden_size': 768,
      'encoder': {
          'num_layers': 12,
          'mlp_dim': 3072,
          'num_heads': 12,
          'dropout_rate': 0.0,
          'attention_dropout_rate': 0.0,
          'attention_qk_norm': True,
          'moe': {'layers': ()},
      }
  })

  # These control how the train state is partitioned across the device mesh.
  if model_type == 'vit':
    config.num_expert_partitions = 1
    config.params_axis_resources = []
  else:
    config.num_expert_partitions = config.model.image.encoder.moe.num_experts
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
