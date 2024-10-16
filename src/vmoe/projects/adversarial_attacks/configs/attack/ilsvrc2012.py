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
r"""Runs PGD adversarial attacks on ILSVRC2012 using a public V-MoE model.

Notice that in the paper we performed experiments on JFT-300M, and ILSVRC2012
(a.k.a. ImageNet-1k) with a a pre-trained model on JFT-300M.
JFT-300M is a proprietary dataset and we cannot release those models. As an
example on how to attack a model, this config uses a model pre-trained on
ImageNet-21k and then fine-tuned to ILSVRC2012. Thus, these results won't match
those reported in the paper.

"""
# pylint: enable=line-too-long
import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Returns the configuration to run PGD adversarial attacks on a model."""
  config = ml_collections.ConfigDict()
  config.dataset = ml_collections.ConfigDict({
      'name': 'imagenet2012',
      'split': 'validation',
      'process': ('decode|resize(384)|value_range(-1,1)|keep("image", "label")|'
                  'onehot(1000, inkey="label", outkey="labels")'),
      'batch_size': 1024,
      'prefetch': 'autotune',
      'prefetch_device': 2,
  })
  # PGD options.
  config.num_updates = 40
  # Notice that RGB values are in the range [-1, +1], not [0, 1].
  config.max_epsilon = 1 / 127.5
  # Restore model to attack.
  # V-MoE B/32, E=32, K=2. Fine-tuned on ILSVRC2012.
  config.restore = ml_collections.ConfigDict({
      'from': 'checkpoint',
      'prefix':
          'gs://vmoe_checkpoints/vmoe_b16_imagenet21k_randaug_strong_ft_ilsvrc2012',
  })
  # Loss used to attack the model.
  config.loss = ml_collections.ConfigDict({
      'name': 'softmax_xent',
      'attack_auxiliary_loss': False,
  })
  # Definition of the model to attack (it must be compatible with the checkpoint
  # specified above).
  config.model = ml_collections.ConfigDict({
      'name': 'VisionTransformerMoe',
      'num_classes': 1000,
      'patch_size': (16, 16),
      'hidden_size': 768,
      'classifier': 'token',
      'representation_size': None,
      'encoder': {
          'num_layers': 12,
          'num_heads': 12,
          'mlp_dim': 3072,
          'dropout_rate': 0.0,
          'attention_dropout_rate': 0.0,
          'moe': {
              'num_experts': 8,
              'layers': (1, 3, 5, 7, 9, 11),
              'split_rngs': False,
              'group_size': ((384 // 16)**2 + 1) * 4,
              'router': {
                  'num_selected_experts': 2,
                  'noise_std': 1.0,
                  'importance_loss_weight': 0.005,
                  'load_loss_weight': 0.005,
                  'dispatcher': {
                      'name': 'einsum',
                      'bfloat16': True,
                      'capacity_factor': 1.5,
                      'partition_spec': (('expert', 'replica'),),
                      'batch_priority': False,
                  }
              }
          }
      }
  })
  # Specify how the model is partitioned. Place one expert on each device.
  config.num_expert_partitions = config.model.encoder.moe.num_experts
  config.params_axis_resources = [('Moe/Mlp/.*', ('expert',))]
  # Extra PRNG keys to generate.
  config.extra_rng_keys = ('dropout', 'gating')
  return config


def get_hyper(hyper):
  # Note that we process images in the range [-1, 1]. Thus these are equivalent
  # to 1/255...20/255 when RGB values are in [0, 1] range.
  max_epsilons = [(x + 1) / 127.5 for x in range(20)]
  # Extra epsilon values below 1/255 (in [0, 1] range).
  max_epsilons += [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3]
  return hyper.sweep('config.max_epsilon', max_epsilons)
