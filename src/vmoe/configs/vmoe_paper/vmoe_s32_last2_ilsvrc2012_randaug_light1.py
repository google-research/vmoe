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
r"""Train ViT model with MoE layers on ImageNet-1k (a.k.a. ILSVRC2012) for 300 epochs.

Important disclaimer for comparisons: Although we named this backbone architecture
ViT-S, it is different from the ViT-S described in https://arxiv.org/abs/2106.10270
and https://arxiv.org/abs/2106.04560. Most notably, we have only 8 layers, while
these works use 12 layers in the ViT-S backbone. Unfortunately, all these were
concurrent works and we used the same name for slightly different things.

Pre-training takes about 5h50m on a TPUv3-32.
From-scratch accuracy (mean over 3 runs with different seeds):
  - ILSVRC2012, validation: 76.8%
  - ILSVRC2012, test: 73.1%
  - ILSVRC2012 ReaL, test: 74.4%
  - ImageNet V2, test: 59.2%

Fine-tuning takes about 48m on a TPUv3-8.
Accuracy at a higher resolution (384px, mean over 3 different fine-tuning seeds,
check the file vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012):
  - ILSVRC2012, validation: 78.8%
  - ILSVRC2012, test: 75.9%
  - ILSVRC2012 ReaL, test: 77.1%
  - ImageNet V2, test: 62.0%

"""
# pylint: enable=line-too-long

import ml_collections
from vmoe.configs.vmoe_paper import common

# Paths to manually downloaded datasets and to the tensorflow_datasets data dir.
TFDS_DATA_DIR = None
TFDS_MANUAL_DIR = None
NUM_CLASSES = 1_000
BATCH_SIZE = 8192
PP_COMMON = f'value_range(-1,1)|onehot({NUM_CLASSES}, inkey="label", outkey="labels")|keep("image", "labels")'


def get_config():
  """Config to train V-MoE S/32 on ILSVRC2012 from scratch."""
  config = common.get_base_config()

  config.dataset = ml_collections.ConfigDict()
  # Dataset variation used for training.
  config.dataset.train = get_data_params_train(
      'imagenet2012', 'train[:99%]', 'light1')
  # Dataset variations used for evaluation.
  config.dataset.val = get_data_params_eval('imagenet2012', 'train[99%:]')
  config.dataset.test = get_data_params_eval('imagenet2012', 'validation')
  config.dataset.imagenet_v2 = get_data_params_eval('imagenet_v2', 'test')
  config.dataset.imagenet_real = common.get_data_config(
      name='imagenet2012_real',
      split='validation',
      process=(f'decode|resize_small(256)|central_crop(224)|value_range(-1,1)|'
               f'onehot({NUM_CLASSES}, inkey="real_label", outkey="labels")|'
               f'keep("image", "labels")'),
      shuffle_buffer=None,
      batch_size=BATCH_SIZE,
      cache='batched')
  # Loss used to train the model.
  config.loss = ml_collections.ConfigDict()
  config.loss.name = 'sigmoid_xent'
  # Model and optimizer parameters depend on the model type.
  config.description = 'ViT-S/32, E=8, K=2, Last 2, 300 Epochs'
  config.model = get_vmoe_params(config.description)
  config.optimizer = get_optimizer_params(config.description)
  config.train_epochs = common.get_num_epochs(config.description)
  # Mixup options.
  config.mixup = common.get_mixup_config('light1')
  # These control how the model parameters are partitioned across the device
  # mesh for running the models efficiently.
  config.num_expert_partitions = config.model.encoder.moe.num_experts
  config.params_axis_resources = [('Moe/Mlp/.*', ('expert',))]
  config.extra_rng_keys = ('dropout', 'gating', 'mixup')
  # Use higher capacity for evaluation.
  config.evaluate.model_overrides = ml_collections.ConfigDict({
      'encoder': {'moe': {'router': {'dispatcher': {'capacity_factor': 4.0}}}}})

  return config


def get_data_params_train(
    name: str, split: str, aug: str) -> ml_collections.ConfigDict:
  """Returns data config for training."""
  randaug = common.get_randaug(aug)
  process = f'decode_jpeg_and_inception_crop(224)|flip_lr|{randaug}|{PP_COMMON}'
  config = common.get_data_config(
      name=name, split=split, batch_size=BATCH_SIZE, process=process,
      shuffle_buffer=50_000, cache=None)
  config.data_dir = TFDS_DATA_DIR
  config.manual_dir = TFDS_MANUAL_DIR
  config.prefetch = 16
  config.prefetch_device = 1
  return config


def get_data_params_eval(name: str, split: str) -> ml_collections.ConfigDict:
  """Returns data config for evaluation."""
  process = f'decode|resize_small(256)|central_crop(224)|{PP_COMMON}'
  config = common.get_data_config(
      name=name, split=split, batch_size=BATCH_SIZE, process=process,
      shuffle_buffer=None, cache='batched')
  config.data_dir = TFDS_DATA_DIR
  config.manual_dir = TFDS_MANUAL_DIR
  config.prefetch = 1
  config.prefetch_device = 1
  return config


def get_optimizer_params(description: str) -> ml_collections.ConfigDict:
  """Returns optimizer parameters for different canonical architectures."""
  config = common.get_optimizer_config(description)
  # Overwrite these params, different from the standard ones in the V-MoE paper.
  config.weight_decay = [('.*/kernel', 0.1)]
  config.learning_rate.peak_value = 3e-3
  config.gradient_clip.global_norm = 1.0
  return config


def get_vmoe_params(description: str) -> ml_collections.ConfigDict:
  config = common.get_vmoe_config(
      description, image_size=224, num_classes=NUM_CLASSES)
  config.encoder.moe.dropout_rate = 0.2
  return config


def get_hyper(hyper):
  # Adjust this to train with multiple seed or adjust other hyperparameters.
  return hyper.sweep('config.seed', [0, 1, 2])
