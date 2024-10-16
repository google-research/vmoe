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
r"""Fine-tune gs://vmoe_checkpoints/vmoe_b16_imagenet21k_randaug_strong on ILSVRC2012 with three seeds.

Accuracy (mean over 3 runs with different fine-tuning seeds):
  - ILSVRC2012, validation: 89.2%
  - ILSVRC2012, test: 85.6%
  - ILSVRC2012 ReaL, test: 84.1%
  - ImageNet V2, test: 76.1%

"""
# pylint: enable=line-too-long
import ml_collections
from vmoe.configs.vmoe_paper import common

# Paths to manually downloaded datasets and to the tensorflow_datasets data dir.
TFDS_MANUAL_DIR = None
TFDS_DATA_DIR = None
# The following configuration was made to fit on TPUv3-32. The number of images
# per device has to be at least 32.
BATCH_SIZE = 1024    # Number of images processed in each step.
NUM_CLASSES = 1_000  # Number of ILSVRC2012 classes.
IMAGE_SIZE = 384     # Image size as input to the model.


def get_config():
  """Fine-tune gs://vmoe_checkpoints/vmoe_b16_imagenet21k_randaug_strong on ILSVRC2012."""
  config = common.get_base_config()

  config.dataset = ml_collections.ConfigDict()
  pp_common = f'value_range(-1,1)|onehot({NUM_CLASSES}, inkey="label", outkey="labels")|keep("image", "labels")'
  # Dataset variation used for training.
  config.dataset.train = get_data_config(
      name='imagenet2012',
      split='train[:99%]',
      process=f'decode_jpeg_and_inception_crop({IMAGE_SIZE})|flip_lr|{pp_common}',
      shuffle_buffer=50_000,
      cache=None)
  # Dataset variation used for validation.
  config.dataset.val = get_data_config(
      name='imagenet2012',
      split='train[99%:]',
      process=f'decode|resize({IMAGE_SIZE})|{pp_common}',
      shuffle_buffer=None,
      cache='batched')
  # Dataset variation used for test.
  config.dataset.test = get_data_config(
      name='imagenet2012',
      split='validation',
      process=f'decode|resize({IMAGE_SIZE})|{pp_common}',
      shuffle_buffer=None,
      cache='batched')
  # Dataset variation used for test with "real" labels.
  config.dataset.test_real = get_data_config(
      name='imagenet2012_real',
      split='validation',
      process=(f'decode|resize({IMAGE_SIZE})|value_range(-1,1)|'
               f'onehot({NUM_CLASSES}, inkey="real_label", outkey="labels")|'
               f'keep("image", "labels")'),
      shuffle_buffer=None,
      cache='batched')
  config.dataset.imagenet_v2 = get_data_config(
      name='imagenet_v2',
      split='test',
      process=f'decode|resize({IMAGE_SIZE})|{pp_common}',
      shuffle_buffer=None,
      cache='batched')
  # Loss used to train the model.
  config.loss = ml_collections.ConfigDict()
  config.loss.name = 'softmax_xent'
  # Fine-tuning steps.
  config.train_steps = 10_000
  # Description of the upstream model to fine-tune.
  config.description = 'ViT-B/16, E=8, K=2, Every 2, 300 Epochs'
  config.model = get_vmoe_config(config.description)
  # Model initialization from the released checkpoints.
  config.initialization = ml_collections.ConfigDict({
      'name': 'initialize_from_vmoe',
      'prefix': 'gs://vmoe_checkpoints/vmoe_b16_imagenet21k_randaug_strong',
      'rules': [
          ('head', ''),              # Do not restore the head params.
          # We pre-trained on 224px and are finetuning on 384px.
          # Resize positional embeddings.
          ('^(.*/pos_embedding)$', r'params/\1', 'vit_zoom'),
          # Restore the rest of parameters without any transformation.
          ('^(.*)$', r'params/\1'),
      ],
      # We are not initializing several arrays from the new train state, do not
      # raise an exception.
      'raise_if_target_unmatched': False,
      # Partition MoE parameters when reading from the checkpoint.
      'axis_resources_regexes': [('Moe/Mlp/.*', ('expert',))],
  })
  config.optimizer = ml_collections.ConfigDict({
      'name': 'sgd',
      'momentum': 0.9,
      'accumulator_dtype': 'float32',
      'learning_rate': {
          'schedule': 'warmup_cosine_decay',
          'peak_value': 0.003,
          'end_value': 1e-5,
          'warmup_steps': 500,
      },
      'gradient_clip': {'global_norm': 10.0},
  })
  # These control how the model parameters are partitioned across the device
  # mesh for running the models efficiently.
  # By setting num_expert_partitions = num_experts, we set at most one expert on
  # each device.
  config.num_expert_partitions = config.model.encoder.moe.num_experts
  # This value specifies that the first axis of all parameters in the MLPs of
  # MoE layers (which has size num_experts) is partitioned across the 'expert'
  # axis of the device mesh.
  config.params_axis_resources = [('Moe/Mlp/.*', ('expert',))]
  config.extra_rng_keys = ('dropout', 'gating')

  return config


def get_data_config(name, split, process, shuffle_buffer, cache):
  """Returns dataset parameters."""
  config = common.get_data_config(
      name=name, split=split, process=process, batch_size=BATCH_SIZE,
      shuffle_buffer=shuffle_buffer, cache=cache)
  config.data_dir = TFDS_DATA_DIR
  config.manual_dir = TFDS_MANUAL_DIR
  return config


def get_vmoe_config(description: str) -> ml_collections.ConfigDict:
  config = common.get_vmoe_config(description, IMAGE_SIZE, NUM_CLASSES)
  config.representation_size = None
  config.encoder.moe.router.dispatcher.capacity_factor = 1.5
  return config


def get_hyper(hyper):
  return hyper.sweep('config.seed', list(range(3)))
