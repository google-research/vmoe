# Copyright 2021 Google LLC.
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
r"""Fine-tune gs://vmoe_checkpoints/vmoe_b16_imagenet21k_randaug_strong on CIFAR10.

Test accuracy (mean over 3 runs with different fine-tuning seeds): 98.7%.

The accuracy is not SOTA. This config file was designed to easily fit on a small
TPUv2-8 or TPUv3-8, and fine-tune in about 10 minutes (TPUv3-8).

"""
# pylint: enable=line-too-long
import ml_collections

# Paths to manually downloaded datasets and to the tensorflow_datasets data dir.
TFDS_MANUAL_DIR = None
TFDS_DATA_DIR = None
# The following configuration was made to fit on TPUv3-32. The number of images
# per device has to be at least 32.
BATCH_SIZE = 1024    # Number of images processed in each step.
NUM_CLASSES = 10     # Number of CIFAR10 classes.
IMAGE_SIZE = 128     # Image size as input to the model.
PATCH_SIZE = 16      # Patch size.
NUM_LAYERS = 12      # Number of encoder blocks in the transformer.
NUM_EXPERTS = 8      # Number of experts in each MoE layer.
NUM_SELECTED_EXPERTS = 2  # Maximum number of selected experts per token.
# For efficiency reasons, the tokens are divided in several groups of the
# following size. The routing is performed independently on each group.
# For efficiency reasons, the group size should be a divisor of the number of
# tokens in each device. The resulting number of groups MUST be a multiple of
# the number of experts.
GROUP_SIZE = 8 * ((IMAGE_SIZE // PATCH_SIZE)**2 + 1)
# This is the number of tokens that are processed per expert per group.
# We give some slack to the expected number of tokens per expert, if the routing
# was perfectly balanced.
CAPACITY_SIZE_RATIO = 1.5
CAPACITY = int(GROUP_SIZE * CAPACITY_SIZE_RATIO * NUM_SELECTED_EXPERTS //
               NUM_EXPERTS)


def get_config():
  """Fine-tune gs://vmoe_checkpoints/vmoe_b16_imagenet21k_randaug_strong on CIFAR10."""
  config = ml_collections.ConfigDict()

  config.dataset = ml_collections.ConfigDict()
  pp_common = f'value_range(-1,1)|onehot({NUM_CLASSES}, inkey="label", outkey="labels")|keep("image", "labels")'
  # Dataset variation used for training.
  config.dataset.train = get_data_params(
      name='cifar10',
      split='train[:98%]',
      process=f'decode|inception_crop({IMAGE_SIZE})|flip_lr|{pp_common}',
      shuffle_buffer=50_000,
      cache=None)
  # Dataset variation used for validation.
  config.dataset.val = get_data_params(
      name='cifar10',
      split='train[98%:]',
      process=f'decode|resize({IMAGE_SIZE})|{pp_common}',
      shuffle_buffer=None,
      cache='batched')
  # Dataset variation used for test.
  config.dataset.test = get_data_params(
      name='cifar10',
      split='test',
      process=f'decode|resize({IMAGE_SIZE})|{pp_common}',
      shuffle_buffer=None,
      cache='batched')
  # Loss used to train the model.
  config.loss = ml_collections.ConfigDict()
  config.loss.name = 'softmax_xent'
  # Model parameters depend on the model type.
  config.description = 'V-MoE-B/16, K=2, Every 2'
  config.train_steps = 1_000
  config.initialization = ml_collections.ConfigDict({
      'name': 'initialize_from_vmoe_release',
      'prefix': 'gs://vmoe_checkpoints/vmoe_b16_imagenet21k_randaug_strong',
      'keep': ['head'],
  })
  config.model = ml_collections.ConfigDict({
      'name': 'VisionTransformerMoe',
      'num_classes': NUM_CLASSES,
      'patch_size': (16, 16),
      'hidden_size': 768,
      'classifier': 'token',
      'representation_size': None,
      'head_bias_init': -10.0,
      'encoder': {
          'num_layers': NUM_LAYERS,
          'num_heads': 12,
          'mlp_dim': 3072,
          'dropout_rate': 0.0,
          'attention_dropout_rate': 0.0,
          'moe': {
              'num_experts': NUM_EXPERTS,
              'group_size': GROUP_SIZE,
              'layers': tuple(range(1, NUM_LAYERS, 2)),
              'dropout_rate': 0.0,
              'split_rngs': False,  # All experts share initialization.
              'router': {
                  'num_selected_experts': NUM_SELECTED_EXPERTS,
                  'noise_std': 1.0,  # This is divided by NUM_EXPERTS.
                  'importance_loss_weight': 0.005,
                  'load_loss_weight': 0.005,
                  'dispatcher': {
                      'name': 'einsum',
                      'bfloat16': True,
                      'capacity': CAPACITY,
                      # This is used to hint pjit about how data is distributed
                      # at the input/output of each MoE layer.
                      # This value means that the tokens are partitioned across
                      # all devices in the mesh (i.e. fully data parallelism).
                      'partition_spec': (('expert', 'replica'),),
                      # We don't use batch priority for training/fine-tuning.
                      'batch_priority': False,
                  },
              },
          },
      },
  })
  config.optimizer = ml_collections.ConfigDict({
      'name': 'sgd',
      'momentum': 0.9,
      'accumulator_dtype': 'float32',
      'learning_rate': {
          'schedule': 'warmup_cosine_decay',
          'peak_value': 0.0015,
          'end_value': 1e-5,
          'warmup_steps': 100,
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
  # Write checkpoints every 1000 steps.
  config.save_checkpoint = ml_collections.ConfigDict()
  config.save_checkpoint.every_steps = 1_000
  config.save_checkpoint.keep_last = 1
  config.save_checkpoint.num_shards = 32  # Target number of checkpoint shards.
  config.save_checkpoint.wait_seconds = 1.0
  # Report training progress every minute.
  config.report_progress = ml_collections.ConfigDict()
  config.report_progress.every_secs = None
  config.report_progress.every_steps = 100
  # Evaluate on the validation set every 1000 steps.
  config.evaluate = ml_collections.ConfigDict()
  config.evaluate.every_steps = 100
  # Run device profiling on process_index = 0, for 5 steps, starting at step 10.
  # Then repeat profiling every hour.
  config.profile = ml_collections.ConfigDict()
  config.profile.all_processes = False
  config.profile.num_profile_steps = 5
  config.profile.first_profile = 10
  config.profile.every_secs = 3600.0

  config.seed = 0

  return config


def get_data_params(name, split, process, shuffle_buffer, cache):
  """Returns dataset parameters."""
  config = ml_collections.ConfigDict()
  config.name = name
  config.split = split
  config.process = process
  config.batch_size = BATCH_SIZE
  config.prefetch = 'autotune'
  config.prefetch_device = 2
  config.data_dir = TFDS_DATA_DIR
  config.manual_dir = TFDS_MANUAL_DIR
  if shuffle_buffer:
    config.shuffle_buffer = shuffle_buffer
  if cache:
    config.cache = cache
  return config


def get_hyper(hyper):
  return hyper.sweep('config.seed', list(range(3)))
