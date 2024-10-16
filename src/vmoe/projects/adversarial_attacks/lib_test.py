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

"""Tests for lib."""
import io
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import ml_collections
import numpy as np
import optax
import tensorflow as tf
from vmoe.projects.adversarial_attacks import lib

TfDatasetIterator = lib.input_pipeline.TfDatasetIterator


def _create_model_with_routing(batch_size: int, image_size: int):
  class ModelWithRouting(nn.Module):
    num_classes: int = 10
    num_experts: int = 2

    @nn.compact
    def __call__(self, x):
      # Per-example routing with as many groups as experts.
      x = x.reshape(self.num_experts, x.shape[0] // self.num_experts, -1)
      router = lib.restore.routing.NoisyTopExpertsPerItemRouter(
          num_experts=self.num_experts,
          num_selected_experts=2,
          deterministic=True,
          dispatcher={
              'name': 'einsum',
              'batch_priority': False,
              'capacity_factor': 1.0,
          },
          name='router')
      dispatcher, metrics = router(x)
      # MoE of linear layers.
      moe_layer = lib.restore.moe.sparse_moe_spmd(
          nn.Dense, has_aux=False, variable_axes={'params': 0},
          split_rngs={'params': True})(features=self.num_classes)
      logits = moe_layer(dispatcher, x)
      logits = logits.reshape(-1, logits.shape[-1])
      return logits, metrics

  flax_module = ModelWithRouting()
  image_size = (batch_size, image_size, image_size, 3)
  variables = flax_module.init(jax.random.PRNGKey(0), np.zeros(image_size))
  variables_axis_resources = jax.tree_util.tree_map(
      lambda _: lib.PartitionSpec(), variables)
  router_keys = {'router/__call__'}
  loss_fn = lambda a, b, _: optax.softmax_cross_entropy(a, b)
  return (flax_module, variables, variables_axis_resources, loss_fn,
          router_keys, {})


def _create_model_without_routing(batch_size: int, image_size: int):
  class ModelWithoutRouting(nn.Module):
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
      x = nn.Dense(features=self.num_classes)(x)
      logits = jax.numpy.mean(x, axis=(1, 2))
      return logits, {}

  flax_module = ModelWithoutRouting()
  image_size = (batch_size, image_size, image_size, 3)
  variables = flax_module.init(jax.random.PRNGKey(0), np.zeros(image_size))
  variables_axis_resources = jax.tree_util.tree_map(
      lambda _: lib.PartitionSpec(), variables)
  loss_fn = lambda a, b, _: optax.softmax_cross_entropy(a, b)
  return flax_module, variables, variables_axis_resources, loss_fn, {}, {}


class RunPGDAttackTest(parameterized.TestCase):
  NUM_BATCHES = 2
  BATCH_SIZE = 32
  IMAGE_SIZE = 8
  NUM_CLASSES = 10
  NUM_UPDATES = 3

  def setUp(self):
    super().setUp()
    self.mock_restore_from_config = self.enter_context(
        mock.patch.object(lib.restore, 'restore_from_config', autospec=True))
    # Mock get_dataset to return a fake dataset.
    _ = self.enter_context(
        mock.patch.object(lib, 'get_dataset', autospec=True,
                          return_value=self._create_fake_dataset()))
    # Mock get_dat_num_examples to return the fake number of examples.
    _ = self.enter_context(
        mock.patch.object(
            lib.input_pipeline,
            'get_data_num_examples',
            autospec=True,
            return_value=self.NUM_BATCHES * self.BATCH_SIZE))

  def _create_fake_dataset(self) -> tf.data.Dataset:
    image_size = (self.NUM_BATCHES, self.BATCH_SIZE, self.IMAGE_SIZE,
                  self.IMAGE_SIZE, 3)
    label_size = (self.NUM_BATCHES, self.BATCH_SIZE, 1)
    return TfDatasetIterator(tf.data.Dataset.from_tensor_slices({
        'image':
            np.random.normal(size=image_size).astype(np.float32),
        'labels':
            (np.random.randint(0, self.NUM_CLASSES, size=label_size) ==
             np.arange(self.NUM_CLASSES)[None, None, :]),
        lib.VALID_KEY:
            # Note: images in the second half of the last batch are not valid.
            np.concatenate([
                np.ones((self.NUM_BATCHES - 1, self.BATCH_SIZE)),
                np.concatenate([
                    np.ones((1, self.BATCH_SIZE // 2)),
                    np.zeros((1, self.BATCH_SIZE // 2)),
                ], axis=1)
            ], axis=0)
    }), checkpoint=False)

  @parameterized.named_parameters(
      ('_restore_from_checkpoint_with_routing',
       {'from': 'checkpoint', 'prefix': '/foo/bar'},
       _create_model_with_routing),
      ('_restore_from_checkpoint_without_routing',
       {'from': 'checkpoint', 'prefix': '/foo/bar'},
       _create_model_without_routing),
  )
  def test(self, restore, create_model_fn):
    self.mock_restore_from_config.return_value = create_model_fn(
        self.BATCH_SIZE, self.IMAGE_SIZE)
    config = ml_collections.ConfigDict({
        'dataset': {
            'batch_size': self.BATCH_SIZE,
        },
        'restore': restore,
        'num_updates': self.NUM_UPDATES,
        'max_epsilon': 0.001,
        'attack_auxiliary_loss': False,
    })
    workdir = self.create_tempdir().full_path
    mesh = jax.sharding.Mesh(np.asarray(jax.local_devices()).reshape((-1, 1)),
                             ('expert', 'replica'))
    with mesh:
      lib.run_pgd_attack(config, workdir, mesh, writer=mock.MagicMock())
    with io.open(os.path.join(workdir, 'pgd_state.npz'), 'rb') as fp:
      pgd_state = dict(np.load(fp))
    self.assertEqual(
        pgd_state['num_images'],
        (self.NUM_BATCHES - 1) * self.BATCH_SIZE + self.BATCH_SIZE // 2)
    self.assertEqual(pgd_state['num_correct'].shape, (2,))
    self.assertEqual(pgd_state['sum_loss'].shape, (2,))
    for k, v in pgd_state.items():
      if k not in ('num_images', 'num_correct', 'sum_loss'):
        self.assertEqual(v.shape, (), msg=f'Wrong shape for key {k}')


if __name__ == '__main__':
  absltest.main()
