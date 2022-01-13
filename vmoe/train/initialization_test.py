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

"""Tests for initialization."""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import flax.core
import jax
from jax.experimental import maps
from jax.experimental import pjit
import numpy as np
from vmoe.train import initialization


class InitializeFromVmoeReleaseTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_load = self.enter_context(
        mock.patch.object(initialization.vmoe_partitioned_checkpoint,
                          'restore_checkpoint'))
    self.mock_load.return_value = flax.core.FrozenDict({
        'foo': {
            'a': np.ones((4, 3), dtype=np.float64),
            'b': np.ones((4, 4), dtype=np.float64),
            'c': np.ones((3, 3), dtype=np.float64),
            'posembed_input': {
                'pos_embedding': np.ones((1, 4, 3), np.float64),
            },
        },
    })
    self.params = flax.core.FrozenDict({
        'foo': {
            'A': np.zeros((4, 3), dtype=np.float32),
            'B': np.zeros((4, 4), dtype=np.float32),
            'C': np.zeros((3, 3), dtype=np.float32),
            'posembed_input': {
                'pos_embedding': np.zeros((1, 16, 3), np.float32),
            },
        },
    })
    self.axis_resources = jax.tree_map(lambda _: pjit.PartitionSpec(),
                                       self.params)

  def test_success(self):
    with self.assertLogs() as logs:
      with maps.mesh(np.asarray(jax.local_devices()), ('d',)):
        params = initialization.initialize_from_vmoe_release(
            params=self.params,
            axis_resources=self.axis_resources,
            prefix='/foo/bar/checkpoint',
            mapping=[('(foo)/A', r'\1/a'), ('(foo)/B', r'\1/b')],
            keep=['.*/C'])
    np.testing.assert_allclose(params['foo']['A'],
                               np.ones((4, 3), dtype=np.float32))
    np.testing.assert_allclose(params['foo']['B'],
                               np.ones((4, 4), dtype=np.float32))
    np.testing.assert_allclose(params['foo']['C'],
                               np.zeros((3, 3), dtype=np.float32))
    self.assertEqual(
        set(x.dtype for x in jax.tree_leaves(params)),
        {np.dtype('float32')})
    self.assertEqual(
        logs.output[-1],
        'INFO:absl:The following parameters were found in the checkpoint but '
        'not used during initialization:\n\tfoo/c')

  def test_parameter_shape_not_equal_raises(self):
    with self.assertRaisesRegex(ValueError,
                                "Parameter 'foo/B' .* shapes are not equal"):
      initialization.initialize_from_vmoe_release(
          params=flax.core.FrozenDict({
              'foo': {
                  'A': np.zeros((4, 3), dtype=np.float32),
                  'B': np.zeros((8, 4, 4), dtype=np.float32),
                  'C': np.zeros((3, 3), dtype=np.float32),
              },
          }),
          axis_resources=self.axis_resources,
          prefix='/foo/bar/checkpoint',
          mapping=[('(foo)/A', r'\1/a'), ('(foo)/B', r'\1/b')],
          keep=['.*/C'])


class InitializeFromVitTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_load = self.enter_context(
        mock.patch.object(initialization.vit_jax_checkpoint, 'load'))
    self.mock_load.return_value = flax.core.FrozenDict({
        'foo': {
            'a': np.ones((4, 3), dtype=np.float64),
            'b': np.ones((4, 4), dtype=np.float64),
            'c': np.ones((3, 3), dtype=np.float64),
            'posembed_input': {
                'pos_embedding': np.ones((1, 4, 3), np.float64),
            },
        },
    })
    self.params = flax.core.FrozenDict({
        'foo': {
            'A': np.zeros((4, 3), dtype=np.float32),
            'B': np.zeros((8, 4, 4), dtype=np.float32),
            'C': np.zeros((3, 3), dtype=np.float32),
            'posembed_input': {
                'pos_embedding': np.zeros((1, 16, 3), np.float32),
            },
        },
    })
    self.axis_resources = jax.tree_map(lambda _: pjit.PartitionSpec(),
                                       self.params)

  def test_success(self):
    with self.assertLogs() as logs:
      with maps.mesh(np.asarray(jax.local_devices()), ('d',)):
        params = initialization.initialize_from_vit(
            params=self.params,
            axis_resources=self.axis_resources,
            filepath='/foo/bar/checkpoint.npz',
            mapping=[('(foo)/A', r'\1/a'), ('(foo)/B', r'\1/b')],
            keep=['.*/C'],
            broadcast=['.*/B'])
    np.testing.assert_allclose(params['foo']['A'],
                               np.ones((4, 3), dtype=np.float32))
    np.testing.assert_allclose(params['foo']['B'],
                               np.ones((8, 4, 4), dtype=np.float32))
    np.testing.assert_allclose(params['foo']['C'],
                               np.zeros((3, 3), dtype=np.float32))
    self.assertEqual(
        set(x.dtype for x in jax.tree_leaves(params)),
        {np.dtype('float32')})
    self.assertEqual(
        logs.output[-1],
        'INFO:absl:The following parameters were found in the checkpoint but '
        'not used during initialization:\n\tfoo/c')

  def test_parameter_not_found_raises(self):
    with self.assertRaisesRegex(KeyError, "Parameter 'foo/C' .* not found"):
      initialization.initialize_from_vit(
          params=self.params,
          axis_resources=self.axis_resources,
          filepath='/foo/bar/checkpoint.npz',
          mapping=[('(foo)/A', r'\1/a'), ('(foo)/B', r'\1/b')],
          keep=[],
          broadcast=['.*/B'])

  def test_parameter_not_broadcastable_raises(self):
    with self.assertRaisesRegex(ValueError,
                                "Parameter 'foo/B' .* not compatible"):
      initialization.initialize_from_vit(
          params=self.params,
          axis_resources=self.axis_resources,
          filepath='/foo/bar/checkpoint.npz',
          mapping=[('(foo)/A', r'\1/a'), ('(foo)/B', r'\1/b')],
          keep=['.*/C'],
          broadcast=[])


class IsBroadcastableTest(parameterized.TestCase):

  @parameterized.parameters(
      ((4, 4), (4, 4, 4), True),
      ((4, 1, 4), (4, 4, 4), True),
      ((4, 4), (4, 3), False),
  )
  def test(self, shape1, shape2, expected):
    self.assertEqual(initialization._is_broadcastable(shape1, shape2), expected)


class ZoomPositionEmbeddingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_none_class_token', 3, 5, False, False, [[[1., 1.]] * 25]),
      ('_source_class_token', 3, 5, True, False, [[[1., 1.]] * 25]),
      ('_target_class_token', 5, 3, False, True, [[[3., 3.]] + [[1., 1.]] * 9]),
      ('_both_class_token', 5, 3, True, True, [[[2., 2.]] + [[1., 1.]] * 9]),
  )
  def test(self, source_grid, target_grid, source_tok, target_tok, expected):
    source = np.ones((1, source_grid**2, 2))
    target = np.zeros((1, target_grid**2, 2))
    if source_tok:
      source = np.concatenate([2 * np.ones((1, 1, 2)), source], axis=1)
    if target_tok:
      target = np.concatenate([3 * np.ones((1, 1, 2)), target], axis=1)
    output = initialization._zoom_position_embedding(source, target)
    np.testing.assert_allclose(output, expected)

  @parameterized.named_parameters(
      ('wrong_shape_source', (1, 1, 16, 2), (1, 16, 2),
       r'source .* must have shape \(1, \?, \?\)'),
      ('wrong_shape_target', (1, 16, 2), (3, 16, 2),
       r'target .* must have shape \(1, \?, \?\)'),
      ('different_hidden_size', (1, 16, 2), (1, 16, 3),
       'hidden_size of source and target does not match'),
      ('grid_not_squared', (1, 18, 3), (1, 16, 3),
       "18 tokens found, .* grid used is not squared. This isn't supported."),
  )
  def test_raises(self, source_shape, target_shape, regex):
    with self.assertRaisesRegex(ValueError, regex):
      initialization._zoom_position_embedding(np.ones(source_shape),
                                              np.ones(target_shape))


if __name__ == '__main__':
  absltest.main()
