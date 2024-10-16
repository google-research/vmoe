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

import io
import json
import os
from unittest import mock
# Trick to simulate 4 devices without the need of TPUs for testing.
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

# pylint: disable=g-import-not-at-top
from absl.testing import absltest
from absl.testing import parameterized
import chex
import flax.core
import jax
from jax.experimental import pjit
import numpy as np
from vmoe.initialization import initialization
# pylint: enable=g-import-not-at-top

Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding
PartitionSpec = jax.sharding.PartitionSpec


def assert_trees_all_equivalent_sharding(*trees):

  def cmp_fn(a, b):
    device_map_a = a.sharding.devices_indices_map(a.shape)
    device_map_b = b.sharding.devices_indices_map(b.shape)
    return device_map_a == device_map_b

  def err_msg_fn(a, b):
    return f'{a.sharding=!r} is not equivalent to {b.sharding=!r}'

  chex.assert_trees_all_equal_comparator(cmp_fn, err_msg_fn, *trees)


class _BaseInitializeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt = flax.core.FrozenDict({
        'foo': {
            'a': np.arange(12, dtype=np.float64).reshape((4, 3)),
            'b': np.arange(16, dtype=np.float32).reshape((4, 4)),
            'c': np.ones((3, 3), dtype=np.float32),
        },
    })
    mesh = Mesh(jax.devices(), ('d',))
    self.target = flax.core.FrozenDict({
        'foo': {
            'A': jax.ShapeDtypeStruct(
                (4, 3), jax.numpy.float32,
                sharding=NamedSharding(mesh, PartitionSpec())),
            'B': jax.ShapeDtypeStruct(
                (8, 4, 4), jax.numpy.float32,
                sharding=NamedSharding(mesh, PartitionSpec('d'))),
            'C': jax.ShapeDtypeStruct(
                (4, 3), jax.numpy.float32,
                sharding=NamedSharding(mesh, PartitionSpec('d'))),
        },
    })
    self.mesh = mesh
    self.rules = [
        ('a', 'A'),
        ('b', 'B', 'expand_tile', 0, 8),  # (4, 4) -> (8, 4, 4)
        ('c', None),                      # Do not restore from the checkpoint.
    ]
    # Expected output values in the expected devices after initialization.
    def to_device(x, s):
      with self.mesh:
        return pjit.pjit(lambda x: x, in_shardings=None, out_shardings=s)(x)
    self.expected = flax.core.FrozenDict({
        'foo': {
            'A': to_device(self.ckpt['foo']['a'],
                           self.target['foo']['A'].sharding),
            'B': to_device(np.stack([self.ckpt['foo']['b']] * 8, axis=0),
                           self.target['foo']['B'].sharding),
            # Note: This is not an array, this is a jax.ShapeDtypeStruct.
            'C': self.target['foo']['C'],
        }
    })


class InitializeFromOrbaxTest(_BaseInitializeTest):

  def setUp(self):
    super().setUp()
    # Create checkpoint directory structure.
    self.directory = self.create_tempdir().full_path
    for name in ('a', 'b', 'c'):
      zarray = self.create_tempfile(
          os.path.join(self.directory, 'foo', name, '.zarray')).full_path
      with io.open(zarray, 'wt') as fp:
        json.dump({'shape': self.ckpt['foo'][name].shape,
                   'dtype': self.ckpt['foo'][name].dtype.str}, fp)
    # Mock the AsyncCheckpointer class.
    self.mock_async_checkpointer = self.enter_context(
        mock.patch.object(initialization, 'AsyncCheckpointerWithStructure'))
    # Mock the structure() method.
    self.mock_structure = self.mock_async_checkpointer.return_value.structure
    self.mock_structure.return_value = {
        'foo': {
            name: 'PLACEHOLDER://' + os.path.join(self.directory, 'foo', name)
            for name in ('a', 'b', 'c')
        }
    }
    # Mock the restore() method.
    self.mock_restore = self.mock_async_checkpointer.return_value.restore
    self.mock_restore.return_value = self.ckpt

  @parameterized.named_parameters(
      ('_no_axis_resources_regexes', None),
      ('_axis_resources_regexes', [('a', ('d',)), ('b', ('d',)), ('c', ())]),
  )
  def test(self, axis_resources_regexes):
    output = initialization.initialize_from_orbax(
        target=self.target, directory=self.directory, mesh=self.mesh,
        rules=self.rules, axis_resources_regexes=axis_resources_regexes,
        raise_if_target_unmatched=False)
    chex.assert_trees_all_equal_structs(output, self.expected)
    chex.assert_trees_all_equal_shapes(output, self.expected)
    # Regardless of the sharding used when copying the checkpoint params to
    # devices, the output sharding has to be equivalent to the one specified
    # in the output tree.
    assert_trees_all_equivalent_sharding(output, self.expected)
    np.testing.assert_allclose(output['foo']['A'], self.expected['foo']['A'])
    np.testing.assert_allclose(output['foo']['B'], self.expected['foo']['B'])
    # Test that restore is called only once with the right args and kwargs.
    self.mock_restore.assert_called_once()
    restore_args = self.mock_restore.call_args_list[0].args
    restore_kwargs = self.mock_restore.call_args_list[0].kwargs
    self.assertEqual(restore_args, (self.directory,))
    self.assertIn('restore_args', restore_kwargs)
    self.assertTrue(all([
        isinstance(v, initialization.orbax_checkpoint.ArrayRestoreArgs)
        for v in jax.tree_util.tree_leaves(restore_kwargs)]))


class InitializeFromVitTest(_BaseInitializeTest):
  """Tests initialization from open sourced ViT checkpoints."""

  def setUp(self):
    super().setUp()
    self.mock_load = self.enter_context(
        mock.patch.object(initialization.vit_jax_checkpoint, 'load'))
    self.mock_load.return_value = self.ckpt

  @parameterized.named_parameters(
      ('_no_axis_resources_regexes', None),
      ('_axis_resources_regexes', [('a', ('d',)), ('b', ('d',)), ('c', ())]),
  )
  def test(self, axis_resources_regexes):
    output = initialization.initialize_from_vit(
        target=self.target, filepath='/foo/bar', mesh=self.mesh,
        rules=self.rules, axis_resources_regexes=axis_resources_regexes,
        raise_if_target_unmatched=False)
    chex.assert_trees_all_equal_structs(output, self.expected)
    chex.assert_trees_all_equal_shapes(output, self.expected)
    # Regardless of the sharding used when copying the checkpoint params to
    # devices, the output sharding has to be equivalent to the one specified
    # in the output tree.
    assert_trees_all_equivalent_sharding(output, self.expected)
    np.testing.assert_allclose(output['foo']['A'], self.expected['foo']['A'])
    np.testing.assert_allclose(output['foo']['B'], self.expected['foo']['B'])


class _BaseInitializeFromVmoeTest(_BaseInitializeTest):

  def setUp(self):
    super().setUp()
    # Mocked index with unknown version.
    def index_info(shape):
      return mock.MagicMock(
          global_shape=jax.ShapeDtypeStruct(shape, jax.numpy.float32))
    self.index = {
        'index': {
            'foo': {
                'a': index_info((4, 3)),
                'b': index_info((4, 4)),
                'c': index_info((3, 3)),
            }
        }
    }
    # V-MoE's restore_checkpoint returns jax.Array devices. Put the ckpt
    # arrays in the devices, but using a different sharding than the target.
    axis_resources = flax.core.FrozenDict({
        'foo': {
            'a': PartitionSpec('d'),
            'b': PartitionSpec('d'),
            'c': PartitionSpec(),
        }
    })
    with self.mesh:
      self.ckpt = pjit.pjit(
          lambda x: x, in_shardings=None, out_shardings=axis_resources
      )(self.ckpt)


class InitializeFromVmoeV1Test(_BaseInitializeFromVmoeTest):
  """Tests initialization from new V-MoE checkpoints."""

  def setUp(self):
    super().setUp()
    # Set V1 to the index.
    self.index['version'] = (
        initialization.vmoe_checkpoint.Version.V1)
    # Mock call to restore the index.
    self.mock_index_restore = self.enter_context(
        mock.patch.object(initialization.vmoe_checkpoint, 'restore_checkpoint'))
    self.mock_index_restore.return_value = self.index
    # Mock call to restore the checkpoint content.
    self.mock_ckpt_restore = self.enter_context(
        mock.patch.object(initialization.vmoe_checkpoint,
                          'restore_checkpoint_partitioned'))
    self.mock_ckpt_restore.return_value = self.ckpt

  def test(self):
    output = initialization.initialize_from_vmoe(
        target=self.target, prefix='/foo/bar', rules=self.rules,
        mesh=self.mesh, raise_if_target_unmatched=False)
    chex.assert_trees_all_equal_structs(output, self.expected)
    chex.assert_trees_all_equal_shapes(output, self.expected)
    # Regardless of the sharding used when copying the checkpoint params to
    # devices, the output sharding has to be equivalent to the one specified
    # in the output tree.
    assert_trees_all_equivalent_sharding(output, self.expected)
    np.testing.assert_allclose(output['foo']['A'], self.expected['foo']['A'])
    np.testing.assert_allclose(output['foo']['B'], self.expected['foo']['B'])


class InitializeFromVmoeUnknownTest(_BaseInitializeFromVmoeTest):
  """Tests initialization from V-MoE checkpoints of an unknown version."""

  def setUp(self):
    super().setUp()
    # Rename ckpt names: when initializing from old V-MoE checkpoints, the
    # structure of the ckpt and target trees must to match.
    self.index['index']['foo'] = {
        'A': self.index['index']['foo']['a'],
        'B': self.index['index']['foo']['b'],
        'C': self.index['index']['foo']['c'],
    }
    self.ckpt = self.ckpt.copy({'foo': {
        'A': self.ckpt['foo']['a'],
        'B': self.ckpt['foo']['b'],
        'C': self.ckpt['foo']['c'],
    }})
    self.rules = [('A', 'A'), ('B', 'B', 'expand_tile', 0, 8), ('C', None)]
    # The call to restore the index is mocked in each test individually, since
    # it's different in each case. Here we only mock the call to restore the
    # checkpoint contents.
    self.mock_ckpt_restore = self.enter_context(
        mock.patch.object(initialization.vmoe_checkpoint,
                          'restore_checkpoint_partitioned'))
    self.mock_ckpt_restore.return_value = self.ckpt

  @mock.patch.object(
      initialization.vmoe_checkpoint, 'restore_checkpoint')
  def test(self, mock_base_restore):
    mock_base_restore.return_value = self.index
    output = initialization.initialize_from_vmoe(
        target=self.target, prefix='/foo/bar', rules=self.rules,
        mesh=self.mesh, raise_if_target_unmatched=False)
    chex.assert_trees_all_equal_structs(output, self.expected)
    chex.assert_trees_all_equal_shapes(output, self.expected)
    # Regardless of the sharding used when copying the checkpoint params to
    # devices, the output sharding has to be equivalent to the one specified
    # in the output tree.
    assert_trees_all_equivalent_sharding(output, self.expected)
    np.testing.assert_allclose(output['foo']['A'], self.expected['foo']['A'])
    np.testing.assert_allclose(output['foo']['B'], self.expected['foo']['B'])

  @mock.patch.object(
      initialization.vmoe_checkpoint, 'restore_checkpoint')
  def test_raises(self, mock_base_restore):
    """Tests that an exception is raised if the target structure doesn't match that in the checkpoint."""
    mock_base_restore.return_value = {'index': (self.index['index']['foo'],)}
    with self.assertRaises(ValueError):
      initialization.initialize_from_vmoe(
          target=self.target, prefix='/foo/bar', rules=self.rules,
          mesh=self.mesh)


del _BaseInitializeTest
del _BaseInitializeFromVmoeTest

if __name__ == '__main__':
  absltest.main()
