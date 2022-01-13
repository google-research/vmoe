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

"""Tests for partitioning."""
import functools
import itertools
import logging
import re
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from vmoe import partitioning

PartitionSpec = partitioning.PartitionSpec


class PartitioningTest(parameterized.TestCase):

  @parameterized.parameters((0, True), (1, False), (2, True))
  def test_process_has_contiguous_device_slice(self, process_index, expected):
    def mk_dev(process_index):
      return _make_device(process_index=process_index)
    devices = np.asarray([
        [mk_dev(0), mk_dev(0), mk_dev(1)],
        [mk_dev(0), mk_dev(0), mk_dev(2)],
        [mk_dev(0), mk_dev(0), mk_dev(1)],
    ])
    self.assertEqual(
        partitioning.process_has_contiguous_device_slice(
            devices, process_index), expected)

  @parameterized.named_parameters(
      ('false', [[0, 0, 1], [0, 0, 2], [0, 0, 1]], False),
      ('true', [[0, 0, 0], [0, 0, 0], [1, 1, 1]], True),
  )
  def test_processes_have_contiguous_device_slices(self, devices, expected):
    def mk_dev(process_index):
      return _make_device(process_index=process_index)
    devices = np.asarray(devices)
    devices = np.vectorize(mk_dev, otypes=[np.object])(devices)
    self.assertEqual(
        partitioning.processes_have_contiguous_device_slices(devices),
        expected)

  @parameterized.parameters(('other'), ('tpu'))
  def test_get_auto_logical_mesh(self, platform):
    """Tests that the right auto_logical_mesh is run, based on the platform."""
    hardware_mesh = mock.MagicMock()
    device = _make_device(platform=platform)
    with mock.patch.object(
        partitioning,
        f'get_hardware_mesh_{platform}',
        return_value=hardware_mesh):
      with mock.patch.object(
          partitioning, f'get_auto_logical_mesh_{platform}') as mock_get:
        partitioning.get_auto_logical_mesh(2, [device])
        mock_get.assert_called_with(2, hardware_mesh)

  @parameterized.named_parameters(
      ('2', 2, (2, 1)),
      ('4', 4, (4, 1)),
      ('8', 8, (4, 2)),
  )
  @mock.patch.object(partitioning, 'get_logical_mesh')
  def test_get_auto_logical_mesh_other(self, num_partitions, expected_tuple,
                                       get_logical_mesh_mock):
    """Tests that each axis is partitioned as expected on devices != TPU."""
    hardware_mesh = np.empty((4, 8))
    partitioning.get_auto_logical_mesh_other(num_partitions, hardware_mesh)
    get_logical_mesh_mock.assert_called_with(expected_tuple, hardware_mesh)

  def test_get_auto_logical_mesh_other_error(self):
    """Tests that an exception is raised if the number of partitions is not supported."""
    hardware_mesh = np.empty((3, 5))
    with self.assertRaisesRegex(ValueError, 'The hardware mesh with shape'):
      partitioning.get_auto_logical_mesh_other(2, hardware_mesh)

  @parameterized.named_parameters(
      ('v3_2', 2, (2, 2, 4, 1), (1, 2, 1, 1)),
      ('v3_4', 4, (2, 2, 4, 1), (1, 2, 2, 1)),
      ('v3_8', 8, (2, 2, 4, 1), (1, 2, 4, 1)),
      ('v3_16', 16, (2, 2, 4, 1), (2, 2, 4, 1)),
      ('v4_2', 2, (2, 2, 4, 2), (1, 1, 1, 2)),
      ('v4_4', 4, (2, 2, 4, 2), (1, 1, 2, 2)),
      ('v4_8', 8, (2, 2, 4, 2), (1, 1, 4, 2)),
      ('v4_16', 16, (2, 2, 4, 2), (1, 2, 4, 2)),
      ('v4_32', 32, (2, 2, 4, 2), (2, 2, 4, 2)),
  )
  @mock.patch.object(partitioning, 'get_logical_mesh')
  def test_get_auto_logical_mesh_tpu(self, num_partitions, hardware_mesh_shape,
                                     expected_tuple, get_logical_mesh_mock):
    """Tests that each axis is partitioned as expected on TPU devices."""
    hardware_mesh = np.empty(hardware_mesh_shape)
    partitioning.get_auto_logical_mesh_tpu(num_partitions, hardware_mesh)
    get_logical_mesh_mock.assert_called_with(expected_tuple, hardware_mesh)

  def test_get_auto_logical_mesh_tpu_error(self):
    """Tests that an exception is raised if the number of partitions is not supported."""
    hardware_mesh = np.empty((3, 5, 7, 9))
    with self.assertRaisesRegex(ValueError, 'The hardware mesh with shape'):
      partitioning.get_auto_logical_mesh_tpu(6, hardware_mesh)

  @parameterized.named_parameters(
      ('cpu0', (0, 0), (0, 0)),
      ('cpu1', (23, 5), (3, 5)),
  )
  @mock.patch.object(partitioning.jax, 'local_device_count', return_value=4)
  def test_get_device_coords_other(self, device_attrs, expected_coord, _):
    """Tests that the device coordinates are good for devices other than TPU."""
    device_id, process_id = device_attrs
    device = _make_device(
        id=device_id, process_index=process_id, platform='cpu')
    self.assertTupleEqual(
        partitioning.get_device_coords_other(device), expected_coord)

  @parameterized.named_parameters(
      ('tpu0', (0, 0, 0, 0)),
      ('tpu1', (0, 1, 2, 3)),
  )
  def test_get_device_coords_tpu(self, expected_coord):
    """Tests that the device coordinates are good for TPU devices."""
    core_on_chip, x, y, z = expected_coord
    device = _make_device(
        core_on_chip=core_on_chip, coords=(x, y, z), platform='tpu')
    self.assertTupleEqual(
        partitioning.get_device_coords_tpu(device), expected_coord)

  def test_get_hardware_mesh_local_shape(self):
    local_devices = [
        # Local devices presented in arbitrary order.
        _make_device(core_on_chip=0, coords=(2, 2, 0), platform='tpu'),
        _make_device(core_on_chip=0, coords=(2, 3, 0), platform='tpu'),
        _make_device(core_on_chip=0, coords=(3, 2, 0), platform='tpu'),
        _make_device(core_on_chip=0, coords=(3, 1, 0), platform='tpu'),
        _make_device(core_on_chip=0, coords=(3, 3, 0), platform='tpu'),
        _make_device(core_on_chip=0, coords=(2, 1, 0), platform='tpu'),
    ]
    shape = partitioning.get_hardware_mesh_local_shape(local_devices)
    expected_shape = (1, 2, 3, 1)
    self.assertEqual(shape, expected_shape)

  @mock.patch.object(partitioning.jax, 'local_device_count', return_value=2)
  def test_get_hardware_mesh_other(self, _):
    """Tests the hardware mesh (with 6 total CPU devices in 2 processes)."""
    devices = []
    for process_index in range(3):
      for device_id in range(process_index * 2, process_index * 2 + 2):
        devices.append(
            _make_device(
                id=device_id, process_index=process_index, platform='cpu'))
    hardware_mesh = partitioning.get_hardware_mesh_other(devices)
    expected_hardware_mesh = np.array([[devices[0], devices[2], devices[4]],
                                       [devices[1], devices[3], devices[5]]])
    np.testing.assert_array_equal(hardware_mesh, expected_hardware_mesh)

  def test_get_hardware_mesh_tpu(self):
    """Tests the hardware mesh (with 12 TPU devices, in a (2, 3, 1, 2) mesh)."""
    devices = []
    for z, y, x, core_on_chip in itertools.product(
        range(2), range(3), range(1), range(2)):
      devices.append(
          _make_device(
              core_on_chip=core_on_chip, coords=(x, y, z), platform='tpu'))
    hardware_mesh = partitioning.get_hardware_mesh_tpu(devices)
    expected_hardware_mesh = np.array([
        # core_on_chip=0.
        [[[devices[0], devices[6]],
          [devices[2], devices[8]],
          [devices[4], devices[10]]]],
        # core_on_chip=1.
        [[[devices[1], devices[7]],
          [devices[3], devices[9]],
          [devices[5], devices[11]]]]
    ], dtype=np.object)
    np.testing.assert_array_equal(hardware_mesh, expected_hardware_mesh)

  def test_get_logical_mesh_default(self):
    """Tests the logical mesh with a 2x4 hardware mesh."""
    # Note: The values in hardware_mesh would typically be Devices, but these
    # are fine for testing. This is a 2x4 hardware mesh.
    hardware_mesh = np.array([[1, 2, 3, 4],   # partition_ids: 0 0 1 1
                              [5, 6, 7, 8]])  #                2 2 3 3
    partitions, replicas = (2, 2), (1, 2)
    mesh = partitioning.get_logical_mesh_default(
        partitions, replicas, hardware_mesh)
    self.assertIsInstance(mesh, partitioning.maps.Mesh)
    np.testing.assert_array_equal(mesh.devices,
                                  [[1, 2], [3, 4], [5, 6], [7, 8]])
    self.assertTupleEqual(mesh.axis_names, ('expert', 'replica'))

  def test_get_logical_mesh_tile_by_process(self):
    # Note: The values in hardware_mesh would typically be Devices, but these
    # are fine for testing. This is a 2x4 hardware mesh.
    # partition_ids: 0 0 1 1 | process_ids: 0 1 2 3
    #                2 2 3 3 |              0 1 2 3
    hardware_mesh = np.asarray([[1, 2, 3, 4],
                                [5, 6, 7, 8]])
    partitions, replicas = (2, 2), (1, 2)
    hardware_mesh_local_shape = (2, 1)
    mesh = partitioning.get_logical_mesh_tile_by_process(
        partitions, replicas, hardware_mesh, hardware_mesh_local_shape)
    self.assertIsInstance(mesh, partitioning.maps.Mesh)
    np.testing.assert_array_equal(mesh.devices,
                                  [[1, 2], [5, 6], [3, 4], [7, 8]])
    self.assertTupleEqual(mesh.axis_names, ('expert', 'replica'))

  def test_get_logical_mesh_tile_by_process_raises(self):
    hardware_mesh = np.zeros((3, 3))
    partitions, replicas = (3, 1), (1, 3)
    hardware_mesh_local_shape = (1, 2)
    with self.assertRaises(ValueError):
      partitioning.get_logical_mesh_tile_by_process(
          partitions, replicas, hardware_mesh, hardware_mesh_local_shape)

  @mock.patch.object(partitioning,
                     'processes_have_contiguous_device_slices',
                     return_value=False)
  @mock.patch.object(partitioning, 'get_hardware_mesh_local_shape')
  def test_get_logical_mesh(self, mock_get_hardware_mesh_local_shape, _):
    # Note: The values in hardware_mesh would typically be Devices, but these
    # are fine for testing. This is a 2x4 hardware mesh.
    # partition_ids: 0 1 2 3 | process_ids: 0 0 2 3
    #                0 1 2 3 |              1 1 2 3
    hardware_mesh = np.asarray([[1, 2, 3, 4],
                                [5, 6, 7, 8]])
    mock_get_hardware_mesh_local_shape.return_value = (2, 1)
    mesh = partitioning.get_logical_mesh((2, 2), hardware_mesh)
    np.testing.assert_array_equal(mesh.devices,
                                  [[1, 2], [5, 6], [3, 4], [7, 8]])

  def test_log_logical_mesh_tpu(self):
    mk_dev = functools.partial(_make_device, platform='tpu')
    devices = [
        [
            mk_dev(core_on_chip=0, coords=(0, 0, 0), process_index=0),
            mk_dev(core_on_chip=1, coords=(0, 0, 0), process_index=1),
            mk_dev(core_on_chip=0, coords=(10, 0, 0), process_index=10),
            mk_dev(core_on_chip=1, coords=(10, 0, 0), process_index=11),
        ],
        [
            mk_dev(core_on_chip=0, coords=(0, 100, 0), process_index=1),
            mk_dev(core_on_chip=1, coords=(0, 100, 0), process_index=2),
            mk_dev(core_on_chip=0, coords=(10, 1, 0), process_index=3),
            mk_dev(core_on_chip=1, coords=(10, 1, 0), process_index=4),
        ],
    ]
    mesh = partitioning.Mesh(devices=np.asarray(devices), axis_names=('a', 'b'))
    logger = logging.getLogger('foo')
    with self.assertLogs(logger) as cm:
      partitioning.log_logical_mesh(mesh, logger=logger)
    self.assertRegex(
        cm.output[0],
        re.escape("Logical device mesh has axis_names = ('a', 'b')"))
    self.assertRegex(
        cm.output[1],
        re.escape('Logical device mesh has shape = (2, 4)'))
    self.assertRegex(cm.output[2], 'Logical device mesh:')
    self.assertRegex(cm.output[3], '\\+[-]+\\+')
    # pylint: disable=line-too-long
    self.assertRegex(
        cm.output[4],
        re.escape('| (0,  0,   0, 0)[ 0] (1,  0,   0, 0)[ 1] (0, 10,   0, 0)[10] (1, 10,   0, 0)[11] |'))
    self.assertRegex(
        cm.output[5],
        re.escape('| (0,  0, 100, 0)[ 1] (1,  0, 100, 0)[ 2] (0, 10,   1, 0)[ 3] (1, 10,   1, 0)[ 4] |'))
    # pylint: enable=line-too-long
    self.assertRegex(cm.output[6], '\\+[-]+\\+')

  @mock.patch.object(jax, 'local_device_count', return_value=4)
  def test_log_logical_mesh_single_axis(self, unused_mock):
    devices = [_make_device(id=0, process_index=0, platform='cpu'),
               _make_device(id=10, process_index=10, platform='cpu')]
    mesh = partitioning.Mesh(devices=np.asarray(devices), axis_names=('a',))
    logger = logging.getLogger('foo')
    with self.assertLogs(logger) as cm:
      partitioning.log_logical_mesh(mesh, logger=logger)
    self.assertRegex(
        cm.output[0], re.escape("Logical device mesh has axis_names = ('a',)"))
    self.assertRegex(
        cm.output[1], re.escape('Logical device mesh has shape = (2,)'))
    self.assertRegex(cm.output[2], 'Logical device mesh:')
    self.assertRegex(cm.output[3], '\\+[-]+\\+')
    self.assertRegex(cm.output[4], re.escape('| (0,  0)[ 0] |'))
    self.assertRegex(cm.output[5], re.escape('| (2, 10)[10] |'))
    self.assertRegex(cm.output[6], '\\+[-]+\\+')

  def test_tree_global_shape(self):
    """Tests that global shape of arrays is obtained correctly."""
    # Note: see _make_tree_axis_resources_mesh_test_data for additional details.
    tree, axis_resources, mesh = _make_tree_axis_resources_mesh_test_data()
    expected_global_aval = {
        'v': jax.ShapedArray(shape=(5, 5), dtype=jnp.float32),
        'w': jax.ShapedArray(shape=(4 * 5, 5), dtype=jnp.float32),
        'x': jax.ShapedArray(shape=(4 * 2 * 5, 5), dtype=jnp.float32),
        'y': jax.ShapedArray(shape=(4 * 5, 2 * 5), dtype=jnp.float32),
        'z': jax.ShapedArray(shape=(4 * 3 * 5, 2 * 5), dtype=jnp.float32),
    }
    global_aval = partitioning.tree_global_shape(tree, axis_resources, mesh)
    self.assertDictEqual(global_aval, expected_global_aval)

  def test_tree_global_shape_raises_structs_not_match(self):
    mesh = partitioning.Mesh(devices=np.zeros((4, 4)), axis_names=('a', 'b'))
    with self.assertRaisesRegex(ValueError, 'The tree structs do not match'):
      partitioning.tree_global_shape({'a': 1, 'b': 2}, {'c': PartitionSpec()},
                                     mesh)

  def test_tree_global_shape_raises_wrong_leaves(self):
    mesh = partitioning.Mesh(devices=np.zeros((4, 4)), axis_names=('a', 'b'))
    with self.assertRaisesRegex(ValueError, 'the input tree must have'):
      partitioning.tree_global_shape({'a': 1}, {'a': PartitionSpec()}, mesh)


def _make_device(**kwargs):
  """Returns a new mocked device."""
  device = mock.MagicMock(partitioning.Device)
  for key, value in kwargs.items():
    setattr(device, key, value)
  return device


def _make_tree_axis_resources_mesh_test_data():
  # Mesh of (4, 3, 2) devices. Each device resides in a different process to
  # simplify the calculation of global shapes of the arrays.
  devices = np.asarray(
      [_make_device(process_index=idx, id=idx) for idx in range(24)],
      dtype=np.object).reshape(4, 3, 2)
  mesh = partitioning.Mesh(devices, axis_names=('a', 'b', 'c'))
  # These shapes are those of the arrays in the process running the code
  # (i.e. process_index=0).
  tree = {
      'v': jax.ShapedArray(shape=(5, 5), dtype=jnp.float32),
      'w': jax.ShapedArray(shape=(5, 5), dtype=jnp.float32),
      'x': jax.ShapedArray(shape=(5, 5), dtype=jnp.float32),
      'y': jax.ShapedArray(shape=(5, 5), dtype=jnp.float32),
      'z': jax.ShapedArray(shape=(5, 5), dtype=jnp.float32),
  }
  axis_resources = {
      # Array 'v' is not partitioned, each device holds a replica of this.
      # Thus, the global shape is (5, 5).
      'v': None,
      # Array 'w' has its first axis partitioned in 4 chunks across the
      # axis 'a' of the logical mesh. Thus, its global shape is (4 * 5, 5).
      'w': PartitionSpec('a'),
      # Array 'x' has its first axis partitioned in 4 * 2 chunks across the
      # axes 'a' and 'c' of the logical mesh. Thus its global shape is
      # (4 * 2 * 5, 5).
      'x': PartitionSpec(('a', 'c'),),
      # Array 'y' has its first axis partitioned in 4 chunks (across logical
      # axis 'a') and the second axis partitioned in 2 chunks (across logical
      # axis 'c'). Thus its global shape is (4 * 5, 2 * 5).
      'y': PartitionSpec('a', 'c'),
      # Array 'z' has its first axis partitioned in 4 * 3 chunks, and the
      # second axis partitioned in 2 chunks. Its global shape is
      # (4 * 3 * 5, 2 * 5).
      'z': PartitionSpec(('a', 'b'), 'c'),
  }
  return tree, axis_resources, mesh


if __name__ == '__main__':
  absltest.main()
