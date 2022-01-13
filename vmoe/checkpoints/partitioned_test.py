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

"""Tests for model checkpoints."""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from vmoe.checkpoints import base
from vmoe.checkpoints import partitioned
from vmoe.checkpoints import types

ArrayChunks = types.ArrayChunks
Device = jax.xla.Device
Mesh = partitioned.Mesh
PartitionSpec = partitioned.PartitionSpec
Slice = partitioned.Slice
SliceNd = partitioned.SliceNd
SliceNdArray = partitioned.SliceNdArray


class MakeSliceNdArrayTest(absltest.TestCase):
  """Tests the function creating SliceNdArrays from a mesh and ShapedArrays."""

  def test_make_slice_nd_arrays(self):
    # (4, 2) mesh with 4 processes, each handling two devices.
    devices = np.asarray([
        [_make_device(process_index=0, id=0),
         _make_device(process_index=2, id=4)],
        [_make_device(process_index=0, id=1),
         _make_device(process_index=2, id=5)],
        [_make_device(process_index=1, id=2),
         _make_device(process_index=3, id=6)],
        [_make_device(process_index=1, id=3),
         _make_device(process_index=3, id=7)],
    ], dtype=np.object)
    mesh = Mesh(devices, ('a', 'b'))
    aval = jax.ShapedArray((16, 8, 3), dtype=jnp.float32)
    partition_spec = partitioned.ParsedPartitionSpec.from_user_input(
        PartitionSpec('a', 'b'), 'input')
    slice_nd_arrays = partitioned._make_slice_nd_arrays(
        [aval], [partition_spec], mesh)
    expected_slice_nd_array = SliceNdArray.create([
        SliceNd(Slice(0, 4), Slice(0, 4), Slice(0, 3)),
        SliceNd(Slice(0, 4), Slice(4, 8), Slice(0, 3)),
        SliceNd(Slice(4, 8), Slice(0, 4), Slice(0, 3)),
        SliceNd(Slice(4, 8), Slice(4, 8), Slice(0, 3)),
        SliceNd(Slice(8, 12), Slice(0, 4), Slice(0, 3)),
        SliceNd(Slice(8, 12), Slice(4, 8), Slice(0, 3)),
        SliceNd(Slice(12, 16), Slice(0, 4), Slice(0, 3)),
        SliceNd(Slice(12, 16), Slice(4, 8), Slice(0, 3)),
    ], shape=(4, 2))
    self.assertLen(slice_nd_arrays, 1)
    np.testing.assert_array_equal(slice_nd_arrays[0], expected_slice_nd_array)


class MatchCheckpointToLocalSlices(absltest.TestCase):

  def test_match_checkpoint_to_local_slices(self):
    local_global_slices = [
        (SliceNd(Slice(0, 4)), SliceNd(Slice(4, 8))),
        (SliceNd(Slice(4, 8)), SliceNd(Slice(0, 4))),
    ]
    ckpt_slices_and_shards = [
        (SliceNd(Slice(6, 12)), 1),
        (SliceNd(Slice(0, 6)), 2),
    ]
    output = list(
        partitioned._match_checkpoint_to_local_slices(local_global_slices,
                                                      ckpt_slices_and_shards))
    expected_output = [
        (1, SliceNd(Slice(6, 12)), SliceNd(Slice(0, 2)), SliceNd(Slice(2, 4))),
        (2, SliceNd(Slice(0, 6)), SliceNd(Slice(4, 6)), SliceNd(Slice(0, 2))),
        (2, SliceNd(Slice(0, 6)), SliceNd(Slice(0, 4)), SliceNd(Slice(4, 8))),
    ]
    self.assertCountEqual(expected_output, output)

  def test_match_checkpoint_to_local_slices_raises(self):
    local_global_slices = [(SliceNd(Slice(0, 4)), SliceNd(Slice(4, 8)))]
    ckpt_slices_and_shards = []
    with self.assertRaises(ValueError):
      _ = list(
          partitioned._match_checkpoint_to_local_slices(local_global_slices,
                                                        ckpt_slices_and_shards))


class PairLocalAndGlobalSlicesTest(parameterized.TestCase):
  """Tests the function pairing local SliceNds and global SliceNds.

  A (local/global) SliceNdArray is an array of SliceNd objects denoting which
  chunk of a particular array each device in the (local/global) mesh holds.
  """

  def _make_partitioned_across_process_data(self):  # pylint: disable=g-unreachable-test-method
    # 2x2 mesh, with two processes handling two devices each.
    # devices | processes
    # [0, 1]  |  [0, 0]
    # [2, 3]  |  [1, 1]
    devices = [
        _make_device(process_index=0, id=0),
        _make_device(process_index=0, id=1),
        _make_device(process_index=1, id=2),
        _make_device(process_index=1, id=3),
    ]
    mesh = Mesh(np.asarray(devices).reshape(2, 2), ('a', 'b'))
    # The global shape of the data is (8, ?), which is chunked in 2 partitions,
    # each one is handled by a different process. The two devices of a given
    # process store the same data, thus the local shape is (4, ?).
    global_slices_array = SliceNdArray.create(
        [SliceNd(Slice(0, 4), Slice()),
         SliceNd(Slice(0, 4), Slice()),
         SliceNd(Slice(4, 8), Slice()),
         SliceNd(Slice(4, 8), Slice())],
        shape=(2, 2))
    local_slices_array = SliceNdArray.create(
        [SliceNd(Slice(0, 4), Slice()),
         SliceNd(Slice(0, 4), Slice())],
        shape=(1, 2))
    return mesh, local_slices_array, global_slices_array

  def _make_partitioned_within_process_data(self):  # pylint: disable=g-unreachable-test-method
    # 2x2 mesh, with two processes handling two devices each.
    # devices | processes
    # [0, 2]  |  [0, 1]
    # [1, 3]  |  [0, 1]
    devices = [
        _make_device(process_index=0, id=0),
        _make_device(process_index=1, id=2),
        _make_device(process_index=0, id=1),
        _make_device(process_index=1, id=3),
    ]
    mesh = Mesh(np.asarray(devices).reshape(2, 2), ('a', 'b'))
    # The global shape of the data is (8, ?), which is chunked in 2 partitions,
    # each one is handled by a different device within the same process.
    # The two processes actually hold the same data. Thus, they have local shape
    # of (8, ?).
    global_slices_array = SliceNdArray.create(
        [SliceNd(Slice(0, 4), Slice()),
         SliceNd(Slice(0, 4), Slice()),
         SliceNd(Slice(4, 8), Slice()),
         SliceNd(Slice(4, 8), Slice())],
        shape=(2, 2))
    local_slices_array = SliceNdArray.create(
        [SliceNd(Slice(0, 4), Slice()),
         SliceNd(Slice(4, 8), Slice())],
        shape=(2, 1))
    return mesh, local_slices_array, global_slices_array

  @parameterized.named_parameters(
      ('process_0_across_process', 0, '_make_partitioned_across_process_data',
       {(SliceNd(Slice(0, 4), Slice()), SliceNd(Slice(0, 4), Slice()))}),
      ('process_1_across_process', 1, '_make_partitioned_across_process_data',
       {(SliceNd(Slice(0, 4), Slice()), SliceNd(Slice(4, 8), Slice()))}),
      ('process_0_within_process', 0, '_make_partitioned_within_process_data',
       {(SliceNd(Slice(0, 4), Slice()), SliceNd(Slice(0, 4), Slice())),
        (SliceNd(Slice(4, 8), Slice()), SliceNd(Slice(4, 8), Slice()))}),
      ('process_1_within_process', 1, '_make_partitioned_within_process_data',
       {(SliceNd(Slice(0, 4), Slice()), SliceNd(Slice(0, 4), Slice())),
        (SliceNd(Slice(4, 8), Slice()), SliceNd(Slice(4, 8), Slice()))}),
  )
  def test_pair_local_and_global_slices(self, process_index, make_data,
                                        expected_pairs):

    mesh, local_slices_array, global_slices_array = getattr(self, make_data)()
    with mock.patch.object(
        jax._src.lib.xla_bridge, 'process_index', return_value=process_index):
      pairs = list(partitioned._pair_local_and_global_slices(
          [local_slices_array], [global_slices_array], mesh, local_mesh=None))
    self.assertLen(pairs, 1)
    self.assertEqual(pairs[0], expected_pairs)


class RemoveUnusedShardsTest(absltest.TestCase):
  """Tests that shards that don't handle any slice are removed."""

  def test_remove_unused_shards(self):
    shards_per_slice = [[1, 3], [2]]
    process_per_shard = [5, 4, 3, 2, 1]
    output = partitioned._remove_unused_shards(shards_per_slice,
                                               process_per_shard)
    expected_shards_per_slice = [[0, 1], [2]]
    expected_process_per_shard = (4, 2, 3)
    self.assertEqual(output[0], expected_shards_per_slice)
    self.assertTupleEqual(output[1], expected_process_per_shard)


class RestoreArrayChunks(parameterized.TestCase):

  def test_restore_array_chunks(self):
    array_chunks = types.ArrayChunks(
        chunks={
            0: [1 * np.ones((5, 4)), 2 * np.ones((4, 5))],
            1: [3 * np.ones((6,)), 4 * np.ones((3,))],
            2: [np.arange(6).reshape((3, 2))],
        },
        global_slices={
            0: [SliceNd(Slice(0, 5), Slice(4, 8)),
                SliceNd(Slice(10, 14), Slice(0, 5))],
            1: [SliceNd(Slice(0, 6)), SliceNd(Slice(6, 9))],
            2: [SliceNd(Slice(7, 10), Slice(4, 6))],
        })
    array_slices_to_restore = {
        1: [(SliceNd(Slice(0, 6)), SliceNd(Slice(1, 3)), SliceNd(Slice(0, 2))),
            (SliceNd(Slice(6, 9)), SliceNd(Slice(0, 2)), SliceNd(Slice(3, 5)))],
        2: [(SliceNd(Slice(7, 10), Slice(4, 6)),
             SliceNd(Slice(0, 3), Slice(0, 1)),
             SliceNd(Slice(0, 3), Slice(2, 3))),
            (SliceNd(Slice(7, 10), Slice(4, 6)),
             SliceNd(Slice(1, 3), Slice(0, 2)),
             SliceNd(Slice(2, 4), Slice(0, 2)))],
    }
    local_arrays = [None, np.zeros((8,)), np.zeros((4, 4))]
    with mock.patch.object(
        base, 'restore_checkpoint', return_value=array_chunks):
      partitioned._restore_array_chunks('foo', local_arrays,
                                        array_slices_to_restore)
    np.testing.assert_array_almost_equal(
        local_arrays[1],
        [3, 3, 0, 4, 4, 0, 0, 0])
    np.testing.assert_array_almost_equal(
        local_arrays[2],
        [[0, 0, 0, 0],
         [0, 0, 2, 0],
         [2, 3, 4, 0],
         [4, 5, 0, 0]])

  @parameterized.parameters(
      # Checkpoint has slice [0:5], process holds global slice [5:10] in a
      # local slice [0:5] of an array.
      # Checkpoint and global slices do not intersect.
      (SliceNd(Slice(0, 5)), SliceNd(Slice(5, 10)), SliceNd(Slice(0, 5)), None),
      # Checkpoint has slice [3:8], process holds global slice [3:8] in a
      # local slice [0:5] of an array.
      # Checkpoint chunk[0:5] must be copied to local array[0:5].
      (SliceNd(Slice(3, 8)), SliceNd(Slice(3, 8)), SliceNd(Slice(0, 5)),
       (SliceNd(Slice(0, 5)), SliceNd(Slice(0, 5)))),
      # Checkpoint has slice [2:5, 0:4], process holds global slice [1:4, 1:3]
      # in local slice [4:7, 4:6] of an array.
      # Checkpoint chunk[0:2, 1:3] must be copied to local array[5:7, 4:6].
      (SliceNd(Slice(2, 5), Slice(0, 4)), SliceNd(Slice(1, 4), Slice(1, 3)),
       SliceNd(Slice(4, 7), Slice(4, 6)),
       (SliceNd(Slice(0, 2), Slice(1, 3)),
        SliceNd(Slice(5, 7), Slice(4, 6)))),
  )
  def test_intersect_slice_nd(self, ckpt_slice_nd, global_slice_nd,
                              local_slice_nd, expected_output):
    output = partitioned._intersect_slice_nd(ckpt_slice_nd, global_slice_nd,
                                             local_slice_nd)
    self.assertEqual(output, expected_output)


class RestoreAndSaveCheckpointTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('process_0_of_2', 0, None,
       [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3]]),
      ('process_1_of_2', 1, None,
       [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3]]),
      ('process_0_of_2_axis_resources',
       0, {'x': None, 'y': None, 'z': PartitionSpec('a')},
       [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]),
      ('process_1_of_2_axis_resources',
       1, {'x': None, 'y': None, 'z': PartitionSpec('a')},
       [[2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 3]]),
  )
  @mock.patch.object(partitioned.jax, 'process_count', return_value=2)
  def test_restore_checkpoint(self, process_index, axis_resources, expected_z,
                              _):
    devices = np.asarray(
        [_make_device(process_index=i // 2, id=i) for i in range(4)])
    mesh = partitioned.Mesh(devices.reshape((2, 2)), ('a', 'b'))
    prefix = self.create_tempfile().full_path
    def side_effect(filepath, *unused_args, **unused_kwargs):
      return {
          prefix + '.index': self._get_expected_index(),
          prefix + '.data-00000-of-00004': self._get_expected_shard_content(0),
          prefix + '.data-00001-of-00004': self._get_expected_shard_content(1),
          prefix + '.data-00002-of-00004': self._get_expected_shard_content(2),
          prefix + '.data-00003-of-00004': self._get_expected_shard_content(3),
      }[filepath]
    with mock.patch.object(partitioned.vmoe.checkpoints.base,
                           'restore_checkpoint', side_effect=side_effect):
      with mock.patch.object(jax, 'process_index', return_value=process_index):
        with mock.patch.object(jax._src.lib.xla_bridge, 'process_index',
                               return_value=process_index):
          restored = partitioned.restore_checkpoint(
              prefix=prefix,
              tree=None,
              axis_resources=axis_resources,
              mesh=mesh)
          np.testing.assert_array_almost_equal(
              restored['x'], np.zeros((5, 5), dtype=np.float32))
          np.testing.assert_array_almost_equal(
              restored['y'], [[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [2, 2, 2, 2, 2],
                              [2, 2, 2, 2, 2],
                              [2, 2, 2, 2, 2],
                              [2, 2, 2, 2, 2],
                              [2, 2, 2, 2, 2]])
          np.testing.assert_array_almost_equal(restored['z'], expected_z)

  def test_restore_checkpoint_empty_mesh(self):
    prefix = self.create_tempfile().full_path
    with self.assertRaisesRegex(ValueError, 'You must pass a non-empty mesh'):
      partitioned.restore_checkpoint(
          prefix=prefix, tree=None, axis_resources=None)

  @parameterized.named_parameters(
      ('process_0', 0, 2, 0),
      ('process_1', 1, 1, 2),
      ('process_2', 2, 1, 1),
      ('process_3', 3, 1, 3),
  )
  @mock.patch.object(partitioned.jax, 'process_count', return_value=4)
  @mock.patch.object(
      partitioned.vmoe.multihost_utils, 'sync_devices', return_value=None)
  def test_save_checkpoint(self, process_index, num_written_files, shard,
                           unused_1, unused_2):
    devices = np.asarray(
        [_make_device(process_index=i, id=i) for i in range(4)]).reshape((2, 2))
    mesh = partitioned.Mesh(devices, ('a', 'b'))
    tree = {
        # 'x' has global_shape = (5, 5), it's written by process 0 in shard 0.
        'x': np.ones((5, 5), dtype=np.float32) * process_index,
        # 'y' has global_shape = (10, 5), first half is written by process 0
        # (shard 0), second half is written by process 2 (shard 1).
        'y': np.ones((5, 5), dtype=np.float32) * process_index,
        # 'z' has global_shape = (10, 10), first quarter is written by process 0
        # (shard 0), second quarter is written by process 1 (shard 2),
        # third quarter is written by process 2 (shard 1), fourth quarter is
        # written by process 3 (shard 3).
        'z': np.ones((5, 5), dtype=np.float32) * process_index,
    }
    axis_resources = {
        'x': None,
        'y': PartitionSpec('a'),
        'z': PartitionSpec('a', 'b'),
    }
    prefix = self.create_tempfile().full_path
    # Note: we need to mock both jax.process_index AND
    # jax._src.lib.process_index.
    with mock.patch.object(jax, 'process_index', return_value=process_index):
      with mock.patch.object(jax._src.lib.xla_bridge, 'process_index',
                             return_value=process_index):
        async_result = partitioned.save_checkpoint(
            prefix=prefix, tree=tree, axis_resources=axis_resources, mesh=mesh)
    written_files = async_result.get()
    # Check that the process writes the expected number of files.
    self.assertLen(written_files, num_written_files)
    # If the process writes the index, load the index and check its icontent.
    if num_written_files == 2:
      index_content = base.restore_checkpoint(prefix + '.index')
      expected_index_content = self._get_expected_index()
      chex.assert_trees_all_equal_comparator(
          lambda x, y: x == y,
          lambda x, y: f'IndexInfos do not match:\n{x}\n{y}',
          index_content, expected_index_content)
      # Check that the process has written the expected sharded file.
      expected_ckpt_shard = prefix + f'.data-{shard:05d}-of-00004'
      self.assertIn(expected_ckpt_shard, written_files)
      array_chunks = base.restore_checkpoint(expected_ckpt_shard)
      expected_array_chunks = self._get_expected_shard_content(shard)
      chex.assert_trees_all_equal_comparator(
          self._compare_array_chunks,
          lambda x, y: f'ArrayChunks do not match:\n{x}\n{y}',
          array_chunks, expected_array_chunks,
      )

  def test_save_checkpoint_empty_mesh(self):
    prefix = self.create_tempfile().full_path
    with self.assertRaisesRegex(ValueError, 'You must pass a non-empty mesh'):
      partitioned.save_checkpoint(prefix=prefix, tree=mock.MagicMock(),
                                  axis_resources=mock.MagicMock())

  def _compare_array_chunks(self, a, b):
    """Compares two ArrayChunks objects."""
    if a.global_slices != b.global_slices:
      return False
    a, sa = jax.tree_flatten(dict(a.chunks))
    b, sb = jax.tree_flatten(dict(b.chunks))
    if sa != sb:
      return False
    return all(map(lambda x, y: (x == y).all, a, b))

  def _get_expected_index(self):
    return {
        'shard_count': 4,
        'index': {
            'x': partitioned.IndexInfo(
                global_shape=jax.ShapedArray((5, 5), dtype=jnp.float32),
                global_slices=((Slice(0, 5), Slice(0, 5)),),
                shards=(0,)),
            'y': partitioned.IndexInfo(
                global_shape=jax.ShapedArray((10, 5), dtype=jnp.float32),
                global_slices=((Slice(0, 5), Slice(0, 5)),
                               (Slice(5, 10), Slice(0, 5))),
                shards=(0, 1)),
            'z': partitioned.IndexInfo(
                global_shape=jax.ShapedArray((10, 10), dtype=jnp.float32),
                global_slices=((Slice(0, 5), Slice(0, 5)),
                               (Slice(0, 5), Slice(5, 10)),
                               (Slice(5, 10), Slice(0, 5)),
                               (Slice(5, 10), Slice(5, 10))),
                shards=(0, 2, 1, 3)),
        },
    }

  def _get_expected_shard_content(self, shard):
    """Returns the ArrayChunks data stored in each shard."""
    return {
        # shard 0 is written by process 0.
        0: ArrayChunks(
            chunks={
                0: [np.zeros((5, 5), dtype=np.float32)],  # x[:, :]
                1: [np.zeros((5, 5), dtype=np.float32)],  # y[0:5, :]
                2: [np.zeros((5, 5), dtype=np.float32)],  # z[0:5, 0:5]
            },
            global_slices={
                0: [(Slice(0, 5), Slice(0, 5))],
                1: [(Slice(0, 5), Slice(0, 5))],
                2: [(Slice(0, 5), Slice(0, 5))],

            }),
        # shard 1 is written by process 2.
        1: ArrayChunks(
            chunks={
                1: [2 * np.ones((5, 5), dtype=np.float32)],  # y[5:10, :]
                2: [2 * np.ones((5, 5), dtype=np.float32)],  # z[5:10, 0:5]
            },
            global_slices={
                1: [(Slice(5, 10), Slice(0, 5))],
                2: [(Slice(5, 10), Slice(0, 5))],
            }),
        # shard 2 is written by process 1.
        2: ArrayChunks(
            chunks={
                2: [np.ones((5, 5), dtype=np.float32)],  # z[0:5, 5:10]
            },
            global_slices={
                2: [(Slice(0, 5), Slice(5, 10))],
            }),
        # shard 2 is written by process 3.
        3: ArrayChunks(
            chunks={
                2: [3 * np.ones((5, 5), dtype=np.float32)],  # z[5:10, 5:10]
            },
            global_slices={
                2: [(Slice(5, 10), Slice(5, 10))],
            }),
    }[shard]


class SliceNdArraysToShardsTest(absltest.TestCase):

  def _create_test_data(self):
    # PyTree used in several tests.
    return [
        # Array 'x' has two axis, none of which is partitioned.
        SliceNdArray.create([SliceNd(Slice(), Slice())] * 6, shape=(3, 2)),
        # Array 'y' is also not partitioned, but only has one axis.
        SliceNdArray.create([SliceNd(Slice(),)] * 6, shape=(3, 2)),
        # Array 'z' is partitioned on its second axis, across the second logical
        # axis in two chunks.
        SliceNdArray.create(
            [
                SliceNd(Slice(None), Slice(0, 3)),  # Processes {0, 1, 2}.
                SliceNd(Slice(None), Slice(3, 6)),  # Processes {3, 4, 5}.
            ],
            shape=(1, 2),
            tile=(3, 1)),
    ]

  @mock.patch.object(partitioned.jax, 'process_count', return_value=6)
  def test_slice_nd_arrays_to_shards(self, _):
    # Assume there's only one device per process to simplify calculations.
    devices = np.asarray([
        [_make_device(process_index=0), _make_device(process_index=3)],
        [_make_device(process_index=1), _make_device(process_index=4)],
        [_make_device(process_index=2), _make_device(process_index=5)],
    ])
    output = partitioned._slice_nd_arrays_to_shards(
        self._create_test_data(), devices, num_shards=6)
    expected_shards_per_slice = [[0], [1], [2, 3]]
    self.assertEqual(output[0], expected_shards_per_slice)
    self.assertTupleEqual(output[1], (0, 1, 2, 3, 4, 5))

  @mock.patch.object(partitioned.jax, 'process_count', return_value=6)
  def test_slice_nd_arrays_to_shards_minimum(self, _):
    # Assume there's only one device per process to simplify calculations.
    # Notice that the process_indices are not contiguous. This affects the slice
    # that each process handles (for additional info, check the 'z' array in
    # _create_slice_axes_array_to_shards_test_data).
    devices = np.asarray([
        [_make_device(process_index=0), _make_device(process_index=3)],
        [_make_device(process_index=1), _make_device(process_index=4)],
        [_make_device(process_index=2), _make_device(process_index=5)],
    ])
    devices = devices.reshape(3, 2)
    output = partitioned._slice_nd_arrays_to_shards(
        self._create_test_data(), devices, num_shards=0)
    expected_shards_per_slice = [
        [0],     # Process 0.
        [0],     # Process 0.
        [0, 1],  # Process {0, 3}.
    ]
    self.assertEqual(output[0], expected_shards_per_slice)
    self.assertTupleEqual(output[1], (0, 3))


def _make_device(**kwargs):
  """Returns a new mocked device."""
  device = mock.MagicMock(Device)
  for key, value in kwargs.items():
    setattr(device, key, value)
  return device


if __name__ == '__main__':
  absltest.main()
