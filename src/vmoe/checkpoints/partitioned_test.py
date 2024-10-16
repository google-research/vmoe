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

"""Tests for partitioned.py."""
import io
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import core
from jax.experimental import pjit
import jax.numpy as jnp
import numpy as np
from vmoe.checkpoints import partitioned

Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding
PartitionSpec = jax.sharding.PartitionSpec
Slice = partitioned.Slice
SliceNd = partitioned.SliceNd


class CreateLocalBuffersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # Array is replicated in two devices of the current process. The output
      # buffers contain a single SliceNd index, covering the full array.
      ('_replicated', {0: (slice(None),), 1: (slice(None),)}, (10,),
       jnp.float32, {SliceNd(Slice(0, 10)): np.zeros((10,), dtype=np.float32)}),
      # The process contains two devices, each storing a shard of the original
      # array. The shards are 0:2 and 6:8, thus those are the keys of the output
      # dictionary.
      ('_sharded', {0: (slice(0, 2),), 1: (slice(6, 8),)}, (2,), jnp.int32,
       {SliceNd(Slice(0, 2)): np.zeros((2,), dtype=np.int32),
        SliceNd(Slice(6, 8)): np.zeros((2,), dtype=np.int32)}),
  )
  def test(self, indices_map, shard_shape, dtype, expected):
    global_shape = (10,)
    sharding = mock.create_autospec(partitioned.Sharding, instance=True)
    sharding.addressable_devices_indices_map.return_value = indices_map
    sharding.shard_shape.return_value = shard_shape
    buffers = partitioned._create_local_buffers(sharding, global_shape, dtype)
    chex.assert_trees_all_close(buffers, expected)

  def test_scalar(self):
    sharding = mock.create_autospec(partitioned.Sharding, instance=True)
    sharding.addressable_devices_indices_map.return_value = {0: (), 1: ()}
    sharding.shard_shape.return_value = ()
    buffers = partitioned._create_local_buffers(sharding, (), np.int32)
    expected = {SliceNd(): np.zeros((), dtype=np.int32)}
    chex.assert_trees_all_close(buffers, expected)


class CreateMapSliceNdToArrayShardsTest(parameterized.TestCase):

  @classmethod
  def _make_shard(cls, index):
    shard = mock.create_autospec(partitioned.Shard, instance=True)
    shard.index = index
    return shard

  @parameterized.named_parameters(
      ('_max_chunk_bytes_none', [(slice(None), slice(None))], None,
       {SliceNd(Slice(0, 4), Slice(0, 10)): [0]}),
  )
  def test(self, global_shards_indexes, max_chunk_bytes, expected):

    global_shards = [self._make_shard(index) for index in global_shards_indexes]
    arr = mock.create_autospec(jax.Array, global_shards=global_shards,
                               shape=(4, 10), dtype=jnp.float32.dtype)
    output = partitioned._create_map_slicend_to_array_shards(
        arr, max_chunk_bytes)
    expected = {
        key: [global_shards[i] for i in values]
        for key, values in expected.items()
    }
    self.assertDictEqual(output, expected)

  def test_scalar(self):
    global_shards = [self._make_shard(())]
    arr = mock.create_autospec(jax.Array, global_shards=global_shards,
                               shape=(), dtype=jnp.float32.dtype)
    output = partitioned._create_map_slicend_to_array_shards(arr)
    expected = {SliceNd(): global_shards}
    self.assertDictEqual(output, expected)


class CreateMapSliceNdToCheckpointShardTest(absltest.TestCase):

  @classmethod
  def _make_shard(cls, process_index):
    device = mock.create_autospec(jax.Device, instance=True)
    device.process_index = process_index
    shard = mock.create_autospec(partitioned.Shard, instance=True)
    shard.device = device
    return shard

  @mock.patch.object(partitioned.jax, 'process_count', return_value=2)
  def test_single_shard_per_process(self, _):
    slicend_to_shards = {
        SliceNd(Slice(0, 10),): [self._make_shard(0), self._make_shard(1)],
        SliceNd(Slice(10, 15),): [self._make_shard(0), self._make_shard(1)],
    }
    bytes_per_shard = [1, 4]
    slicend_to_ckpt_shard = partitioned._create_map_slicend_to_checkpoint_shard(
        slicend_to_shards, 2, bytes_per_shard)
    self.assertListEqual(bytes_per_shard, [21, 14])
    self.assertDictEqual(slicend_to_ckpt_shard, {
        SliceNd(Slice(0, 10),): 0,
        SliceNd(Slice(10, 15),): 1,
    })

  @mock.patch.object(partitioned.jax, 'process_count', return_value=1)
  def test_multiple_shard_per_process(self, _):
    slicend_to_shards = {
        SliceNd(Slice(0, 10),): [self._make_shard(0)],
        SliceNd(Slice(10, 15),): [self._make_shard(0)],
        SliceNd(Slice(15, 20),): [self._make_shard(0)],
        SliceNd(Slice(20, 30),): [self._make_shard(0)],
    }
    bytes_per_shard = [1, 2, 3]
    slicend_to_ckpt_shard = partitioned._create_map_slicend_to_checkpoint_shard(
        slicend_to_shards, 2, bytes_per_shard)
    self.assertListEqual(bytes_per_shard, [21, 32, 13])
    self.assertDictEqual(slicend_to_ckpt_shard, {
        SliceNd(Slice(0, 10),): 0,
        SliceNd(Slice(10, 15),): 1,
        SliceNd(Slice(15, 20),): 2,
        SliceNd(Slice(20, 30),): 1,
    })

  @mock.patch.object(partitioned.jax, 'process_count', return_value=2)
  def test_scalar(self, _):
    slicend_to_shards = {SliceNd(): [self._make_shard(0), self._make_shard(1)]}
    bytes_per_shard = [1, 4, 2, 3]
    slicend_to_ckpt_shard = partitioned._create_map_slicend_to_checkpoint_shard(
        slicend_to_shards, 2, bytes_per_shard)
    self.assertListEqual(bytes_per_shard, [3, 4, 2, 3])
    self.assertDictEqual(slicend_to_ckpt_shard, {SliceNd(): 0})


class FindCkptShardsToRestoreTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # Process with two devices, but a replicated array with a single chunk in
      # the checkpoint with ID equal to 3.
      ('_replicated_array_single_chunk', {0: (slice(None),), 1: (slice(None),)},
       (10,), [SliceNd(Slice(0, 10))], [3], {3}),
      # Process with a single device, handling the shard with items [0:5]. The
      # checkpoint contains 5 shards [0:2, 2:4, 4:6, 6:8, 8:10]. So, the process
      # must restore values from 3 checkpoint shards: 1, 2, and 3.
      ('_sharded_array_multiple_chunks', {0: (slice(0, 5),)}, (5,), [
          SliceNd(Slice(0, 2)),
          SliceNd(Slice(2, 4)),
          SliceNd(Slice(4, 6)),
          SliceNd(Slice(6, 8)),
          SliceNd(Slice(8, 10)),
      ], [1, 2, 3, 4, 5], {1, 2, 3}),
  )
  def test(self, indices_map, shard_shape, ckpt_slices, ckpt_shards, expected):
    global_shape = (10,)
    sharding = mock.create_autospec(partitioned.Sharding, instance=True)
    sharding.addressable_devices_indices_map.return_value = indices_map
    sharding.shard_shape.return_value = shard_shape
    ckpt_shards = partitioned._find_ckpt_shards_to_restore(
        sharding, global_shape, ckpt_slices, ckpt_shards)
    self.assertSetEqual(set(ckpt_shards), expected)

  def test_scalar(self):
    # Process with two devices, and a replicated scalar, stored in a single
    # checkpoint shard (with id equal to 1).
    sharding = mock.create_autospec(partitioned.Sharding, instance=True)
    sharding.addressable_devices_indices_map.return_value = {0: (), 1: ()}
    sharding.shard_shape.return_value = ()
    ckpt_shards = partitioned._find_ckpt_shards_to_restore(
        sharding, global_shape=(), ckpt_slices=[SliceNd()], ckpt_shards=[1])
    self.assertSetEqual(set(ckpt_shards), {1})


class RestoreCheckpointTest(absltest.TestCase):

  def _index(self, version):
    index = {
        'shard_count': 1,
        'index': {
            'x': partitioned.IndexInfo(
                global_shape=core.ShapedArray(shape=(5,), dtype=jnp.float32),
                global_slices=[SliceNd(Slice(0, 5))],
                shards=[0]),
            'y': partitioned.IndexInfo(
                global_shape=core.ShapedArray(shape=(), dtype=jnp.int32),
                global_slices=[SliceNd()],
                shards=[0]),
        }
    }
    if version is not None:
      index['version'] = version.value
    return index

  def _write_checkpoint(self, prefix, version):
    # Write index.
    partitioned.base.save_checkpoint(prefix + '.index', self._index(version))
    # Write data.
    x = np.arange(5, dtype=np.float32)
    y = np.ones((), dtype=np.int32)
    s_x = SliceNd(Slice(0, 5))
    s_y = SliceNd()
    data = partitioned.LazyArrayChunks(chunks={
        0: [(x, s_x, s_x)],
        1: [(y, s_y, s_y)],
    })
    partitioned.base.save_checkpoint(prefix + '.data-00000-of-00001', data)

  def test_unknown(self):
    prefix = self.create_tempfile().full_path
    self._write_checkpoint(prefix, partitioned.Version.UNKNOWN)
    restored = partitioned.restore_checkpoint(
        prefix, {
            'x': np.zeros((5,), dtype=np.float32),
            'y': np.zeros((), dtype=np.int32),
        },
        max_concurrent_bytes=None)
    chex.assert_trees_all_close(restored, {
        'x': np.arange(5, dtype=np.float32),
        'y': np.ones((), dtype=np.int32),
    })

  def test_unknown_no_tree_no_sharding_fails(self):
    # This fails: with the UNKNOWN version checkpoints we must specify either
    # tree or sharding (or both).
    prefix = self.create_tempfile().full_path
    self._write_checkpoint(prefix, partitioned.Version.UNKNOWN)
    with self.assertRaises(ValueError):
      partitioned.restore_checkpoint(prefix, tree=None)

  def test_v1(self):
    prefix = self.create_tempfile().full_path
    self._write_checkpoint(prefix, partitioned.Version.V1)
    restored = partitioned.restore_checkpoint(
        prefix, {
            'x': np.zeros((5,), dtype=np.float32),
            'y': np.zeros((), dtype=np.int32),
        })
    chex.assert_trees_all_close(restored, {
        'x': np.arange(5, dtype=np.float32),
        'y': np.ones((), dtype=np.int32),
    })

  def test_v1_no_tree_no_sharding_works(self):
    # This works fine. With the V1 version, when no tree or sharding is given,
    # the state dict is returned and the data is fully replicated.
    prefix = self.create_tempfile().full_path
    self._write_checkpoint(prefix, partitioned.Version.V1)
    restored = partitioned.restore_checkpoint(prefix, tree=None)
    chex.assert_trees_all_close(restored, {
        'x': np.arange(5, dtype=np.float32),
        'y': np.ones((), dtype=np.int32),
    })

  def test_v1_struct_does_not_match(self):
    # This fails since the structure of the given tree does not match with that
    # of the index.
    prefix = self.create_tempfile().full_path
    self._write_checkpoint(prefix, partitioned.Version.V1)
    with self.assertRaises(ValueError):
      partitioned.restore_checkpoint(prefix, {'y': np.zeros((1,))})


class RestoreLocalBuffersTest(absltest.TestCase):
  # This tests a process restoring parts of the local buffers of 4 arrays from
  # a single checkpoint shard file.
  # The process holds different shards for each of the arrays. For the first
  # array, it holds [4:8, 0:2]; for the second it holds shards [0:2] and [2:4];
  # for the third one it holds the shard [2:6]; the fourth array is a scalar.
  # The checkpoint shard stores none of the chunks of the first array, and it
  # stores the chunks [1:3] for the second array (value=1), the chunks [2:4]
  # and [4:8] of the third array (values=2 and 3 respectively), and the scalar.

  @classmethod
  def _create_array_chunks(cls):
    return partitioned.ArrayChunks(
        chunks={1: [np.ones((2,), dtype=np.int32)],
                2: [np.full((2,), 2, dtype=np.int64),
                    np.full((4,), 3, dtype=np.float32)],
                3: [9 * np.ones((), dtype=np.float64)]},
        global_slices={1: [SliceNd(Slice(1, 3))],
                       2: [SliceNd(Slice(2, 4)), SliceNd(Slice(4, 8))],
                       3: [SliceNd()]})

  @mock.patch.object(partitioned.base, 'restore_checkpoint')
  def test(self, mock_restore_checkpoint):
    mock_restore_checkpoint.return_value = self._create_array_chunks()
    local_buffers = [
        {SliceNd(Slice(4, 8), Slice(0, 2)): np.zeros((4, 2), dtype=np.float32)},
        {
            SliceNd(Slice(0, 2)): np.zeros((2,), dtype=np.int32),
            SliceNd(Slice(2, 4)): np.zeros((2,), dtype=np.int32),
        },
        {SliceNd(Slice(2, 6)): np.zeros((4,), dtype=np.float32)},
        {SliceNd(): np.zeros((), dtype=np.float64)},
    ]
    partitioned._restore_local_buffers('/foo/bar', local_buffers)
    expected_local_buffers = [
        {SliceNd(Slice(4, 8), Slice(0, 2)): np.zeros((4, 2), dtype=np.float32)},
        {
            SliceNd(Slice(0, 2)): np.asarray([0, 1], dtype=np.int32),
            SliceNd(Slice(2, 4)): np.asarray([1, 0], dtype=np.int32),
        },
        {SliceNd(Slice(2, 6)): np.asarray([2, 2, 3, 3], dtype=np.float32)},
        {SliceNd(): np.asarray(9, dtype=np.float64)},
    ]
    chex.assert_trees_all_close(local_buffers, expected_local_buffers)


class SaveCheckpointTest(parameterized.TestCase):

  @classmethod
  def _create_tree(cls, axis_resources):

    def fn():
      d = jax.device_count()
      return {
          'x': jnp.arange(5 * d * 10).reshape((5, d, 10)),
          'y': jnp.arange(4),
          'z': jnp.asarray(9, dtype=jnp.int32),
      }

    with Mesh(np.asarray(jax.devices()), 'd'):
      return pjit.pjit(fn, in_shardings=(), out_shardings=axis_resources)()

  @classmethod
  def _read_index(cls, filepath):
    with io.open(filepath, 'rb') as fp:
      return partitioned.serialization.msgpack_restore(fp.read())

  def test_no_data_sharding_one_checkpoint_shard(self):
    prefix = self.create_tempfile().full_path
    tree = self._create_tree(axis_resources={
        'x': PartitionSpec(),
        'y': PartitionSpec(),
        'z': PartitionSpec(),
    })
    async_result = partitioned.save_checkpoint(prefix, tree, num_shards=1)
    created_files = async_result.get()
    expected_files = [prefix + '.index', prefix + '.data-00000-of-00001']
    self.assertSetEqual(set(created_files), set(expected_files))
    for fpath in created_files:
      self.assertTrue(os.path.exists(fpath))
    index = self._read_index(prefix + '.index')
    self.assertEqual(index['shard_count'], 1)
    self.assertLen(index['index']['x'].shards, 1)
    self.assertLen(index['index']['y'].shards, 1)
    self.assertLen(index['index']['z'].shards, 1)

  def test_data_sharding_two_checkpoint_shards(self):
    prefix = self.create_tempfile().full_path
    tree = self._create_tree(axis_resources={
        'x': PartitionSpec(None, 'd', None),
        'y': PartitionSpec(),
        'z': PartitionSpec(),
    })
    async_result = partitioned.save_checkpoint(prefix, tree, num_shards=2)
    created_files = async_result.get()
    expected_files = [
        prefix + '.index',
        prefix + '.data-00000-of-00002', prefix + '.data-00001-of-00002']
    self.assertSetEqual(set(created_files), set(expected_files))
    for fpath in created_files:
      self.assertTrue(os.path.exists(fpath))
    index = self._read_index(prefix + '.index')
    self.assertEqual(index['shard_count'], 2)
    self.assertLen(index['index']['x'].shards, jax.device_count())
    self.assertLen(index['index']['y'].shards, 1)
    self.assertLen(index['index']['z'].shards, 1)

  def test_no_jax_array_leave_raises(self):
    prefix = self.create_tempfile().full_path
    with self.assertRaises(ValueError):
      _ = partitioned.save_checkpoint(prefix, {'x': np.asarray((5,))}).get()


class SaveAndRestoreTest(absltest.TestCase):
  """Tests save_checkpoint and restore_checkpoint functions end-to-end."""

  @classmethod
  def _create_tree(cls, axis_resources, dtype=jnp.float32):

    def fn():
      d = jax.device_count()
      x = jnp.arange(5 * 2 * d * 10, dtype=dtype).reshape((5, 2 * d, 10))
      y = jnp.asarray(9, dtype=jnp.int32)
      return {'x': x, 'y': y}

    with Mesh(np.asarray(jax.devices()), 'd'):
      return pjit.pjit(
          fn,
          in_shardings=(),
          out_shardings={'x': axis_resources, 'y': PartitionSpec()},
      )()

  @classmethod
  def _create_sharding(cls, axis_resources):
    spec = NamedSharding(
        mesh=Mesh(np.asarray(jax.devices()), 'd'), spec=axis_resources)
    return {'x': spec, 'y': PartitionSpec()}

  def test_both_replicated(self):
    prefix = self.create_tempfile().full_path
    tree = self._create_tree(PartitionSpec())
    _ = partitioned.save_checkpoint(
        prefix, tree, num_shards=5, max_chunk_bytes=10).get()
    restored_tree = partitioned.restore_checkpoint(prefix, None)
    chex.assert_trees_all_close(restored_tree, tree)

  def test_save_replicated_restore_sharded(self):
    prefix = self.create_tempfile().full_path
    tree = self._create_tree(PartitionSpec())
    _ = partitioned.save_checkpoint(prefix, tree).get()
    restored_tree = partitioned.restore_checkpoint(prefix, None)
    expected_tree = self._create_tree(PartitionSpec(None, 'd', None))
    chex.assert_trees_all_close(restored_tree, expected_tree)

  def test_save_sharded_restore_replicated(self):
    prefix = self.create_tempfile().full_path
    tree = self._create_tree(PartitionSpec(None, 'd', None))
    _ = partitioned.save_checkpoint(prefix, tree).get()
    restored_tree = partitioned.restore_checkpoint(prefix, None)
    expected_tree = self._create_tree(PartitionSpec())
    chex.assert_trees_all_close(restored_tree, expected_tree)

  def test_save_and_restore_bfloat16(self):
    prefix = self.create_tempfile().full_path
    tree = self._create_tree(PartitionSpec(), dtype=jnp.bfloat16)
    _ = partitioned.save_checkpoint(prefix, tree).get()
    restored_tree = partitioned.restore_checkpoint(prefix, None)
    chex.assert_trees_all_equal_dtypes(restored_tree, tree)
    chex.assert_trees_all_close(restored_tree, tree)

  def test_save_and_restore_structure(self):
    prefix = self.create_tempfile().full_path
    tree = self._create_tree(PartitionSpec(), dtype=jnp.bfloat16)
    _ = partitioned.save_checkpoint(prefix, tree).get()
    with Mesh(np.asarray(jax.devices()), 'd'):
      tree_zeros = pjit.pjit(
          fun=lambda tree: jax.tree_util.tree_map(jnp.zeros_like, tree),
          out_shardings=PartitionSpec(),
      )(tree)
    restored_tree = partitioned.restore_checkpoint(prefix, tree_zeros)
    chex.assert_trees_all_close(restored_tree, tree)


class SplitSliceNdTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_scalar', (), None, [()]),
      ('_primes', ((0, 5),), 3, [((0, 1),), ((1, 2),), ((2, 3),), ((3, 4),),
                                 ((4, 5),)]),
      ('_max_chunk_items_none', ((0, 2), (4, 8)), None, [((0, 2), (4, 8))]),
      ('_max_chunk_items_4', ((0, 2), (4, 8)), 4, [((0, 1), (4, 8)),
                                                   ((1, 2), (4, 8))]),
      ('_max_chunk_items_2', ((0, 2), (4, 8)), 2, [((0, 1), (4, 6)),
                                                   ((0, 1), (6, 8)),
                                                   ((1, 2), (4, 6)),
                                                   ((1, 2), (6, 8))]),
      ('_max_chunk_items_1', ((0, 2), (4, 8)), 1, [((0, 1), (4, 5)),
                                                   ((0, 1), (5, 6)),
                                                   ((0, 1), (6, 7)),
                                                   ((0, 1), (7, 8)),
                                                   ((1, 2), (4, 5)),
                                                   ((1, 2), (5, 6)),
                                                   ((1, 2), (6, 7)),
                                                   ((1, 2), (7, 8))]),
  )
  def test(self, slicend, max_chunk_items, expected):
    tup2slice = lambda tup: Slice(*tup)
    slicend = SliceNd(*map(tup2slice, slicend))
    expected = [SliceNd(*map(tup2slice, s))  for s in expected]
    output = partitioned._split_slicend(slicend, max_chunk_items)
    self.assertSetEqual(set(output), set(expected))


class UpdateLazyArrayChunksWithLeafTest(absltest.TestCase):

  @classmethod
  def _make_shard(cls, process_index, index):
    device = mock.create_autospec(jax.Device, instance=True)
    device.process_index = process_index
    shard = mock.create_autospec(partitioned.Shard, instance=True)
    shard.device = device
    shard.index = index
    shard.data = None  # The value is not used, but shard.data is accessed.
    return shard

  def test_good(self):
    ckpt_shard_to_lazy_array_chunks = {
        0: mock.create_autospec(partitioned.LazyArrayChunks, instance=True),
        1: mock.create_autospec(partitioned.LazyArrayChunks, instance=True),
    }
    slicend_to_ckpt_shard = {
        SliceNd(Slice(10, 20),): 0,
        SliceNd(Slice(20, 30),): 1,
    }
    slicend_to_arr_shards = {
        SliceNd(Slice(10, 20),): [self._make_shard(0, (slice(10, 30),))],
        SliceNd(Slice(20, 30),): [self._make_shard(0, (slice(10, 30),))],
    }
    partitioned._update_lazy_array_chunks_with_leaf(
        3, slicend_to_arr_shards, slicend_to_ckpt_shard,
        ckpt_shard_to_lazy_array_chunks)
    ckpt_shard_to_lazy_array_chunks[0].add.assert_called_once_with(
        3, None, SliceNd(Slice(0, 10)), SliceNd(Slice(10, 20)))
    ckpt_shard_to_lazy_array_chunks[1].add.assert_called_once_with(
        3, None, SliceNd(Slice(10, 20)), SliceNd(Slice(20, 30)))

  def test_raises(self):
    ckpt_shard_to_lazy_array_chunks = {
        0: mock.create_autospec(partitioned.LazyArrayChunks, instance=True),
    }
    slicend_to_ckpt_shard = {SliceNd(Slice(10, 20),): 0}
    slicend_to_arr_shards = {
        SliceNd(Slice(10, 20),): [self._make_shard(1, (slice(10, 20),))],
    }
    with self.assertRaises(ValueError):
      partitioned._update_lazy_array_chunks_with_leaf(
          3, slicend_to_arr_shards, slicend_to_ckpt_shard,
          ckpt_shard_to_lazy_array_chunks)


if __name__ == '__main__':
  absltest.main()
