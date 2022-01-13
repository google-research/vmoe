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

"""Tests for serialization."""

from absl.testing import absltest
import jax
import numpy as np
from vmoe.checkpoints import serialization

ArrayChunks = serialization.ArrayChunks
IndexInfo = serialization.IndexInfo
LazyArrayChunks = serialization.LazyArrayChunks
Slice = serialization.Slice
SliceNd = serialization.SliceNd
SliceNdArray = serialization.SliceNdArray


class SerializationTest(absltest.TestCase):

  def test_pytree_to_from_bytes(self):
    lazy_array_chunks, array_chunks = _make_lazy_and_normal_array_chunks()
    tree = {
        'a': 'foo',
        'b': jax.ShapedArray((5, 3), dtype=jax.numpy.bfloat16),
        'c': np.float32(5.0),
        'd': np.asarray([1, 2, 3]),
        'e': complex(3., 4.),
        'f': Slice(),
        'g': SliceNd(),
        'h': SliceNdArray.create([SliceNd(Slice(), Slice()), SliceNd()]),
        'i': lazy_array_chunks,
        'j': IndexInfo(
            global_shape=jax.ShapedArray((20,), dtype=jax.numpy.float32),
            global_slices=(SliceNd(Slice(2, 5)), SliceNd(Slice(3, 9))),
            shards=(1, 3))
    }
    restored_tree = serialization.from_bytes(tree, serialization.to_bytes(tree))
    self.assertEqual(tree['a'], restored_tree['a'])
    self.assertEqual(tree['b'], restored_tree['b'])
    self.assertEqual(tree['c'], restored_tree['c'])
    np.testing.assert_array_equal(tree['d'], restored_tree['d'])
    self.assertEqual(tree['e'], restored_tree['e'])
    self.assertEqual(tree['f'], restored_tree['f'])
    self.assertEqual(tree['g'], restored_tree['g'])
    np.testing.assert_array_equal(tree['h'], restored_tree['h'])
    # LazyArrayChunks are deserialized as ArrayChunks.
    self.assertIsInstance(restored_tree['i'], ArrayChunks)
    np.testing.assert_array_equal(restored_tree['i'].chunks.get(3),
                                  array_chunks.chunks.get(3))
    np.testing.assert_array_equal(restored_tree['i'].chunks.get(5),
                                  array_chunks.chunks.get(5))
    self.assertEqual(restored_tree['i'].global_slices.get(3),
                     array_chunks.global_slices.get(3))
    self.assertEqual(restored_tree['i'].global_slices.get(5),
                     array_chunks.global_slices.get(5))
    self.assertEqual(tree['j'], restored_tree['j'])


def _make_lazy_and_normal_array_chunks():
  # When this LazyArrayChunks is serialized, the expected restored object is
  # the next ArrayChunks.
  lazy_array_chunks = LazyArrayChunks()
  lazy_array_chunks.add(3, np.arange(10), SliceNd(Slice(3, 5)),
                        SliceNd(Slice(103, 105)))
  big_size = serialization._MAX_CHUNK_SIZE + 2048
  big_array = np.arange(big_size, dtype=np.int8)
  lazy_array_chunks.add(5, big_array, SliceNd(Slice(0, big_size)),
                        SliceNd(Slice(0, big_size)))
  array_chunks = ArrayChunks()
  array_chunks.add(3, np.asarray([3, 4]), SliceNd(Slice(103, 105)))
  array_chunks.add(5, big_array, SliceNd(Slice(0, big_size)))
  return lazy_array_chunks, array_chunks


if __name__ == '__main__':
  absltest.main()
