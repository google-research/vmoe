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

"""Tests for types."""
import collections.abc

from absl.testing import absltest
import numpy as np
from vmoe.checkpoints import types

ArrayChunks = types.ArrayChunks
LazyArrayChunks = types.LazyArrayChunks
Slice = types.Slice
SliceNd = types.SliceNd
SliceNdArray = types.SliceNdArray


class SliceTest(absltest.TestCase):

  def test_init(self):
    self.assertEqual(Slice().tuple, (None, None, None))
    self.assertEqual(Slice(5).tuple, (None, 5, None))
    self.assertEqual(Slice(1, 5).tuple, (1, 5, None))
    self.assertEqual(Slice(1, 5, 2).tuple, (1, 5, 2))
    self.assertEqual(Slice(slice(1, 3, 1)).tuple, (1, 3, 1))
    self.assertEqual(Slice(Slice(1)).tuple, (None, 1, None))

  def test_slice(self):
    s = Slice(5, 10, 1).slice
    self.assertEqual((s.start, s.stop, s.step), (5, 10, 1))

  def test_comparator(self):
    self.assertEqual(Slice(1, 3), Slice(1, 3, None))
    self.assertNotEqual(Slice(1, 3), Slice(1, 3, 2))
    self.assertLess(Slice(1, 3), Slice(1, 5))
    self.assertGreater(Slice(1, 5), Slice(1, 3))

  def test_hash(self):
    self.assertEqual(hash(Slice()), hash(Slice(None)))
    self.assertEqual(hash(Slice(5)), hash(Slice(5)))
    self.assertNotEqual(hash(Slice(1)), hash(Slice(2)))

  def test_repr(self):
    self.assertEqual(repr(Slice()), 'Slice()')
    self.assertEqual(repr(Slice(None, 5, None)), 'Slice(5)')
    self.assertEqual(repr(Slice(1, 3)), 'Slice(1, 3)')
    self.assertEqual(repr(Slice(1, 3, 1)), 'Slice(1, 3, 1)')


class SliceNdTest(absltest.TestCase):

  def test_new(self):
    a, b, c = Slice(1), Slice(2), Slice(3)
    self.assertEqual(SliceNd(a, b, c), (a, b, c))
    self.assertEqual(SliceNd([a, b, c]), (a, b, c))
    with self.assertRaises(ValueError):
      _ = SliceNd('a', 'b', 'c')

  def test_chunk(self):
    x = np.arange(24).reshape(6, 4)
    y = SliceNd(Slice(2, 4), Slice(1, 3)).chunk(x)
    np.testing.assert_array_equal(y, [[9, 10], [13, 14]])


class SliceNdArrayTest(absltest.TestCase):

  def test_create(self):
    self.assertEqual(SliceNdArray.create([SliceNd(Slice(1))]).shape, (1,))
    self.assertEqual(
        SliceNdArray.create([SliceNd(Slice(1)), SliceNd(Slice(2))],
                            shape=(1, 2)).shape, (1, 2))
    self.assertEqual(
        SliceNdArray.create([SliceNd(Slice(1)), SliceNd(Slice(2))],
                            shape=(1, 2),
                            tile=(3, 1)).shape, (3, 2))


class ArrayChunksTest(absltest.TestCase):

  def test_add(self):
    x = np.ones((5, 4), dtype=np.int32)
    array_chunks = ArrayChunks()
    array_chunks.add(2, x, SliceNd(Slice(3)))
    self.assertLen(array_chunks.chunks, 1)
    np.testing.assert_array_equal(array_chunks.chunks.get(2), [x])
    self.assertEqual(array_chunks.global_slices.get(2), [SliceNd(Slice(3))])

  def test_iter_chunks(self):
    array_chunks = ArrayChunks()
    array_chunks.add(2, np.arange(10), SliceNd(Slice(10, 20)))
    iter_chunks = array_chunks.iter_chunks(2)
    self.assertIsInstance(iter_chunks, collections.abc.Iterator)
    chunks = list(iter_chunks)
    self.assertLen(chunks, 1)
    np.testing.assert_array_equal(chunks[0][0], np.arange(10))
    self.assertEqual(chunks[0][1], SliceNd(Slice(10, 20)))
    # This raises an exception because index 3 does not exist.
    with self.assertRaises(KeyError):
      _ = array_chunks.iter_chunks(3)


class LazyArrayChunksTest(absltest.TestCase):

  def test_add(self):
    x = np.ones((5, 4), dtype=np.int32)
    lazy_array_chunks = LazyArrayChunks()
    lazy_array_chunks.add(2, x, SliceNd(Slice(2), Slice()),
                          SliceNd(Slice(2), Slice(2)))
    lazy_array_chunks.add(2, x, SliceNd(Slice(1), Slice()),
                          SliceNd(Slice(3), Slice(3)))
    self.assertLen(lazy_array_chunks.ndarray, 1)
    np.testing.assert_array_equal(lazy_array_chunks.ndarray.get(2), x)
    self.assertEqual(
        lazy_array_chunks.local_slices.get(2),
        [SliceNd(Slice(2), Slice()), SliceNd(Slice(1), Slice())])
    self.assertEqual(
        lazy_array_chunks.global_slices.get(2),
        [SliceNd(Slice(2), Slice(2)), SliceNd(Slice(3), Slice(3))])
    # This raises an exception because the ndarray assigned to index 2 is not
    # the same as before (although they are equal).
    with self.assertRaises(AssertionError):
      lazy_array_chunks.add(2, np.ones((5, 4)), SliceNd(), SliceNd())

  def test_iter_chunks(self):
    lazy_array_chunks = LazyArrayChunks()
    lazy_array_chunks.add(
        2, np.arange(10), SliceNd(Slice(5)), SliceNd(Slice(1, 6)))
    iter_chunks = lazy_array_chunks.iter_chunks(2)
    self.assertIsInstance(iter_chunks, collections.abc.Iterator)
    chunks = list(iter_chunks)
    self.assertLen(chunks, 1)
    # The arange(10) is sliced dynamically when iterated, and the appropriate
    # chunk of the array is returned.
    np.testing.assert_array_equal(chunks[0][0], [0, 1, 2, 3, 4])
    self.assertEqual(chunks[0][1], SliceNd(Slice(1, 6)))
    # This raises an exception because index 3 does not exist.
    with self.assertRaises(KeyError):
      _ = lazy_array_chunks.iter_chunks(3)


if __name__ == '__main__':
  absltest.main()
