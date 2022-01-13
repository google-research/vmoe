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

"""Classes used during checkpointing which can be serialized.

To see the details of the serialization, check serialization.py.
"""
import collections
import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple, Union

import jax
import numpy as np
import vmoe.utils

Array = Union[jax.numpy.ndarray, np.ndarray]


class Slice:
  """A class mimicking Python's slice() but supporting hashing and comparisons."""

  def __init__(self, *args):
    if not args:
      self._slice = slice(None)
    elif len(args) == 1 and isinstance(args[0], slice):
      self._slice = args[0]
    elif len(args) == 1 and isinstance(args[0], Slice):
      self._slice = args[0]._slice
    else:
      self._slice = slice(*args)

  @property
  def slice(self) -> slice:
    return self._slice

  @property
  def tuple(self):
    return (self._slice.start, self._slice.stop, self._slice.step)

  @property
  def start(self):
    return self._slice.start

  @property
  def step(self):
    return self._slice.step

  @property
  def stop(self):
    return self._slice.stop

  def __eq__(self, other) -> bool:
    other = other if isinstance(other, Slice) else Slice(other)
    return self.tuple == other.tuple

  def __lt__(self, other) -> bool:
    other = other if isinstance(other, Slice) else Slice(other)
    return self.tuple < other.tuple

  def __hash__(self) -> int:
    return hash(self.tuple)

  def __repr__(self) -> str:
    start, stop, step = self.tuple
    if start is None and stop is None and step is None:
      return 'Slice()'
    if start is None and step is None:
      return f'Slice({stop})'
    if step is None:
      return f'Slice({start}, {stop})'
    return f'Slice({start}, {stop}, {step})'


class SliceNd(tuple):
  """A tuple of Slices, used to slice over multiple axes of an array."""

  def __new__(cls, *args) -> 'SliceNd':
    if all(isinstance(a, Slice) for a in args):
      # SliceNd(Slice(...), Slice(...), Slice(...))
      return tuple.__new__(cls, args)
    elif len(args) == 1 and isinstance(args[0], collections.abc.Iterable):
      # SliceNd([Slice(...), Slice(...), Slice(...)])
      return tuple.__new__(cls, map(Slice, args[0]))
    else:
      raise ValueError(f'Unsupported arguments given to SliceNd {args!r}')

  def chunk(self, array: Array) -> Array:
    assert len(self) <= array.ndim, f'{len(self)} vs. {array.ndim}'
    return array[tuple(s.slice for s in self)]


class SliceNdArray(np.ndarray):
  """A Numpy array of SliceNd elements."""

  def __new__(cls, ndarray: np.ndarray) -> 'SliceNdArray':
    assert all(isinstance(x, SliceNd) for x in ndarray.flat)
    return np.asarray(ndarray).view(cls)

  @classmethod
  def create(cls,
             slices: Sequence[SliceNd],
             shape: Optional[Tuple[int, ...]] = None,
             tile: Optional[Tuple[int, ...]] = None) -> 'SliceNdArray':
    """Creates a SliceNdArray from a sequence of SliceNds."""
    array = np.empty(len(slices), dtype=np.object)
    for i, slice_axes in enumerate(slices):
      array[i] = SliceNd(slice_axes)
    if shape:
      array = array.reshape(shape)
    if tile:
      assert len(tile) == array.ndim, f'{len(tile)} vs. {array.ndim}'
      array = np.tile(array, tile)
    return np.asarray(array).view(cls)


def ddlist_field():
  return dataclasses.field(
      default_factory=lambda: collections.defaultdict(list))


@dataclasses.dataclass
class ArrayChunks:
  """Holds chunks of arrays, each array is identified by its ID.

  A "chunk" is the result of slicing and array using some SliceNd object.
  When the context is cleaer, we sometimes refer to the "chunks" as "slices",
  but in many occasions we need to differentiate between a "chunk" (the data)
  and the "slice" (the slices from which that chunk of data was obtained).

  Attributes:
    chunks: Dict mapping IDs to a list of chunks.
    global_slices: Dict mapping IDs to a list of SliceNd with the slices of
      each corresponding chunk.
  """
  chunks: Dict[int, List[Array]] = ddlist_field()
  global_slices: Dict[int, List[SliceNd]] = ddlist_field()

  def add(self, index: int, chunk: Array, global_slice: SliceNd):
    self.chunks[index].append(chunk)
    self.global_slices[index].append(global_slice)

  def iter_chunks(self, index: int):
    chunks = self.chunks.get(index)
    global_slices = self.global_slices.get(index)
    if chunks is None or global_slices is None:
      raise KeyError(f'Index {index} not found.')
    return vmoe.utils.safe_zip(chunks, global_slices)


@dataclasses.dataclass
class LazyArrayChunks:
  """A lazy version of ArrayChunks, to avoid multiple copies of the data."""
  ndarray: Dict[int, Array] = dataclasses.field(default_factory=dict)
  local_slices: Dict[int, List[SliceNd]] = ddlist_field()
  global_slices: Dict[int, List[SliceNd]] = ddlist_field()

  def add(self, index: int, ndarray: Array, local_slice: SliceNd,
          global_slice: SliceNd):
    curr_ndarray = self.ndarray.get(index)
    if curr_ndarray is None:
      self.ndarray[index] = ndarray  # pylint: disable=unsupported-assignment-operation
    else:
      assert curr_ndarray is ndarray, f'{id(curr_ndarray)} vs. {id(ndarray)}'
    self.local_slices[index].append(local_slice)
    self.global_slices[index].append(global_slice)

  def iter_chunks(self, index: int):
    ndarray = self.ndarray.get(index)
    local_slices = self.local_slices.get(index)
    global_slices = self.global_slices.get(index)
    if ndarray is None or local_slices is None or global_slices is None:
      raise KeyError(f'Index {index} not found.')
    return vmoe.utils.safe_map(lambda loc, glo: (loc.chunk(ndarray), glo),
                               local_slices, global_slices)


@dataclasses.dataclass
class IndexInfo:
  """Info for a checkpointed PyTree containing arrays.

  Attributes:
    global_shape: ShapedArray with the global shape and dtype of the array.
    global_slices: Sequence of SliceNd representing the global slices of each
      of the array's chunks.
    shards: Sequence of integers representing the shard index that contains the
      corresponding array chunk.
  """
  global_shape: jax.ShapedArray
  global_slices: Sequence[SliceNd]
  shards: Sequence[int]
