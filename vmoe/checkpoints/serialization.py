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

"""Extension to flax.serialization module supporting additional custom types.

This is almost a verbatim copy of the flax.serialization module, but we have
added support for serializing some of the types that we have introduced, such
as ShapedArray, Slice, SliceNd, SliceNdArray, IndexInfo, ArrayChunks, etc.
"""
import enum
from typing import Any, Dict, Mapping

import flax.serialization
import jax
import numpy as np
import vmoe.checkpoints.types

# Public functions.
__all__ = [
    'from_state_dict',
    'to_state_dict',
    'register_serialization_state',
    'from_bytes',
    'to_bytes',
    'msgpack_restore',
    'msgpack_serialize',
]


ArrayChunks = vmoe.checkpoints.types.ArrayChunks
IndexInfo = vmoe.checkpoints.types.IndexInfo
LazyArrayChunks = vmoe.checkpoints.types.LazyArrayChunks
PyTree = Any
Slice = vmoe.checkpoints.types.Slice
SliceNd = vmoe.checkpoints.types.SliceNd
SliceNdArray = vmoe.checkpoints.types.SliceNdArray


msgpack = flax.serialization.msgpack

from_state_dict = flax.serialization.from_state_dict
to_state_dict = flax.serialization.to_state_dict
register_serialization_state = flax.serialization.register_serialization_state


def from_bytes(target: PyTree, encoded_bytes: bytes) -> PyTree:
  """Restore optimizer or other object from msgpack-serialized state-dict.

  Args:
    target: template object with state-dict registrations that matches
      the structure being deserialized from `encoded_bytes`.
    encoded_bytes: msgpack serialized object structurally isomorphic to
      `target`.  Typically a flax model or optimizer.

  Returns:
    A new object structurally isomorphic to `target` containing the updated
    leaf data from saved data.
  """
  state_dict = msgpack_restore(encoded_bytes)
  return from_state_dict(target, state_dict)


def msgpack_restore(encoded_pytree: bytes) -> PyTree:
  """Restore data structure from bytes in msgpack format.

  Low-level function that only supports python trees with array leaves,
  for custom objects use `from_bytes`.

  Args:
    encoded_pytree: msgpack-encoded bytes of python tree.

  Returns:
    Python tree of dict, list, tuple with python primitive
    and array leaves.
  """
  state_dict = msgpack.unpackb(
      encoded_pytree,
      ext_hook=_msgpack_ext_unpack,
      object_hook=_array_chunks_decode,
      raw=False)
  return _unchunk_array_leaves_in_place(state_dict)


def msgpack_serialize(pytree: PyTree, in_place: bool = False) -> bytes:
  """Save data structure to bytes in msgpack format.

  Low-level function that only supports python trees with array leaves,
  for custom objects use `to_bytes`.  It splits arrays above MAX_CHUNK_SIZE into
  multiple chunks.

  Args:
    pytree: python tree of dict, list, tuple with python primitives
      and array leaves.
    in_place: boolean specifyng if pytree should be modified in place.

  Returns:
    msgpack-encoded bytes of pytree.
  """
  if not in_place:
    pytree = jax.tree_map(lambda x: x, pytree)
  pytree = _np_convert_in_place(pytree)
  pytree = _chunk_array_leaves_in_place(pytree)
  return msgpack.packb(pytree, default=_msgpack_ext_pack, strict_types=True)


def to_bytes(target: PyTree) -> bytes:
  """Save optimizer or other object as msgpack-serialized state-dict.

  Args:
    target: template object with state-dict registrations to be
      serialized to msgpack format.  Typically a flax model or optimizer.

  Returns:
    Bytes of msgpack-encoded state-dict of `target` object.
  """
  state_dict = to_state_dict(target)
  return msgpack_serialize(state_dict, in_place=True)


# Private functions.

# pylint: disable=protected-access
_chunk_array_leaves_in_place = flax.serialization._chunk_array_leaves_in_place
_dtype_from_name = flax.serialization._dtype_from_name
_ndarray_to_bytes = flax.serialization._ndarray_to_bytes
_ndarray_from_bytes = flax.serialization._ndarray_from_bytes
_np_convert_in_place = flax.serialization._np_convert_in_place
_unchunk_array_leaves_in_place = flax.serialization._unchunk_array_leaves_in_place
_MAX_CHUNK_SIZE = flax.serialization.MAX_CHUNK_SIZE
# pylint: enable=protected-access


_ARRAY_CHUNKS_MAGIC_KEY = '__msgpack_array_chunk_a87ca2__'


class _MsgpackExtType(enum.IntEnum):
  ndarray = 1
  native_complex = 2
  npscalar = 3
  shaped_array = 4
  slice = 5
  slice_nd = 6
  slice_nd_array = 7
  index_info = 8


def _shaped_array_to_bytes(x: jax.ShapedArray) -> bytes:
  tpl = (x.shape, x.dtype.name, x.weak_type, x.named_shape)
  assert all(isinstance(key, str) for key in x.named_shape)
  return msgpack.packb(tpl, use_bin_type=True)


def _shaped_array_from_bytes(data: bytes) -> jax.ShapedArray:
  shape, dtype_name, weak_type, named_shape = msgpack.unpackb(data, raw=True)
  named_shape = {k.decode('utf-8'): v for k, v in named_shape.items()}
  return jax.ShapedArray(
      shape=shape,
      dtype=_dtype_from_name(dtype_name),
      weak_type=weak_type,
      named_shape=named_shape)


def _slice_to_bytes(x: Slice) -> bytes:
  tpl = (x.start, x.stop, x.step)
  return msgpack.packb(tpl, use_bin_type=True)


def _slice_from_bytes(data: bytes) -> Slice:
  start, stop, step = msgpack.unpackb(data, raw=True)
  return Slice(start, stop, step)


def _slice_nd_to_bytes(x: SliceNd) -> bytes:
  tpl = tuple(_slice_to_bytes(s) for s in x)
  return msgpack.packb(tpl, use_bin_type=True)


def _slice_nd_from_bytes(data: bytes) -> SliceNd:
  slices_bytes = msgpack.unpackb(data, raw=True)
  return SliceNd(map(_slice_from_bytes, slices_bytes))


def _slice_nd_array_to_bytes(x: SliceNdArray) -> bytes:
  tpl = (x.shape, tuple(_slice_nd_to_bytes(sa) for sa in x.flatten()))
  return msgpack.packb(tpl, use_bin_type=True)


def _slice_nd_array_from_bytes(data: bytes) -> SliceNdArray:
  shape, slice_nd_bytes = msgpack.unpackb(data, raw=True)
  slice_nd = tuple(map(_slice_nd_from_bytes, slice_nd_bytes))
  slice_nd_array = np.empty(len(slice_nd), dtype=np.object)
  for i, array_slice in enumerate(slice_nd):
    slice_nd_array[i] = array_slice
  return SliceNdArray(slice_nd_array.reshape(shape))


def _lazy_array_chunks_encode(
    lazy_array_chunks: LazyArrayChunks) -> Dict[str, Any]:
  """Encodes a LazyArrayChunks object as a dict."""

  def _encode_chunk(pair):
    chunk, global_slice = pair
    chunk = chunk.flatten()
    subchunk_size = max(1, int(_MAX_CHUNK_SIZE / chunk.dtype.itemsize))
    subchunks = [
        chunk[i:i + subchunk_size]
        for i in range(0, chunk.size, subchunk_size)
    ]
    return [subchunks, global_slice]

  state = {_ARRAY_CHUNKS_MAGIC_KEY: True}
  for array_index in lazy_array_chunks.ndarray.keys():
    state[str(array_index)] = list(
        map(_encode_chunk, lazy_array_chunks.iter_chunks(array_index)))
  return state


def _array_chunks_decode(mapping: Mapping[str, Any]) -> Any:
  """Decodes a dict representing an ArrayChunks object."""

  def _decode_chunk(pair):
    subchunks, global_slice = pair
    flat_chunk = np.concatenate(subchunks)
    shape = tuple(s.stop - s.start for s in global_slice)
    return flat_chunk.reshape(shape), global_slice

  if _ARRAY_CHUNKS_MAGIC_KEY in mapping:
    output = ArrayChunks()
    for array_index, chunks in mapping.items():
      if array_index == _ARRAY_CHUNKS_MAGIC_KEY:
        continue
      array_index = int(array_index)
      for chunk, global_slice in map(_decode_chunk, chunks):
        output.add(array_index, chunk, global_slice)
    return output
  return mapping


def _index_info_to_bytes(index_info: IndexInfo) -> bytes:
  tpl = (_shaped_array_to_bytes(index_info.global_shape),
         tuple(map(_slice_nd_to_bytes, index_info.global_slices)),
         tuple(index_info.shards))
  return msgpack.packb(tpl, use_bin_type=True)


def _index_info_from_bytes(data: bytes) -> IndexInfo:
  global_shape, global_slices, shards = msgpack.unpackb(data, raw=True)
  return IndexInfo(
      global_shape=_shaped_array_from_bytes(global_shape),
      global_slices=tuple(map(_slice_nd_from_bytes, global_slices)),
      shards=tuple(shards))


def _msgpack_ext_pack(x):
  """Msgpack encoders for custom types."""
  if isinstance(x, complex):
    return msgpack.ExtType(_MsgpackExtType.native_complex,
                           msgpack.packb((x.real, x.imag)))
  if isinstance(x, jax.ShapedArray):
    return msgpack.ExtType(_MsgpackExtType.shaped_array,
                           _shaped_array_to_bytes(x))
  if isinstance(x, Slice):
    return msgpack.ExtType(_MsgpackExtType.slice, _slice_to_bytes(x))
  if isinstance(x, SliceNd):
    return msgpack.ExtType(_MsgpackExtType.slice_nd, _slice_nd_to_bytes(x))
  if isinstance(x, SliceNdArray):
    return msgpack.ExtType(_MsgpackExtType.slice_nd_array,
                           _slice_nd_array_to_bytes(x))
  if isinstance(x, LazyArrayChunks):
    return _lazy_array_chunks_encode(x)
  if isinstance(x, IndexInfo):
    return msgpack.ExtType(_MsgpackExtType.index_info, _index_info_to_bytes(x))
  if isinstance(x, (np.ndarray, jax.xla.DeviceArray)):
    return msgpack.ExtType(_MsgpackExtType.ndarray, _ndarray_to_bytes(x))
  if np.issctype(type(x)):
    return msgpack.ExtType(_MsgpackExtType.npscalar,
                           _ndarray_to_bytes(np.asarray(x)))
  return x


def _msgpack_ext_unpack(code, data):
  """Msgpack decoders for custom types."""
  if code == _MsgpackExtType.native_complex:
    complex_tuple = msgpack.unpackb(data)
    return complex(complex_tuple[0], complex_tuple[1])
  if code == _MsgpackExtType.shaped_array:
    return _shaped_array_from_bytes(data)
  if code == _MsgpackExtType.slice:
    return _slice_from_bytes(data)
  if code == _MsgpackExtType.slice_nd:
    return _slice_nd_from_bytes(data)
  if code == _MsgpackExtType.slice_nd_array:
    return _slice_nd_array_from_bytes(data)
  if code == _MsgpackExtType.index_info:
    return _index_info_from_bytes(data)
  if code == _MsgpackExtType.ndarray:
    return _ndarray_from_bytes(data)
  if code == _MsgpackExtType.npscalar:
    ar = _ndarray_from_bytes(data)
    return ar[()]  # unpack ndarray to scalar
  return msgpack.ExtType(code, data)
