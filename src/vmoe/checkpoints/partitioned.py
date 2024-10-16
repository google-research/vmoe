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

"""Functions for checkpointing PyTrees with jax.Array leaves.

jax.Array is a unified type that subsumes many of the existing array types in
JAX, that different functions (jit, pmap, pjit, xmap, ...) returned. The main
benefit of jax.Array is that it was designed to make parallelism (a.k.a. model
partitioning) a core feature of JAX, and simplifies and unifies JAX internals.

The two main functions of this module are `save_checkpoint`, used to save a
sharded checkpoint of a PyTree of jax.Arrays (possibly in a multi-process
system); and `restore_checkpoint`, which restores the values of a PyTree from
a sharded checkpoint.

Sharded checkpoints are made of an index file (with the '.index' suffix), and
a set of data shards (with the '.data-nnnnn-of-NNNNN' suffix). When running the
save and restore functions, each process will handle their corresponding chunks
of each array to save and restore the data in an efficient manner, in a
multi-process system. Sharded checkpoints are also useful for non-partitioned if
the checkpointing is I/O bounded.

"""
import collections
import enum
import functools
import multiprocessing.pool
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from absl import logging
import jax
from jax import core
from jax import Shard
import numpy as np
from vmoe import multihost_utils
from vmoe import utils
from vmoe.checkpoints import base
from vmoe.checkpoints import serialization
from vmoe.checkpoints import types

__all__ = ['restore_checkpoint', 'save_checkpoint']

ArrayChunks = types.ArrayChunks
IndexInfo = types.IndexInfo
LazyArrayChunks = types.LazyArrayChunks
MapResult = multiprocessing.pool.MapResult
GSPMDSharding = jax.sharding.GSPMDSharding
PyTree = Any
Sharding = jax.sharding.Sharding
Slice = types.Slice
SliceNd = types.SliceNd
ThreadPool = multiprocessing.pool.ThreadPool
tree_flatten = jax.tree_util.tree_flatten
tree_map = jax.tree_util.tree_map

# When checkpoints are restored, we can load multiple files concurrently, using
# at most this number of bytes.
MAX_CONCURRENT_BYTES = 8_589_934_592  # 8GB.
# When checkpoints are saved, we chunk each array in smaller pieces so that each
# chunk has at most this number of bytes.
MAX_CHUNK_BYTES = 268_435_456  # 256MB.


from_state_dict = serialization.from_state_dict
to_state_dict = serialization.to_state_dict
safe_map = utils.safe_map
safe_zip = utils.safe_zip


class Version(enum.Enum):
  UNKNOWN = None
  V1 = '20220622'


def restore_checkpoint(
    prefix: str,
    tree: Optional[PyTree],
    *,
    thread_pool: Optional[ThreadPool] = None,
    max_concurrent_bytes: Optional[int] = MAX_CONCURRENT_BYTES) -> PyTree:
  """Restores a PyTree with jax.Array leaves from sharded checkpoint.

  Args:
    prefix: Prefix of the checkpoint file (e.g. "/tmp/checkpoint_step_2").
    tree: Optional PyTree with the expected structure to restore. If None,
      a nested dict is returned (i.e. the FLAX state dict). If given, the
      restored structure must match the given one.
    thread_pool: ThreadPool used to read the checkpoint files asynchronously.
      If None, a new pool will be created.
    max_concurrent_bytes: Maximum number of bytes to read concurrentlly. This
      maximum is useful to prevent out of memory errors while reading files from
      disk. If None, no maximum is enforced.

  Returns:
    The restored PyTree.
  """
  # Restore index as a state dict.
  index = base.restore_checkpoint(prefix + '.index')
  version = Version(index.get('version', Version.UNKNOWN))
  shard_count = index['shard_count']
  index = index['index']
  if version == Version.UNKNOWN:
    return _restore_checkpoint_from_index_unknown(
        prefix, shard_count, index, tree, thread_pool=thread_pool,
        max_concurrent_bytes=max_concurrent_bytes)
  if version == Version.V1:
    return _restore_checkpoint_from_index_v1(
        prefix, shard_count, index, tree, thread_pool=thread_pool,
        max_concurrent_bytes=max_concurrent_bytes)
  raise NotImplementedError(f'Unsupported checkpoint version: {version!r}')


def save_checkpoint(
    prefix: str,
    tree: PyTree,
    *,
    num_shards: Optional[int] = None,
    overwrite: bool = True,
    makedirs: bool = True,
    thread_pool: Optional[ThreadPool] = None,
    max_chunk_bytes: Optional[int] = MAX_CHUNK_BYTES,
    version: Version = Version.V1) -> MapResult:
  """Saves a PyTree with jax.Array leaves to a sharded checkpoint.

  Args:
    prefix: Prefix of the checkpoint file (e.g. "/tmp/checkpoint_step_2").
    tree: PyTree to save, with jax.Array leaves.
    num_shards: Number of checkpoint shards. If None, `jax.process_count()` is
      used. This number should be always larger than or equal to the number of
      processes.
    overwrite: If True, rewrites any file that might exist. If False, raises an
      exception if a given file existed.
    makedirs: If True, create the base dir of prefix if it doesn't exist.
      If False, the existence of the base dir is assumed.
    thread_pool: ThreadPool used to write the checkpoint files asynchronously.
      If None, a new pool will be created.
    max_chunk_bytes: Optional integer with the maximum size (in bytes) of each
      array chunk. If an array is bigger than this, it will be broken into
      smaller pieces so that each chunk has at most this number of bytes.
      This is useful to write multiple chunks of a big array into several
      checkpoint shards to increase the speed of saving/restoring checkpoints.
    version: Write checkpoints using this version. DO NOT CHANGE UNLESS YOU KNOW
      WHAT YOU ARE DOING.

  Returns:
    A MapResult object.
  """
  assert version == Version.V1, f'{version=} != {Version.V1}'
  tree = to_state_dict(tree)

  leaves, struct = tree_flatten(tree)
  if not all(isinstance(x, jax.Array) for x in leaves):
    raise ValueError('All leaves in the input tree must be jax.Array.')
  # Start asynchronous copy to host memory.
  tree_map(lambda x: x.copy_to_host_async(), leaves)

  if num_shards is None or num_shards < jax.process_count():
    if num_shards is not None:
      logging.warning(
          'While saving checkpoint with prefix=%r we found num_shards=%d and '
          'process_count=%d. Resetting num_shards=%d.',
          prefix, num_shards, jax.process_count(), jax.process_count())
    num_shards = jax.process_count()

  bytes_per_shard = [0] * num_shards
  index_leaves = []
  # Dictionary with the checkpoint shards handled by the current process.
  ckpt_shard_to_lazy_array_chunks = {
      i: LazyArrayChunks()
      for i in range(num_shards)
      if i % jax.process_count() == jax.process_index()
  }
  for i, arr in enumerate(leaves):
    slice_to_arr_shards = _create_map_slicend_to_array_shards(
        arr, max_chunk_bytes)
    slice_to_ckpt_shard = _create_map_slicend_to_checkpoint_shard(
        slice_to_arr_shards, arr.dtype.itemsize, bytes_per_shard)
    index_leaves.append(_make_index_info(slice_to_ckpt_shard, arr))
    _update_lazy_array_chunks_with_leaf(i, slice_to_arr_shards,
                                        slice_to_ckpt_shard,
                                        ckpt_shard_to_lazy_array_chunks)
  # Convert jax.Arrays (which might be on-device) to Numpy arrays and replace
  # references in the LazyArrayChunks structures. This is important when
  # checkpointing happens asynchronously, since JAX could release a device
  # buffer that hasn't been checkpointed yet.
  _replace_jax_with_numpy_in_lazy_array_chunks(ckpt_shard_to_lazy_array_chunks)
  filepath_to_data = _create_map_filepath_to_data(
      prefix, struct, index_leaves, ckpt_shard_to_lazy_array_chunks,
      num_shards, version)
  del ckpt_shard_to_lazy_array_chunks

  if makedirs:
    _make_dir_sync(os.path.dirname(prefix))
  return base.save_multiple_checkpoints_async(
      filepath_to_data,
      overwrite=overwrite,
      makedirs=False,
      thread_pool=thread_pool)


def _create_local_buffers(
    sharding: Sharding,
    global_shape: Tuple[int, ...],
    dtype,
) -> Dict[SliceNd, np.ndarray]:
  """Creates arrays to store values of addressable array shards."""
  shard_shape = sharding.shard_shape(global_shape)
  output = {}
  for index in sharding.addressable_devices_indices_map(global_shape).values():
    global_slice = _shard_index_to_slicend(index, global_shape)
    if global_slice not in output:
      output[global_slice] = np.zeros(shard_shape, dtype=dtype)
  return output


def _create_map_filepath_to_data(
    prefix: str,
    struct,
    index_leaves: List[IndexInfo],
    ckpt_shard_to_lazy_array_chunks: Dict[int, LazyArrayChunks],
    shard_count: int,
    version: Version = Version.V1,
) -> Dict[str, Any]:
  """Creates a map from checkpoint shard filenames to the contents to store."""
  shard_fpath_fn = functools.partial(
      base.add_shard_suffix,
      filepath=prefix + '.data',
      shard_count=shard_count)
  filepath_to_data = {
      shard_fpath_fn(shard=ckpt_shard): obj
      for ckpt_shard, obj in ckpt_shard_to_lazy_array_chunks.items()
  }
  if jax.process_index() == 0:
    index_content = {
        'shard_count': shard_count,
        'index': struct.unflatten(index_leaves),
    }
    if version is not Version.UNKNOWN:
      index_content['version'] = version.value
    filepath_to_data[prefix + '.index'] = index_content

  return filepath_to_data


def _create_map_slicend_to_array_shards(
    arr,
    max_chunk_bytes: Optional[int] = MAX_CHUNK_BYTES,
) -> Dict[SliceNd, List[Shard]]:
  """Returns a dict mapping SliceNd to the corresponding array shards.

  The shards of an array are potentially split in smaller chunks so that every
  chunk has at most max_chunk_bytes.

  Args:
    arr: jax.Array.
    max_chunk_bytes: Maximum size (in bytes) of the chunks.

  Returns:
    A dict mapping from a chunk Index to a list of JAX array Shards.
  """
  slicend_to_shards = collections.defaultdict(list)
  for shard in arr.global_shards:
    slicend = _shard_index_to_slicend(shard.index, arr.shape)
    slicend_to_shards[slicend].append(shard)
  if max_chunk_bytes and max_chunk_bytes >= 1:
    max_chunk_items = max(max_chunk_bytes // arr.dtype.itemsize, 1)
  else:
    max_chunk_items = None
  # pylint: disable=g-complex-comprehension
  slicend_to_shards = {
      subslicend: shards for slicend, shards in slicend_to_shards.items()
      for subslicend in _split_slicend(slicend, max_chunk_items)
  }
  # pylint: enable=g-complex-comprehension
  return slicend_to_shards


def _create_map_slicend_to_checkpoint_shard(
    slicend_to_shards: Dict[SliceNd, List[Shard]],
    itemsize: int,
    bytes_per_ckpt_shard: List[int],
) -> Dict[SliceNd, int]:
  """Returns a mapping from SliceNd to the checkpoint shard storing them.

  The checkpoint shard for each SliceNd is chosen so that all checkpoint shards
  hold roughly the same number of bytes.

  Args:
    slicend_to_shards: Dict mapping from SliceNd to the corresponding list of
      JAX array Shards.
    itemsize: Size (in bytes) of each chunk element.
    bytes_per_ckpt_shard: List containing the number of bytes that each
      checkpoint shard holds.
      This list is updated according to the selected checkpoint shard for each
      index.

  Returns:
    A dict mapping from Index to an integer, representing the id of the
    checkpoint shard in which it will be stored.
  """
  slicend_to_ckpt_shard = {}
  for slicend in sorted(slicend_to_shards):
    # Array shards that contain the current Slice.
    shards = slicend_to_shards[slicend]
    # Processes that hold each of the Array shards, sorted by process_index.
    processes = tuple(sorted(set([s.device.process_index for s in shards])))
    # Find the smallest checkpoint shard handled by one of the processes holding
    # the SliceNd.
    ckpt_shard = _find_smallest_ckpt_shard(bytes_per_ckpt_shard, processes)
    # The current SliceNd will be stored in the selected checkpoint shard (and
    # written by the corresponding process).
    slicend_to_ckpt_shard[slicend] = ckpt_shard
    # Update the number of bytes written to the selected checkpoint shard.
    size = np.prod(tuple(s.stop - s.start for s in slicend)) * itemsize
    bytes_per_ckpt_shard[ckpt_shard] += size
  return slicend_to_ckpt_shard


def _find_ckpt_shards_to_restore(
    sharding: Sharding,
    global_shape: Tuple[int, ...],
    ckpt_slices: Sequence[SliceNd],
    ckpt_shards: Sequence[int],
) -> Iterator[int]:
  """Iterates over checkpoint shards to restore values of addressable array shards."""
  for index in sharding.addressable_devices_indices_map(global_shape).values():
    global_slice = _shard_index_to_slicend(index, global_shape)
    for ckpt_slice, ckpt_shard in safe_zip(ckpt_slices, ckpt_shards):
      if (global_slice == ckpt_slice or  # account for scalar with shape ().
          _intersect_slicend(global_slice, ckpt_slice)):
        yield ckpt_shard


def _find_smallest_ckpt_shard(
    bytes_per_ckpt_shard: List[int], processes: Tuple[int, ...]) -> int:
  """Returns the smallest checkpoint shard, handled by one of the given processes."""
  # Sort checkpoint shards by their size, in increasing order.
  bytes_per_ckpt_shard = [(b, i) for i, b in enumerate(bytes_per_ckpt_shard)]
  bytes_per_ckpt_shard = sorted(bytes_per_ckpt_shard)
  # The i-th ckpt shard is handled by the process index = i % process_count.
  # From the previous sorted bytes_per_ckpt_shard, ignore all shards that aren't
  # handled by one of the given processes.
  return [
      i for _, i in bytes_per_ckpt_shard
      if i % jax.process_count() in processes][0]


def _get_array_sharding_or_default(arr: jax.Array) -> Sharding:
  if hasattr(arr, 'sharding'):
    return arr.sharding
  else:
    return GSPMDSharding.get_replicated(jax.devices())


def _intersect_slicend(a: SliceNd, b: SliceNd) -> Optional[SliceNd]:
  output = []
  for s_a, s_b in safe_zip(a, b):
    start = max(s_a.start or 0, s_b.start or 0)
    stop = min(np.inf if s_a.stop is None else s_a.stop,
               np.inf if s_b.stop is None else s_b.stop)
    if stop <= start:
      return None
    output.append(Slice(start, None if stop == np.inf else stop))
  return SliceNd(output)


def _make_dir_sync(workdir: str):
  # Process 0 creates the workdir if it doesn't exist. All processes wait
  # until it's done.
  if jax.process_index() == 0:
    base.gfile.makedirs(workdir)
  multihost_utils.sync_devices(f'checkpoints:mkdir:{workdir}')


def _make_index_info(
    slicend_to_ckpt_shards: Dict[SliceNd, int],
    array: jax.Array) -> IndexInfo:
  global_shape = core.ShapedArray(
      shape=array.shape, dtype=array.dtype, weak_type=array.weak_type)
  global_slices = sorted(slicend_to_ckpt_shards.keys())
  shards = [slicend_to_ckpt_shards[slicend] for slicend in global_slices]
  return IndexInfo(global_shape=global_shape,
                   global_slices=global_slices,
                   shards=shards)


def _replace_jax_with_numpy_in_lazy_array_chunks(
    ckpt_shard_to_lazy_array_chunks: Dict[int, LazyArrayChunks],
):
  id_to_array = {}
  for lac in ckpt_shard_to_lazy_array_chunks.values():
    for lst in lac.chunks.values():
      for arr, _, _ in lst:
        id_to_array[id(arr)] = arr
  id_to_array = jax.tree_util.tree_map(np.asarray, id_to_array)
  for lac in ckpt_shard_to_lazy_array_chunks.values():
    for i, lst in lac.chunks.items():
      lac.chunks[i] = [
          (id_to_array[id(arr)], ls, gs) for arr, ls, gs in lst
      ]


def _restore_checkpoint_from_index(
    prefix: str,
    shard_count: int,
    index: PyTree,
    sharding: PyTree,
    thread_pool: Optional[ThreadPool] = None,
    max_concurrent_bytes: Optional[int] = MAX_CONCURRENT_BYTES,
) -> PyTree:
  """Restores a PyTree of partitioned arrays from an index."""
  thread_pool = thread_pool or ThreadPool()
  # Flatten index and axis_resources. Check that the two are compatible.
  index, struct = tree_flatten(index)
  shardings, struct2 = tree_flatten(sharding)
  if struct != struct2:
    raise ValueError(f'The tree structs do not match.\n'
                     f'index: {struct}\n'
                     f'sharding: {struct2}')
  # Create Numpy arrays to store the values of each of the addressable array
  # shards addressable.
  local_buffers = [
      _create_local_buffers(s, i.global_shape.shape, i.global_shape.dtype)
      for s, i in safe_zip(shardings, index)
  ]
  # Find which checkpoint shards must be read by the current process to restore
  # values of the addressable array shards.
  ckpt_shards_to_restore = [
      _find_ckpt_shards_to_restore(s, i.global_shape.shape, i.global_slices,
                                   i.shards)
      for s, i in safe_zip(shardings, index)
  ]
  ckpt_shards_to_restore = frozenset().union(*ckpt_shards_to_restore)
  ckpt_shards_to_restore = [
      base.add_shard_suffix(prefix + '.data', i, shard_count)
      for i in ckpt_shards_to_restore
  ]
  # If max_concurrent_bytes is specified, calculate the maximum number of files
  # to read from concurrently. Otherwise, we just read from as many files as
  # possible.
  if max_concurrent_bytes:
    total_bytes_to_restore = sum(
        thread_pool.map(lambda fpath: base.gfile.stat(fpath).length,
                        ckpt_shards_to_restore))
    num_ckpt_shards_per_thread = round(
        total_bytes_to_restore / max_concurrent_bytes)
    num_ckpt_shards_per_thread = max(num_ckpt_shards_per_thread, 1)
  else:
    num_ckpt_shards_per_thread = None
  # Restore the values of the local buffers (i.e. addressable shards), reading
  # from the corresponding checkpoint shards.
  thread_pool.map(
      lambda fpath: _restore_local_buffers(fpath, local_buffers),
      ckpt_shards_to_restore,
      chunksize=num_ckpt_shards_per_thread)
  # Create JAX Arrays from the Numpy buffers.
  def _make_jax_array(i: int) -> jax.Array:
    shape, sharding = index[i].global_shape.shape, shardings[i]
    cb = lambda idx: local_buffers[i][_shard_index_to_slicend(idx, shape)]
    array = jax.make_array_from_callback(shape, sharding, cb)
    local_buffers[i] = None
    return array
  arrays = thread_pool.map(_make_jax_array, range(len(index)))
  return struct.unflatten(arrays)


def _restore_checkpoint_from_index_unknown(
    prefix: str, shard_count: int, index: PyTree, tree: PyTree,
    *args, **kwargs):
  """Restores from an unknown (old) checkpoint version."""
  if tree is None:
    raise ValueError(
        'You must specify the tree arguments when restoring from '
        f'{Version.UNKNOWN}.')
  sharding = tree_map(_get_array_sharding_or_default, tree)
  # The index has the same structure as the input tree and the sharding.
  # Obtain such structure before calling _restore_checkpoint.
  index = from_state_dict(target=tree, state=index)
  return _restore_checkpoint_from_index(
      prefix, shard_count, index, sharding, *args, **kwargs)


def _restore_checkpoint_from_index_v1(
    prefix: str, shard_count: int, index: PyTree, tree: Optional[PyTree],
    *args, **kwargs):
  """Restores from a V1 checkpoint."""
  if tree is not None:
    sharding = tree_map(_get_array_sharding_or_default, tree)
    sharding_state_dict = to_state_dict(sharding)
  else:
    sharding = tree_map(_get_array_sharding_or_default, index)
    sharding_state_dict = sharding
  state_dict = _restore_checkpoint_from_index(
      prefix, shard_count, index, sharding_state_dict, *args, **kwargs)
  return from_state_dict(target=sharding, state=state_dict)


def _restore_local_buffers(
    filepath: str,
    local_buffers: Sequence[Dict[SliceNd, np.ndarray]],
):
  """Restores array chunks from a checkpoint file, filling local arrays.

  Args:
    filepath: Filepath of the file to restore from. This is typically the
      filepath of a checkpoint shard.
    local_buffers: Sequence of dictionaries mapping from global slices
      (identifying array shards) to Numpy arrays. Parts of the Numpy arrays are
      filled with the values read from the corresponding checkpoint shard.
  """
  ckpt_chunks = base.restore_checkpoint(filepath, ArrayChunks())
  for i in ckpt_chunks.chunks:
    for ckpt_chunk, ckpt_slice in zip(
        ckpt_chunks.chunks[i], ckpt_chunks.global_slices[i]):
      for global_slice, buffer in local_buffers[i].items():
        intersect_slice = _intersect_slicend(global_slice, ckpt_slice)
        if (global_slice == ckpt_slice or  # account for scalars with shape ().
            intersect_slice):
          local_intersect_slice = tuple(
              slice(a.start - b.start, a.stop - b.start)
              for a, b in safe_zip(intersect_slice, global_slice))
          ckpt_intersect_slice = tuple(
              slice(a.start - b.start, a.stop - b.start)
              for a, b in safe_zip(intersect_slice, ckpt_slice))
          buffer[local_intersect_slice] = ckpt_chunk[ckpt_intersect_slice]


def _shard_index_to_slicend(
    index: Tuple[slice, ...], global_shape: Tuple[int, ...]) -> SliceNd:
  return SliceNd([
      Slice(s.start or 0, size if s.stop is None else s.stop)
      for s, size in safe_zip(index, global_shape)])


def _split_slicend(
    slicend: SliceNd,
    max_chunk_items: Optional[int] = None,
) -> Sequence[SliceNd]:
  """Splits an SliceNd into smaller chunks with at most max_chunk_items."""
  shape = tuple(s.stop - s.start for s in slicend)
  chunk_items = np.prod(shape)
  chunks = [slicend]
  max_chunk_items: int = max_chunk_items or -1
  if max_chunk_items < 1 or chunk_items <= max_chunk_items:
    return chunks
  # Split each dimension into a suitable number of splits, in increasing order
  # of dimension size, to minimize the number of chunks and ensure that the
  # number of items in the chunks is <= max_chunk_items.
  for i, size in sorted(enumerate(shape), key=lambda t: (t[1], t[0])):
    if size == 1: continue
    # pylint: disable=g-complex-comprehension
    splits = [
        # Note: we should try with all divisors of size here, but we simply try
        # a few integers to simplify the code.
        s for s in range(2, 10)
        if size % s == 0 and chunk_items // s <= max_chunk_items
    ]
    splits = (splits or [size])[0]
    start, stop, step = slicend[i].start, slicend[i].stop, size // splits
    chunks = [chunk[:i] + (Slice(j, j + step),) + chunk[i + 1:]
              for chunk in chunks for j in range(start, stop, step)]
    chunk_items /= splits
    if chunk_items <= max_chunk_items:
      break
    # pylint: enable=g-complex-comprehension
  return [SliceNd(chunk) for chunk in chunks]


def _update_lazy_array_chunks_with_leaf(
    index: int,
    slicend_to_arr_shards: Dict[SliceNd, List[Shard]],
    slicend_to_ckpt_shard: Dict[SliceNd, int],
    ckpt_shard_to_lazy_array_chunks: Dict[int, LazyArrayChunks],
):
  """Updates the LazyArrayChunks of checkpoint shards with the chunks of a leaf.

  Args:
    index: Index of the leaf.
    slicend_to_arr_shards: Mapping from a SliceNd to the list of corresponding
      JAX array shards. The SliceNd represents a global (sub) slice of the array
      in the leaf. At least 1 shard must be held by process_index.
    slicend_to_ckpt_shard: Mapping from a SliceNd to the checkpoint shard that
      will store it.
    ckpt_shard_to_lazy_array_chunks: Dictionary mapping from a checkpoint shard
      to a LazyArrayChunks object. This object will be updated.
  """
  # Maps a global (sub) SliceNd to a local one, to chunk data from the array
  # shard. E.g. suppose an array shard with index (slice(5, 10), slice(3, 9)).
  # We have the global slice (slice(6, 8), slice(3, 8)). Then this global slice
  # corresponds to the local slice (slice(1, 3), slice(0, 5)).
  def _global2local(shard: Shard, global_slice: SliceNd) -> SliceNd:
    return SliceNd([
        Slice(s.start - (i.start or 0), s.stop - (i.start or 0))
        for i, s in safe_zip(shard.index, global_slice)
    ])

  for global_slice, ckpt_shard in slicend_to_ckpt_shard.items():
    if ckpt_shard % jax.process_count() == jax.process_index():
      found = False
      for arr_shard in slicend_to_arr_shards[global_slice]:
        if arr_shard.device.process_index == jax.process_index():
          local_slice = _global2local(arr_shard, global_slice)
          ckpt_shard_to_lazy_array_chunks[ckpt_shard].add(
              index, arr_shard.data, local_slice, global_slice)
          found = True
          break
      if not found:
        raise ValueError(
            f'No array shards for leaf with {index=} are held by '
            f'{jax.process_index()=}. The problem was found while processing '
            f'{global_slice=} in {ckpt_shard=}. This is a bug, it should '
            'never happen.')
