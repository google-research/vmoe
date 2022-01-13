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

"""Functions for checkpointing partitioned models."""
import collections
import functools
import itertools
import os
from typing import Any, Iterator, Iterable, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.experimental.maps as maps
import jax.experimental.pjit as pjit
import numpy as np
import vmoe.checkpoints.base
import vmoe.checkpoints.types
import vmoe.multihost_utils
import vmoe.utils

__all__ = ['restore_checkpoint', 'save_checkpoint']


Array = Union[jax.numpy.ndarray, np.ndarray]
ArrayChunks = vmoe.checkpoints.types.ArrayChunks
AsyncResult = vmoe.checkpoints.base.AsyncResult
IndexInfo = vmoe.checkpoints.types.IndexInfo
LazyArrayChunks = vmoe.checkpoints.types.LazyArrayChunks
Mesh = maps.Mesh
ParsedPartitionSpec = pjit.ParsedPartitionSpec
PartitionSpec = pjit.PartitionSpec
PyTree = Any
Slice = vmoe.checkpoints.types.Slice
SliceNd = vmoe.checkpoints.types.SliceNd
SliceNdArray = vmoe.checkpoints.types.SliceNdArray
ThreadPool = vmoe.checkpoints.base.ThreadPool

safe_map = vmoe.utils.safe_map
safe_zip = vmoe.utils.safe_zip


def restore_checkpoint(*,
                       prefix: str,
                       tree: Optional[PyTree],
                       axis_resources: Optional[PyTree],
                       mesh: Optional[Mesh] = None,
                       thread_pool: Optional[ThreadPool] = None) -> PyTree:
  """Restores a PyTree of partitioned arrays from sharded checkpoint.

  Args:
    prefix: Prefix of the checkpoint file (e.g. "/tmp/checkpoint_step_2").
    tree: Optional PyTree with the expected structure to restore. If given,
      and the restored structure doesn't match an exception is raised.
    axis_resources: Optional PyTree with PartitionSpec leaves, with the same
      structure as `tree`, indicating how each axis of the corresponding array
      is partitioned across the axes of the logical mesh.
    mesh: Logical mesh, indicating which device holds each element of the mesh.
    thread_pool: ThreadPool used to read the checkpoint files asynchronously.
      If None, a new pool will be created.

  Returns:
    The restored PyTree.
  """
  if mesh is None:
    mesh = _get_current_mesh()
  if mesh.empty:
    raise ValueError("You must pass a non-empty mesh. If you didn't pass any, "
                     "check that you called restore_checkpoint from a "
                     "maps.mesh context.")
  index = vmoe.checkpoints.base.restore_checkpoint(prefix + '.index', {
      'shard_count': 0,
      'index': tree if tree is not None else axis_resources,
  })
  shard_count = index['shard_count']
  index = index['index']
  # axis_resources indicates how the data to be loaded will be partitioned.
  # If no axis_resources is given, assume that we don't want any partitioning.
  # This implies that all devices will store a copy of all the parameters, thus
  # all processes will have to read from all checkpoint shards.
  if axis_resources is None:
    axis_resources = jax.tree_map(lambda _: PartitionSpec(), index)
  # Flatten index and axis_resources. Check that the two are compatible.
  index, struct = jax.tree_flatten(index)
  _, axis_resources, struct2 = _prepare_axis_resources(axis_resources,
                                                       'axis_resources')
  if struct != struct2:
    raise ValueError(f'The tree structs do not match.\n'
                     f'index: {struct}\n'
                     f'axis_resources: {struct2}')
  # Get local (w.r.t. process) and global ShapedArray of all arrays.
  # This indicates the shape and type of all the arrays in the tree to restore.
  global_avals = [x.global_shape for x in index]
  positional_semantics = [_PositionalSemantics.LOCAL for _ in global_avals]
  local_avals = pjit.global_to_local(positional_semantics, mesh, global_avals,
                                     axis_resources)
  local_mesh = mesh.local_mesh
  # For each array to restore, get local/global SliceNdArrays indicating which
  # local/global slice is stored in each device of the local/global mesh.
  local_slices_arrays = _make_slice_nd_arrays(local_avals, axis_resources,
                                              local_mesh)
  global_slices_arrays = _make_slice_nd_arrays(global_avals, axis_resources,
                                               mesh)
  # For each array to restore: get a set of (local slice, global slice) that
  # will be handled by the devices of the current process.
  local_global_slices = _pair_local_and_global_slices(
      local_slices_arrays, global_slices_arrays, mesh, local_mesh)
  local_global_slices = list(local_global_slices)
  # For each array to restore: get the (global) slices in the checkpoint and the
  # shard that stores the corresponding chunk of data. The result is a sequence
  # of (ckpt slice, shard).
  ckpt_slices_and_shards = safe_map(
      lambda x, y: tuple(safe_zip(x, y)),
      [x.global_slices for x in index], [x.shards for x in index])
  # Create a map from shard number, to array index, to a list of slices to load
  # from that shard. The list has tuple (ckpt slice, ckpt subslice,
  # local subslice).
  shard2array2slices = collections.defaultdict(
      lambda: collections.defaultdict(list))
  for a, (array_local_global_slices, array_ckpt_slices_and_shards) in enumerate(
      zip(local_global_slices, ckpt_slices_and_shards)):
    for shard, *slices in _match_checkpoint_to_local_slices(
        array_local_global_slices, array_ckpt_slices_and_shards):
      shard2array2slices[shard][a].append(tuple(slices))
  shard2array2slices = {
      vmoe.checkpoints.base.add_shard_suffix(prefix + '.data', shard,
                                             shard_count): value
      for shard, value in shard2array2slices.items()
  }
  # Allocate memory for all local arrays to restore. The values of these arrays
  # will be restored asynchronously with _restore_array_chunks.
  local_arrays = [
      np.zeros(aval.shape, dtype=aval.dtype) for aval in local_avals
  ]
  thread_pool = thread_pool or ThreadPool()
  # pylint: disable=g-long-lambda
  thread_pool.map(
      lambda args: _restore_array_chunks(args[0], local_arrays, args[1]),
      shard2array2slices.items())
  # pylint: enable=g-long-lambda
  return struct.unflatten(local_arrays)


def save_checkpoint(*,
                    prefix: str,
                    tree: PyTree,
                    axis_resources: PyTree,
                    mesh: Optional[Mesh] = None,
                    num_shards: int = 0,
                    overwrite: bool = True,
                    makedirs: bool = True,
                    thread_pool: Optional[ThreadPool] = None) -> AsyncResult:
  """Saves a PyTree of partitioned arrays into a sharded checkpoint.

  Args:
    prefix: Prefix of the checkpoint file (e.g. "/tmp/checkpoint_step_2").
    tree: PyTree with ndarray leaves to checkpoint.
    axis_resources: PyTree with PartitionSpec leaves, with the same structure as
      `tree`, indicating how each axis of the corresponding array is
      partitioned across the axes of the logical mesh.
    mesh: Logical mesh, indicating which device holds each element of the mesh.
    num_shards: Number of checkpoint shards. If `num_shards <= 0`, the minimum
      number of shards will be used. If `num_shards > 0`, this number is only
      tentative.
    overwrite: If True, rewrites any file that might exist. If False, raises an
      exception if a given file existed.
    makedirs: If True, create the base dir of prefix if it doesn't exist.
      If False, the existence of the base dir is assumed.
    thread_pool: ThreadPool used to write the checkpoint files asynchronously.
      If None, a new pool will be created.

  Returns:
    An AsyncResult object.
  """
  if mesh is None:
    mesh = _get_current_mesh()
  if mesh.empty:
    raise ValueError("You must pass a non-empty mesh. If you didn't pass any, "
                     "check that you called save_checkpoint from a maps.mesh "
                     "context.")
  filepath_map = _make_save_checkpoint_filepath_map(prefix, tree,
                                                    axis_resources, mesh,
                                                    num_shards)
  if makedirs:
    # Process 0 creates the workdir if it doesn't exist. All processes wait
    # until it's done.
    wdir = os.path.dirname(prefix)
    if jax.process_index() == 0:
      vmoe.checkpoints.base.gfile.makedirs(wdir)
    vmoe.multihost_utils.sync_devices(f'checkpoints:mkdir:{wdir}')
  # Write checkpoint shards asynchronously.
  return vmoe.checkpoints.base.save_multiple_checkpoints_async(
      filepath_map,
      overwrite=overwrite,
      makedirs=False,
      thread_pool=thread_pool)


# pylint: disable=protected-access
_PositionalSemantics = maps._PositionalSemantics
_prepare_axis_resources = pjit._prepare_axis_resources
# pylint: enable=protected-access


def _convert_native_to_slices_nd_array(
    slices_array: np.ndarray,
    aval: jax.ShapedArray) -> SliceNdArray:
  """Converts a ndarray with tuples of Python `slice` to a SliceNdArray."""
  original_shape = slices_array.shape
  slices_array = slices_array.flatten()
  for i, slices in enumerate(slices_array):
    assert all(isinstance(s, slice) for s in slices)
    # Convert the slice's start/stop from None to int.
    # start=None -> 0, stop=None -> size.
    slices = (slice(s.start or 0, size if s.stop is None else s.stop, s.step)
              for s, size in zip(slices, aval.shape))
    slices_array[i] = SliceNd(Slice(s) for s in slices)
  return SliceNdArray(slices_array.reshape(original_shape))


def _create_lazy_array_chunks_per_shard(
    ndarrays: Sequence[Array],
    local_slices_arrays: Sequence[SliceNdArray],
    global_slices_arrays: Sequence[SliceNdArray],
    shard_per_global_slices: Sequence[Sequence[int]],
    process_per_shard: Tuple[int, ...],
    mesh: Mesh,
    local_mesh: Optional[Mesh],
) -> Mapping[int, LazyArrayChunks]:
  """Creates mapping from shard to LazyArrayChunks.

  A "chunk" is the result of taking a SliceNd object and using it to slice
  a particular array. A LazyArrayChunks object represents all the chunks from
  all arrays that must be written on a particular checkpoint shard. We don't
  directly chunk the arrays here, since this could incur in some extra memory
  used (at the moment of writing a shard: original data + chunks + serialized
  chunks).

  Args:
    ndarrays: Sequence of input arrays.
    local_slices_arrays: Sequence of SliceNdArray indicating how each input
      array is partitioned across the local (for the current process) mesh of
      devices.
    global_slices_arrays: Sequence of SliceNdArray indicating how each input
      array is partitioned across the global mesh of devices.
    shard_per_global_slices: Sequence of sequences of integers that indicate,
      for each input array, which shard will write the corresponding (global)
      slice of the array.
    process_per_shard: Tuple of integers indicating which process handles each
      shard.
    mesh: Global mesh of devices.
    local_mesh: Local (restricted to a given process) mesh of devices.

  Returns:
    A dictionary mapping shards (integers) to LazyArrayChunks.
  """
  local_mesh = local_mesh or mesh.local_mesh
  # For each leaf array, we pair the local slices and the global slices.
  # The pairs are unique (repetitions due to replication have been removed).
  local_global_slices = _pair_local_and_global_slices(local_slices_arrays,
                                                      global_slices_arrays,
                                                      mesh, local_mesh)
  output = collections.defaultdict(LazyArrayChunks)
  outer_zip = safe_zip(ndarrays, global_slices_arrays, local_global_slices,
                       shard_per_global_slices)
  for index, (ndarray, global_slice_ndarray, local_global_slice_nds,
              shard_per_slice) in enumerate(outer_zip):
    global2local_slices = dict((gs, ls) for ls, gs in local_global_slice_nds)
    global_slice_ndarray = sorted(set(global_slice_ndarray.flatten()))
    for global_slice, shard in safe_zip(global_slice_ndarray, shard_per_slice):
      if process_per_shard[shard] == jax.process_index():
        local_slice = global2local_slices[global_slice]
        output[shard].add(index, ndarray, local_slice, global_slice)
  return output


def _get_current_mesh() -> Mesh:
  return maps.thread_resources.env.physical_mesh


def _intersect_slice_nd(
    ckpt_slice_nd: SliceNd,
    global_slice_nd: SliceNd,
    local_slice_nd: SliceNd,
) -> Optional[Tuple[SliceNd, SliceNd]]:
  """Finds the intersection of a checkpoint SliceNd with a local SliceNd."""
  intersect_slice_nd_ckpt, intersect_slice_nd_local = [], []
  for c, g, l in vmoe.utils.safe_zip(ckpt_slice_nd, global_slice_nd,
                                     local_slice_nd):
    # Assumes that step=None, and that start/stop are not None.
    start, stop = max(c.start, g.start), min(c.stop, g.stop)
    if stop <= start:
      return None
    ckpt_stop = None if stop == np.inf else stop - c.start
    intersect_slice_nd_ckpt.append(Slice(start - c.start, ckpt_stop))
    intersect_slice_nd_local.append(
        Slice(start - g.start + l.start, stop - g.start + l.start))
  return SliceNd(intersect_slice_nd_ckpt), SliceNd(intersect_slice_nd_local)


def _make_save_checkpoint_filepath_map(
    prefix: str, tree: PyTree, axis_resources: PyTree, mesh: Mesh,
    num_shards: int = 0):
  """Makes a dictionary of filepaths mapping to the content that must be serialized."""
  filepath_map = {}  # Result.
  tree_leaves, struct = jax.tree_flatten(tree)
  _, axis_resources, struct2 = _prepare_axis_resources(axis_resources,
                                                       'axis_resources')
  if struct != struct2:
    raise ValueError('The tree structs do not match.\n'
                     f'tree: {struct}\n'
                     f'axis_resources: {struct2}')
  # Get local (w.r.t. process) and global ShapedArray of all arrays.
  # This indicates the shape and type of all the arrays in the input tree.
  local_avals = [
      x.aval if hasattr(x, 'aval') else jax.ShapedArray(x.shape, x.dtype)
      for x in tree_leaves
  ]
  positional_semantics = [_PositionalSemantics.LOCAL for _ in local_avals]
  global_avals = pjit.local_to_global(positional_semantics, mesh, local_avals,
                                      axis_resources)
  local_mesh = mesh.local_mesh
  # For each input array, get local/global SliceNdArrays indicating which
  # local/global slice is stored in each device of the local/global mesh.
  local_slices_arrays = _make_slice_nd_arrays(local_avals, axis_resources,
                                              local_mesh)
  global_slices_arrays = _make_slice_nd_arrays(global_avals, axis_resources,
                                               mesh)
  # `shard_per_global_slices` contains, for each input array, a sequence of
  # integers indicating which shard must write each of the global slices of the
  # corresponding array. Its size is that of the number of different slices of
  # the corresponding input array.
  # `process_per_shard` is a tuple of integers denoting the `process_index` of
  # the process in charge of writing the corresponding checkpoint shard.
  shard_per_global_slices, process_per_shard = _slice_nd_arrays_to_shards(
      global_slices_arrays, mesh.devices, num_shards)
  # It's possible that some shards are empty, if so remove unused shards.
  if num_shards > 0:
    shard_per_global_slices, process_per_shard = _remove_unused_shards(
        shard_per_global_slices, process_per_shard)
  # Create LazyArrayChunks for each shard. A "chunk" is the result of slicing
  # an array with some (local) SliceNd. Instead of chunking the arrays here,
  # we'll chunk them on the fly when the LazyArrayChunks are serialized, to
  # prevent unnecessary copies of the data.
  lazy_array_chunks_per_shard = _create_lazy_array_chunks_per_shard(
      tree_leaves, local_slices_arrays, global_slices_arrays,
      shard_per_global_slices, process_per_shard, mesh, local_mesh)
  shard_count = len(process_per_shard)
  if jax.process_index() == 0:
    # We store an 'index' file using the process_index == 0, which contains
    # global information about the checkpoint:
    #   - The global shape of each array.
    #   - The slices of each array.
    #   - The shard in which each slice is located.
    index_leaves = [
        IndexInfo(  # pylint: disable=g-complex-comprehension
            global_shape=global_shape,
            global_slices=sorted(set(global_slices_array.flatten())),
            shards=shard_per_slice)
        for global_shape, global_slices_array, shard_per_slice in safe_zip(
            global_avals, global_slices_arrays, shard_per_global_slices)
    ]
    filepath_map[prefix + '.index'] = {
        'shard_count': shard_count,
        'index': struct.unflatten(index_leaves),
    }
  # Assign the LazyArrayChunks objects to the corresponding shard filepaths.
  shard_fpath_fn = functools.partial(
      vmoe.checkpoints.base.add_shard_suffix,
      filepath=prefix + '.data',
      shard_count=shard_count)
  for shard, lazy_array_chunks in lazy_array_chunks_per_shard.items():
    filepath_map[shard_fpath_fn(shard=shard)] = lazy_array_chunks

  return filepath_map


def _make_slice_nd_arrays(
    avals: Sequence[jax.ShapedArray],
    parsed_partition_specs: Sequence[ParsedPartitionSpec],
    mesh: Mesh) -> Sequence[SliceNdArray]:
  """Returns a SliceNdArray indicating the array slice stored in each device."""
  # We use pjit/pxla functions to obtain the slice of each aval which is
  # contained in every devices of the mesh.
  # This can be used to map local/global devices to local/global array axes.
  make_sharding_spec_fn = pjit.pxla.mesh_sharding_specs(mesh.shape,
                                                        mesh.axis_names)
  sharding_specs_leaves = [
      make_sharding_spec_fn(aval, pjit.get_array_mapping(spec))
      for aval, spec in zip(avals, parsed_partition_specs)
  ]
  slices_arrays = [
      spec.indices(aval.shape)
      for aval, spec in zip(avals, sharding_specs_leaves)
  ]
  return [
      _convert_native_to_slices_nd_array(x, aval)
      for x, aval in zip(slices_arrays, avals)
  ]


def _match_checkpoint_to_local_slices(
    local_global_slices: Iterable[Tuple[SliceNd, SliceNd]],
    ckpt_slices_and_shards: Iterable[Tuple[SliceNd, int]],
) -> Iterator[Tuple[int, SliceNd, SliceNd, SliceNd]]:
  """Matches slices in checkpoints to local slices.

  When a checkpoint is written, a given array might be partitioned in a certain
  way which differs from the target partitioning when the checkpoint is
  restored (e.g. if a model is fine-tuned on a different hardware topology).

  We must map the global slices that the local devices hold with the slices in
  the checkpoint files. Notice that, because the partitioning is different, this
  is not a 1-to-1 mapping. If a global slice intersects with a checkpoint slice,
  then part of the array in the checkpoint needs to be restored to fill part of
  the local array.

  This function yields (shard, ckpt_slice, ckpt_subslice, local_subslice),
  respectively denoting:
    - The checkpoint shard to be restored.
    - The (global) checkpoint slice that needs to be restored from the shard.
    - The (sub)slice in the checkpoint chunk that needs to be copied to
    - the (sub)slice in the local array.

  Args:
    local_global_slices: Iterable of (local slice, global slice).
    ckpt_slices_and_shards: Iterable of (ckpt slice, shard).

  Yields:
    A sequence of tuples (int, SliceNd, SliceNd, SliceNd).
  """
  ckpt_slices_and_shards = list(ckpt_slices_and_shards)
  for local_slice, global_slice in local_global_slices:
    matched_in_ckpt = False
    for ckpt_slice, ckpt_shard in ckpt_slices_and_shards:
      intersection = _intersect_slice_nd(ckpt_slice, global_slice, local_slice)
      if intersection is not None:
        matched_in_ckpt = True
        ckpt_subslice, local_subslice = intersection
        yield ckpt_shard, ckpt_slice, ckpt_subslice, local_subslice
    if not matched_in_ckpt:
      raise ValueError(
          f'The global slice {global_slice} does not intersect with any '
          f'slice available in the checkpoint:\n'
          f'{[s for s, _ in ckpt_slices_and_shards]}')


def _pair_local_and_global_slices(
    local_slices_arrays: Sequence[SliceNdArray],
    global_slices_arrays: Sequence[SliceNdArray],
    mesh: Mesh,
    local_mesh: Optional[Mesh] = None,
) -> Iterator[Sequence[Tuple[SliceNd, SliceNd]]]:
  """Returns an iterator over sets of pairs (local SliceNd, global SliceNd).

  Args:
    local_slices_arrays: Sequence of SliceNdArray indicating how each input
      array is partitioned across the local (for the current process) mesh of
      devices.
    global_slices_arrays: Sequence of SliceNdArray indicating how each input
      array is partitioned across the global mesh of devices.
    mesh: Global mesh of devices.
    local_mesh: Local (restricted to a given process) mesh of devices.

  Returns:
    Iterator over sequences of (global sliceNd, local sliceNd).
  """
  local_mesh = local_mesh or mesh.local_mesh
  global_devices = mesh.devices.flatten()
  local_devices = local_mesh.devices.flatten()

  def _fn(local_slices_array, global_slices_array):
    local_slices_array = local_slices_array.flatten()
    global_slices_array = global_slices_array.flatten()
    # Map from global SliceNd to device ID.
    device2global = dict(
        safe_map(lambda s, d: (d.id, s), global_slices_array, global_devices))
    # Create a set of pairs (local SliceNd, global SliceNd).
    return set(safe_map(lambda s, d: (s, device2global[d.id]),
                        local_slices_array, local_devices))

  return safe_map(_fn, local_slices_arrays, global_slices_arrays)


def _remove_unused_shards(
    shard_per_slices: Sequence[Sequence[int]],
    process_per_shard: Tuple[int, ...],
) -> Tuple[Sequence[Sequence[int]], Tuple[int, ...]]:
  """Removes unused shards from the inputs."""
  old_to_new_shard_map = {}
  process_per_shard_new = []
  shard_per_slices_new = []
  for shard_per_slice in shard_per_slices:
    shard_per_slice_new = []
    for shard_index in shard_per_slice:
      if shard_index in old_to_new_shard_map:
        new_shard_index = old_to_new_shard_map[shard_index]
      else:
        new_shard_index = len(old_to_new_shard_map)
        old_to_new_shard_map[shard_index] = new_shard_index
        process_per_shard_new.append(process_per_shard[shard_index])
      shard_per_slice_new.append(new_shard_index)
    shard_per_slices_new.append(shard_per_slice_new)
  return shard_per_slices_new, tuple(process_per_shard_new)


def _restore_array_chunks(
    filepath: str,
    local_arrays: Sequence[np.ndarray],
    array_slices_to_restore: Mapping[int, Iterable[Tuple[SliceNd, SliceNd,
                                                         SliceNd]]],
):
  """Restores array chunks from a checkpoint file, filling local arrays.

  Args:
    filepath: Filepath of the file to restore from. This is typically the
      filepath of a checkpoint shard.
    local_arrays: Sequence of Numpy arrays containing the local arrays.
    array_slices_to_restore: Mapping from array indices to a sequence of
      (ckpt_slice_nd, ckpt_subslice_nd, local_subslice_nd), where ckpt_slice_nd
      is the checkpoint SliceNd to restore, and ckpt_subslice_nd is a SliceNd
      representing a subslice in the checkpoint chunk to copy to the subslice
      local_subslice_nd of the corresponding local array.
  """
  array_chunks = vmoe.checkpoints.base.restore_checkpoint(
      filepath, ArrayChunks())
  for index, slices_to_restore in array_slices_to_restore.items():
    slices_to_restore = {
        k: list(v) for k, v in itertools.groupby(
            sorted(slices_to_restore), key=lambda x: x[0])
    }
    for ckpt_chunk, ckpt_slice in array_chunks.iter_chunks(index):
      for _, ckpt_chunk_slice, local_chunk_slice in slices_to_restore.get(
          ckpt_slice, []):
        local_chunk_slice = tuple(s.slice for s in local_chunk_slice)
        ckpt_chunk_sliced = ckpt_chunk_slice.chunk(ckpt_chunk)
        local_arrays[index][local_chunk_slice] = ckpt_chunk_sliced


def _slice_nd_arrays_to_shards(
    slice_nd_arrays: Sequence[SliceNdArray],
    devices: np.ndarray,
    num_shards: int) -> Tuple[Sequence[Sequence[int]], Tuple[int, ...]]:
  """Returns the shards used to store each slice, for each SliceNdArray.

  Example:
    Suppose that an array with shape (8, 4), is sliced on a (2, 2) mesh as
    specified by the following SliceNdArray object:
    [
      [(Slice(0, 4), Slice()), (Slice(0, 4), Slice())],
      [(Slice(4, 8), Slice()), (Slice(4, 8), Slice())],
    ]
    Notice that the first axis has two slices across the first mesh axis, and
    the second axis is replicated.

    Suppose that each device is handled by a different process, and they are
    arranged in the mesh as follows (each device represented by their ID):
    [[0, 2],
     [1, 3]]

    Now, suppose that we want to checkpoint the array using 2 checkpoint shards.
    Then, the output array will be [0, 1], indicating that the first slice
    (i.e. (Slice(0, 4), Slice())) will be stored in the shard=0, and the second
    slice (i.e. (Slice(4, 8), Slice())) will be saved in the next shard=1.

  Args:
    slice_nd_arrays: A sequence of SliceNdArray.
    devices: Numpy array representing the logical mesh of devices.
    num_shards: (Tentative) number of checkpoint shards to use. The slices will
      be roughly uniformly distributed among these. If `num_shards <= 0` then,
      the minimum number of shards necessary is used.

  Returns:
    - A list of the same size as the input list (i.e. number of arrays),
      where each element is a list of integers denoting which shard is used to
      store the corresponding slice.
    - A tuple containing the process_index that handles each shard.
  """
  devices = devices.flatten()
  # Each checkpoint shard is handled by a different process, we assign them in a
  # round-robin fashion. We also keep track of the number of slices stored in
  # each shard. A slice can only be stored in a shard if that slice is allocated
  # in some device of the process handling the shard.
  shards = [(s % jax.process_count(), 0) for s in range(num_shards)]
  # This function is used to find the best shard to store an array slice,
  # The viable shards are those that are handled by one of the processes given
  # by `process_indices`. We select the viable shard with the smallest number of
  # slices. If no viable shard is found, we simply create a new shard handled by
  # one of the specified processes (that with the smallest ID).
  def find_best_viable_shard_index(process_indices):
    viable_shards = [
        (c, s) for s, (p, c) in enumerate(shards) if p in process_indices
    ]
    if viable_shards:
      # Take the viable shard with the least number of slices.
      return sorted(viable_shards)[0][1]
    else:
      # Create a new shard is handled by one of the passed processes.
      p = min(process_indices)
      shards.append((p, 0))
      return len(shards) - 1

  shard_per_slices = []
  for slice_nd_array in slice_nd_arrays:
    slice_nd_array = slice_nd_array.flatten()
    # Map each slice of the array to a list of processes that hold it.
    # Note: an `array_slice` is a `SliceNd` object.
    processes_per_slice = collections.defaultdict(list)
    for slice_nd, device in safe_zip(slice_nd_array, devices):
      processes_per_slice[slice_nd].append(device.process_index)
    # Now, for each slice_nd, find the best shard to handle it.
    shard_per_slice = []
    for _, processes in sorted(processes_per_slice.items()):
      s = find_best_viable_shard_index(set(processes))
      shards[s] = (shards[s][0], shards[s][1] + 1)  # Update num_slices.
      shard_per_slice.append(s)  # Assign i-th slice to the s-th shard.
    # For each input array, we have a new integer array with as many elements
    # as slices the array has, denoting which shard will store each slice.
    shard_per_slices.append(shard_per_slice)

  process_per_shard = tuple(p for (p, _) in shards)
  return shard_per_slices, process_per_shard
