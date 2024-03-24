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

"""Module with input pipeline functions specific for pjit."""
import collections
import itertools
from typing import Any, Iterator, List, Optional, Sequence, Union

from absl import logging
from clu.data import dataset_iterator
import jax
from jax.interpreters import pxla
import numpy as np
import tensorflow as tf

Mesh = jax.sharding.Mesh
PyTree = Any


def get_dataset_shape_dtype_struct(
    iterator: Union[tf.data.Dataset, dataset_iterator.DatasetIterator],
    mesh: Optional[Mesh] = None,
) -> PyTree:
  """Returns the jax.ShapeDtypeStruct."""
  mesh = mesh or pxla.thread_resources.env.physical_mesh
  assert mesh is not None and not mesh.empty, f'No mesh or empty mesh. {mesh=}'

  pspec = jax.sharding.PartitionSpec(mesh.axis_names,)
  sharding = jax.sharding.NamedSharding(mesh, pspec)

  def fn(x):
    # Dtype and local shape (of this particular process) of the given array x.
    shape, dtype = x.shape, x.dtype
    dtype = dtype.as_numpy_dtype if hasattr(dtype, 'as_numpy_dtype') else dtype
    # Global shape
    shape = (shape[0] * jax.process_count(),) + shape[1:]
    # Return a ShapeDtypeStruct with the global shape and sharding.
    return jax.ShapeDtypeStruct(shape=shape, dtype=dtype, sharding=sharding)

  return jax.tree_util.tree_map(fn, iterator.element_spec)


def prefetch_to_device(
    iterator: Iterator[PyTree],
    size: int,
    mesh: Optional[Mesh] = None,
) -> Iterator[PyTree]:
  """Iterates data and transfers it to devices creating jax.Arrays.

  This utility takes an iterator and returns a new iterator which fills a
  device prefetch buffer. Eager prefetching can improve the performance of
  training loops significantly by overlapping compute and data transfer.
  This is similar to `flax.jax_utils.prefetch_to_device` but works with `pjit`.

  Args:
    iterator: An iterator that returns a PyTree of ndarrays.
    size: The size of the prefetch buffer.
    mesh: If given, shards the arrays using this mesh. If None, uses the active
      mesh.

  Yields:
    The original items from the iterator where each ndarray is now sharded as
    specified by `axis_resources`.
  """
  mesh = mesh or pxla.thread_resources.env.physical_mesh
  assert mesh is not None and not mesh.empty, f'No mesh or empty mesh. {mesh=}'

  pspec = jax.sharding.PartitionSpec(mesh.axis_names,)
  sharding = jax.sharding.NamedSharding(mesh, pspec)

  local_devices = mesh.local_devices

  def _to_global(x):
    if isinstance(x, tf.RaggedTensor):
      logging.log_first_n(
          logging.WARNING,
          'A RaggedTensor cannot with dtype=%r and shape=%r cannot be '
          'converted to a jax.Array.', 1, x.dtype, x.shape)
      return x
    # View x as a numpy array (in case it's a TF tensor).
    x = np.asarray(memoryview(x))
    if not np.issubdtype(x.dtype, np.number) and x.dtype != bool:
      logging.log_first_n(
          logging.WARNING,
          'A Numpy array with dtype=%r and shape=%r cannot be converted to a '
          'jax.Array.', 1, x.dtype, x.shape)
      return x
    device_buffers = put_to_devices(x, local_devices)
    global_shape = (x.shape[0] * jax.process_count(),) + x.shape[1:]
    return jax.make_array_from_single_device_arrays(
        global_shape, sharding, device_buffers)

  if size and size > 0:
    # We fill items to this queue, and pop from it when a new item is yielded.
    queue = collections.deque()

    def enqueue(n):
      for data in itertools.islice(iterator, n):
        queue.append(jax.tree_util.tree_map(_to_global, data))

    enqueue(size)
    while queue:
      yield queue.popleft()
      enqueue(1)
  else:
    # If size is None, 0 or negative, simply create jax.Arrays without
    # prefetching.
    for data in iterator:
      yield jax.tree_util.tree_map(_to_global, data)


def put_to_devices(host_array: np.ndarray,
                   local_devices: Sequence[Any]) -> List[Any]:
  """Transfers a host array to the local devices, sharding the first axis entirely."""
  local_device_count = len(local_devices)
  try:
    per_device_arrays = np.split(host_array, local_device_count, axis=0)
  except ValueError as array_split_error:
    raise ValueError(
        f'Unable to put to devices shape {host_array.shape} with '
        f'local device count {local_device_count}') from array_split_error
  device_buffers = [
      jax.device_put(arr, d) for arr, d in zip(per_device_arrays, local_devices)
  ]
  return device_buffers
