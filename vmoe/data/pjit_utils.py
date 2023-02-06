# Copyright 2022 Google LLC.
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
from typing import Any, Iterator, Optional, Union

from clu.data import dataset_iterator
import jax
from jax.experimental import maps
from jax.experimental import multihost_utils
from jax.experimental import pjit

Mesh = jax.sharding.Mesh
PyTree = Any
PartitionSpec = Union[jax.sharding.PartitionSpec, PyTree]


def prefetch_to_device(
    iterator: Iterator[PyTree],
    axis_resources: PyTree,
    size: int,
    mesh: Optional[Mesh] = None,
) -> Iterator[PyTree]:
  """Iterates data and transfers it to devices creating jax.Arrays.

  This utility takes an iterator and returns a new iterator which fills an on
  device prefetch buffer. Eager prefetching can improve the performance of
  training loops significantly by overlapping compute and data transfer.
  This is similar to `flax.jax_utils.prefetch_to_device` but works with `pjit`.

  Args:
    iterator: An iterator that returns a PyTree of ndarrays.
    axis_resources: A PyTree with the same structure as those returned from the
      iterator, specifying how the ndarrays are partitioned.
    size: The size of the prefetch buffer.
    mesh: If given, shards the arrays using this mesh. If None, uses the active
      mesh.

  Yields:
    The original items from the iterator where each ndarray is now sharded as
    specified by `axis_resources`.
  """
  mesh = mesh or maps.thread_resources.env.physical_mesh
  assert mesh is not None and not mesh.empty, f'No mesh or empty mesh. {mesh=}'

  def _to_global(data):
    return multihost_utils.host_local_array_to_global_array(
        data, mesh, axis_resources)

  if size and size > 0:
    # We fill items to this queue, and pop from it when a new item is yielded.
    queue = collections.deque()

    def enqueue(n):
      for data in itertools.islice(iterator, n):
        queue.append(_to_global(data))

    enqueue(size)
    while queue:
      yield queue.popleft()
      enqueue(1)
  else:
    # If size is None, 0 or negative, simply create jax.Arrays without
    # prefetching.
    for data in iterator:
      yield _to_global(data)


def get_dataset_shape_dtype_struct(
    iterator: dataset_iterator.DatasetIterator,
    pspec: PartitionSpec,
    mesh: Optional[Mesh] = None,
) -> PyTree:
  """Returns the jax.ShapeDtypeStruct."""
  mesh = mesh or maps.thread_resources.env.physical_mesh
  assert mesh is not None and not mesh.empty, f'No mesh or empty mesh. {mesh=}'

  def fn(x, s):
    # Dtype and local shape (of this particular process) of the given array x.
    shape, dtype = x.shape, x.dtype
    dtype = dtype.as_numpy_dtype if hasattr(dtype, 'as_numpy_dtype') else dtype
    # Get the global shape of the array, given the mesh and the pspec s.
    shape = multihost_utils._local_to_global_aval(  # pylint: disable=protected-access
        jax.ShapedArray(shape, dtype), mesh, s).shape
    sharding = maps.NamedSharding(mesh, s)
    # Return a ShapeDtypeStruct with the global shape and sharding.
    return jax.ShapeDtypeStruct(shape=shape, dtype=dtype, sharding=sharding)

  if isinstance(pspec, pjit.PartitionSpec):
    return jax.tree_util.tree_map(lambda x: fn(x, pspec), iterator.element_spec)
  else:
    return jax.tree_util.tree_map(fn, iterator.element_spec, pspec)
