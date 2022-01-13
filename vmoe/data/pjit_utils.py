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

"""Module with input pipeline functions specific for pjit."""
import collections
import itertools
from typing import Any, Iterator, Optional

import jax
from jax.experimental import maps
from jax.experimental import pjit

Mesh = maps.Mesh
PyTree = Any


def prefetch_to_device(
    iterator: Iterator[PyTree],
    axis_resources: PyTree,
    size: int,
    mesh: Optional[Mesh] = None,
) -> Iterator[PyTree]:
  """Transfers array shards to specified devices and form a ShardedDeviceArray.

  This utility takes an iterator and returns a new iterator which fills an on
  device prefetch buffer. Eager prefetching can improve the performance of
  training loops significantly by overlapping compute and data transfer.
  This is similar to `flax.jax_utils.prefetch_to_device` but works with `pjit`.

  Args:
    iterator: An iterator that retuns a PyTree of ndarrays.
    axis_resources: A PyTree with the same structure as those returned from the
      iterator, specifying how the ndarrays are partitioned.
    size: The size of the prefetch buffer.
    mesh: If given, shards the arrays using this mesh. If None, uses the active
      mesh.

  Yields:
    The original items from the iterator where each ndarray is now a sharded as
    specified by `axis_resources`.
  """
  if not size or size < 1:
    # If size is None, 0 or negative, simply return the original values.
    for data in iterator:
      yield data
    return

  mesh = mesh or maps.thread_resources.env.physical_mesh
  local_mesh = mesh.local_mesh
  local_devices = mesh.local_devices
  sharding_spec_fn = jax.pxla.mesh_sharding_specs(local_mesh.shape,
                                                  local_mesh.axis_names)
  # Parse PartitionSpecs and get the corresponding array axis mapping.
  def _get_array_mapping(spec):
    spec = pjit.ParsedPartitionSpec.from_user_input(spec, 'prefetch_to_device')
    return pjit.get_array_mapping(spec)
  axis_mapping = jax.tree_map(_get_array_mapping, axis_resources)
  # Get first item to compute some (assumed to be) constant values.
  first = next(iterator)

  def _get_shaped_array(x):
    return jax.ShapedArray(x.shape, jax.dtypes.canonicalize_dtype(x.dtype))
  local_avals = jax.tree_map(_get_shaped_array, first)
  sharding_specs = jax.tree_map(sharding_spec_fn, local_avals, axis_mapping)
  indices = jax.tree_map(lambda a, s: jax.pxla.spec_to_indices(a.shape, s),
                         local_avals, sharding_specs)
  # Put the first item back to the iterator.
  iterator = itertools.chain([first], iterator)
  del first  # Remove reference to this, so that memory can be freed.

  # Next, we fill items to this queue, and pop from it when a new item is
  # yielded.
  queue = collections.deque()

  @jax.profiler.annotate_function
  def _prefetch(x, local_aval, sharding_spec, indices):
    device_buffers = jax.pxla._shard_arg(x, local_devices, indices)  # pylint: disable=protected-access
    return jax.pxla.make_sharded_device_array(
        local_aval, sharding_spec, device_buffers, indices)

  def enqueue(n):
    for data in itertools.islice(iterator, n):
      queue.append(
          jax.tree_map(_prefetch, data, local_avals, sharding_specs, indices))

  enqueue(size)
  while queue:
    yield queue.popleft()
    enqueue(1)
