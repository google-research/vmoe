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

"""Utilities for handling the partitioning of expert models.

Main concepts:

- Hardware Mesh: A numpy array indicating how devices are physically arranged
  in hardware. This is either a 4d array (TPUs: core_on_chip, x, y, z) or a
  2d array (GPUs: device_on_process, process).
- Logical Mesh: A JAX pxla.Mesh indicating how devices are logically arranged
  into the axes ('expert', 'replica').

For a model with `num_experts` experts, we can partition the experts into
`num_partitions` as long as `num_partitions` is a divisor of `num_experts`, and
`num_partitions` can be factored as a product of divisors of each axis of the
hardware mesh.

Examples:

Suppose that we are running a model on a TPUv3 with 8 TPU cores. Then, the
devices (0..7) are arranged in a hardware mesh like this (notice that the
hardware mesh contains the Device objects, but here we represent only their
IDs):

  [[[[0], [4]], [[2], [6]]],
   [[[1], [5]], [[3], [7]]]]

Now, suppose that the model has 32 experts, we can decide to partition the
32 experts into any divisor. Because we have 8 devices, it makes sense to
use 8 partitions, so that each device handles 4 experts. Then, the logical mesh
would be a Mesh object, with the following `devices` array (again, only the
device IDs are represented here):

  [[0],
   [1],
   [2],
   [3],
   [4],
   [5],
   [6],
   [7]]

Another option would be to partition the 32 experts into 4 partitions. Then, the
logical mesh would be (notice that here we have two replicas for each partition,
and each partition will contain 32 / 4 = 8 experts):

  [[0, 1],
   [2, 3],
   [4, 5],
   [6, 7]]
"""
import functools
from typing import Any, Optional, Sequence, Tuple, Union

from absl import logging
import jax
from jax.experimental import maps
from jax.experimental import pjit
import numpy as np

Device = jax.lib.xla_client.Device
Mesh = maps.Mesh
PartitionSpec = pjit.PartitionSpec
PyTree = Any
TpuCoords = Tuple[int, int, int, int]
OtherCoords = Tuple[int, int]


def process_has_contiguous_device_slice(devices: np.ndarray,
                                        process_index: int) -> bool:
  """Checks if the devices of a process form a contiguous slice in the mesh."""
  is_local_device = np.vectorize(
      lambda device: device.process_index == process_index, otypes=[bool])(
          devices)
  # Shape is (num_axes, num_nonzero_elements).
  local_device_indices = np.asarray(np.nonzero(is_local_device))
  # Take a slice that covers all local devices, and checks that no other device
  # in that slice belongs to a different process.
  s = tuple(map(lambda i, j: slice(i, j+1),
                np.min(local_device_indices, axis=1),
                np.max(local_device_indices, axis=1)))
  return is_local_device[s].all()


def processes_have_contiguous_device_slices(devices: np.ndarray) -> bool:
  """Checks if the devices of all process form contiguous slices in the mesh."""
  process_indices = set(device.process_index for device in devices.flatten())
  return all(
      process_has_contiguous_device_slice(devices, process_index)
      for process_index in process_indices)


def get_auto_logical_mesh(
    num_partitions: int,
    devices: Optional[Sequence[Device]] = None) -> Mesh:
  """Returns a heuristic logical mesh with axes ('expert', 'replica')."""
  devices = devices or jax.devices()
  if devices[0].platform == 'tpu':
    return get_auto_logical_mesh_tpu(num_partitions,
                                     get_hardware_mesh_tpu(devices))
  else:
    return get_auto_logical_mesh_other(num_partitions,
                                       get_hardware_mesh_other(devices))


def get_auto_logical_mesh_other(num_partitions: int,
                                hardware_mesh: np.ndarray) -> Mesh:
  """Returns a heuristic logical mesh on CPU/GPU."""
  assert hardware_mesh.ndim == 2, f'hardware_mesh.ndim = {hardware_mesh.ndim} != 2'
  hardware_mesh_shape = hardware_mesh.shape
  # Put as many partitions as possible within devices of a single process, to
  # prevent across-process communication as much as possible.
  d = min(num_partitions, hardware_mesh_shape[0])
  h = min(num_partitions // d, hardware_mesh_shape[1])
  if (d * h != num_partitions or
      any(s % p != 0 for p, s in zip((d, h), hardware_mesh_shape))):
    raise ValueError(
        f'The hardware mesh with shape {hardware_mesh_shape} cannot be '
        f'automatically partitioned into {num_partitions}. The number of '
        f'partitions must be factored in divisors of {hardware_mesh_shape}.')
  return get_logical_mesh((d, h), hardware_mesh)


def get_auto_logical_mesh_tpu(num_partitions: int,
                              hardware_mesh: np.ndarray) -> Mesh:
  """Returns a heuristic logical mesh with axes ('expert', 'replica') on TPU."""
  assert hardware_mesh.ndim == 4, f'hardware_mesh.ndim = {hardware_mesh.ndim} != 4'
  hardware_mesh_shape = hardware_mesh.shape
  if hardware_mesh_shape[3] == 1:
    # TPUv2 and TPUv3: A good rule of thumb for best speed is to place as
    # many experts as possible along the x-axis, then y-axis, and finally the
    # core-axis. Good average performance across ViT-{B,L,H} + Experts.
    z = 1
    x = min(num_partitions, hardware_mesh_shape[1])
    y = min(num_partitions // x, hardware_mesh_shape[2])
    c = min(num_partitions // (x * y), hardware_mesh_shape[0])
  else:
    # TPUv4: Place as many experts as possible along the z-axis, then y, then x,
    # and finally along the core-axis. Good average performance across
    # ViT-{B,L,H} + Experts.
    z = min(num_partitions, hardware_mesh_shape[3])
    y = min(num_partitions // z, hardware_mesh_shape[2])
    x = min(num_partitions // (z * y), hardware_mesh_shape[1])
    c = min(num_partitions // (z * y * x), hardware_mesh_shape[0])
  if (c * x * y * z != num_partitions or
      any(s % p != 0 for p, s in zip((c, x, y, z), hardware_mesh_shape))):
    raise ValueError(
        f'The hardware mesh with shape {hardware_mesh_shape} cannot be '
        f'automatically partitioned into {num_partitions}. The number of '
        f'partitions must be factored in divisors of {hardware_mesh_shape}.')
  return get_logical_mesh((c, x, y, z), hardware_mesh)


def get_device_coords(device: Device) -> Union[TpuCoords, OtherCoords]:
  if device.platform == 'tpu':
    return get_device_coords_tpu(device)
  else:
    return get_device_coords_other(device)


def get_device_coords_other(device: Device) -> OtherCoords:
  """Returns the (device_on_process, process) coordinates of a CPU/GPU device."""
  return (device.id % jax.local_device_count(), device.process_index)


def get_device_coords_tpu(device: Device) -> TpuCoords:
  """Returns the (core_on_chip, x, y, z) coordinates of a TPU device."""
  assert device.platform == 'tpu', f'{device!r} is not a TPU'
  assert hasattr(device, 'core_on_chip'), f'{device!r} lacks "core_on_chip"'
  assert hasattr(device, 'coords'), f'{device!r} lacks "coords"'
  core_on_chip = int(device.core_on_chip)
  coords = tuple(map(int, device.coords))
  return (core_on_chip, *coords)


def get_hardware_mesh_local_shape(
    local_devices: Optional[Sequence[Device]] = None) -> Tuple[int, ...]:
  local_devices = local_devices or jax.local_devices()
  coords = np.asarray(tuple(map(get_device_coords, local_devices))).transpose()
  return tuple(len(set(c)) for c in coords)


def get_hardware_mesh_other(devices: Sequence[Device]) -> np.ndarray:
  """Returns a 2-dim array with the CPU/GPU hardware mesh."""
  mesh_dict = {get_device_coords_other(device): device for device in devices}
  nd, nh = map(lambda x: x + 1, sorted(mesh_dict.keys())[-1])
  mesh = np.empty((nd, nh), dtype=np.object)
  for (d, h), device in mesh_dict.items():
    mesh[(d, h)] = device
  return mesh


def get_hardware_mesh_tpu(devices: Sequence[Device]) -> np.ndarray:
  """Returns a 4-dim array with the TPU hardware mesh."""
  mesh_dict = {get_device_coords_tpu(device): device for device in devices}
  nc, nx, ny, nz = map(lambda x: x + 1, sorted(mesh_dict.keys())[-1])
  mesh = np.empty((nc, nx, ny, nz), dtype=np.object)
  for (c, x, y, z), device in mesh_dict.items():
    mesh[(c, x, y, z)] = device
  return mesh


def get_logical_mesh(partitions: Tuple[int, ...],
                     hardware_mesh: np.ndarray) -> Mesh:
  """Maps a hardware mesh to a logical mesh with axes ('expert', 'replica')."""
  # Number of replicas in each dimension of the hardware mesh.
  replicas = tuple(
      s // p for p, s in jax.util.safe_zip(partitions, hardware_mesh.shape))
  # Transpose hardware axes to (Z, Y, X, C) / (H, D) for TPU / other devices.
  replicas = tuple(reversed(replicas))
  partitions = tuple(reversed(partitions))
  hardware_axes_order = tuple(reversed(range(hardware_mesh.ndim)))
  hardware_mesh = hardware_mesh.transpose(hardware_axes_order)
  logical_mesh = get_logical_mesh_default(partitions, replicas, hardware_mesh)
  if not processes_have_contiguous_device_slices(logical_mesh.devices):
    hardware_mesh_local_shape = tuple(reversed(get_hardware_mesh_local_shape()))
    logical_mesh = get_logical_mesh_tile_by_process(partitions, replicas,
                                                    hardware_mesh,
                                                    hardware_mesh_local_shape)
  return logical_mesh


def get_logical_mesh_default(
    partitions: Tuple[int, ...],
    replicas: Tuple[int, ...],
    hardware_mesh: np.ndarray) -> Mesh:
  """Returns the default logical mesh."""
  # Split each axis of the hardware mesh in (partitions, replicas).
  # For TPUs: (Z, Y, X, C) -> (Z_P, Z_R, Y_P, Y_R, X_P, X_R, C_P, C_R).
  # For other devices: (H, D) -> (H_P, H_R, D_P, D_R).
  shape = functools.reduce(lambda a, b: a + b, zip(partitions, replicas))
  devices = hardware_mesh.reshape(shape)
  # Put all partition axes first, then all replica axes.
  devices = devices.transpose(tuple(range(0, 2 * hardware_mesh.ndim, 2)) +
                              tuple(range(1, 2 * hardware_mesh.ndim, 2)))
  # Reshape devices to (num_partitions, num_replicas).
  num_partitions = np.prod(partitions)
  num_replicas = np.prod(replicas)
  devices = devices.reshape((num_partitions, num_replicas))
  return Mesh(devices=devices, axis_names=('expert', 'replica'))


def get_logical_mesh_tile_by_process(
    partitions: Tuple[int, ...],
    replicas: Tuple[int, ...],
    hardware_mesh: np.ndarray,
    hardware_mesh_local_shape: Tuple[int, ...]) -> Mesh:
  """Returns a logical mesh where all process' devices form a contiguous slice."""
  # Split each axis of the hardware mesh in
  # (partitions_process, partitions_device, replica_process, replica_device).
  # For TPUs: (Z, Y, X, C) -> (Z_P_H, Z_P_D, Z_R_H, Z_R_D, Y_P_H, ...).
  def fn(p, r, l):
    # (p, r) are the number of (partitions, replicas) across a given axis.
    # l is the number of devices per process across a given axis.
    if p % l == 0:
      return p // l, l, r, 1
    elif r % l == 0:
      return 1, p, r // l, l
    else:
      raise ValueError(
          f'Neither p = {p}, nor r = {r} are multiples of l = {l}')
  shape = functools.reduce(
      lambda a, b: a + b,
      map(fn, partitions, replicas, hardware_mesh_local_shape))
  devices = hardware_mesh.reshape(shape)
  # Put all partition_process axes first, then partition_devices axes, then
  # all replica_process axes, and finally all replica_device axes.
  devices = devices.transpose(tuple(range(0, 4 * hardware_mesh.ndim, 4)) +
                              tuple(range(1, 4 * hardware_mesh.ndim, 4)) +
                              tuple(range(2, 4 * hardware_mesh.ndim, 4)) +
                              tuple(range(3, 4 * hardware_mesh.ndim, 4)))
  # Reshape devices to (num_partitions, num_replicas).
  num_partitions = np.prod(partitions)
  num_replicas = np.prod(replicas)
  devices = devices.reshape((num_partitions, num_replicas))
  return Mesh(devices=devices, axis_names=('expert', 'replica'))


def log_logical_mesh(mesh: Mesh, *, logger=logging):
  """Prints the logical device mesh to the logs."""
  logger.info('Logical device mesh has axis_names = %r', mesh.axis_names)
  logger.info('Logical device mesh has shape = %r', tuple(mesh.shape.values()))
  # Compute the number of digits necessary to represent each dimension of the
  # device coordinates.
  def _ndig(x):
    return int(np.log10(x) + 1)
  # Format coordinates of each device in the mesh.
  coords = [get_device_coords(device) for device in mesh.devices.flatten()]
  coord_fmt = [
      f'{{:>{_ndig(mc + 1)}}}' for mc in np.max(np.asarray(coords), axis=0)
  ]
  coords = [
      '(' + ', '.join(f.format(c) for c, f in zip(coord, coord_fmt)) + ')'
      for coord in coords
  ]
  coords = np.array(coords, dtype=np.object)
  coords = coords.reshape(mesh.devices.shape)
  # Format process_index of each device in the mesh.
  process_index = np.vectorize(
      lambda device: device.process_index, otypes=[int])(mesh.devices)
  process_index_fmt = f'[{{:>{_ndig(np.max(process_index) + 1)}}}]'
  process_index = np.vectorize(
      process_index_fmt.format, otypes=[np.object])(process_index)
  ndim = mesh.devices.ndim
  if ndim == 1:
    coords = np.expand_dims(coords, axis=-1)
    process_index = np.expand_dims(process_index, axis=-1)
    ndim = 2
  if ndim == 2:
    def row(i):
      return ' '.join([f'{c}{p}' for c, p in zip(coords[i], process_index[i])])
    row_length = len(row(0))
    logger.info('Logical device mesh:')
    logger.info('%s', '+-' + ('-' * row_length) + '-+')
    for i in range(coords.shape[0]):
      logger.info('| %s |', row(i))
    logger.info('%s', '+-' + ('-' * row_length) + '-+')


def tree_global_shape(tree: PyTree, axis_resources: PyTree,
                      mesh: Mesh) -> PyTree:
  """Returns a PyTree of ShapedArray leaves with the global shape of the arrays in the input tree."""
  tree_leaves, struct = jax.tree_flatten(tree)
  # pylint: disable=protected-access
  _, axis_resources_leaves, struct2 = pjit._prepare_axis_resources(
      axis_resources, 'axis_resources')
  if struct != struct2:
    raise ValueError(f'The tree structs do not match.\n'
                     f'tree: {struct}\n'
                     f'resource_axis: {struct2}')
  if not all(
      hasattr(x, 'aval') or (hasattr(x, 'shape') and hasattr(x, 'dtype'))
      for x in tree_leaves):
    raise ValueError(
        "All leaves in the input tree must have an 'aval', or 'shape' and "
        "'dtype' attributes.")
  tree_leaves = [
      x.aval if hasattr(x, 'aval') else jax.ShapedArray(x.shape, x.dtype)
      for x in tree_leaves
  ]
  positional_semantics = [maps._PositionalSemantics.LOCAL for _ in tree_leaves]
  global_aval_leaves = pjit.local_to_global(
      positional_semantics, mesh, tree_leaves, axis_resources_leaves)
  return jax.tree_unflatten(struct, global_aval_leaves)
  # pylint: enable=protected-access


def with_sharding_constraint(x: PyTree, partition_spec: PartitionSpec):
  """Specifies a partition_spec for a given array to help pjit's sharding."""
  if maps.thread_resources.env.physical_mesh.empty or partition_spec is None:
    return x
  else:
    return pjit.with_sharding_constraint(x, partition_spec)
