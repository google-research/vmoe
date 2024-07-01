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

"""Functions to initialize a TrainState from pre-trained checkpoints."""
import json
import multiprocessing.pool
import os
from typing import Any, Optional, Union

from etils import epath
import flax.serialization
import flax.traverse_util
import jax
from jax.experimental import pjit
import numpy as np
from orbax import checkpoint as orbax_checkpoint
import tensorflow as tf
from vit_jax import checkpoint as vit_jax_checkpoint
from vmoe import checkpoints as vmoe_checkpoint
from vmoe import partitioning
from vmoe.checkpoints import serialization as vmoe_serialization
from vmoe.initialization import mapping

Mesh = partitioning.Mesh
NamedSharding = jax.sharding.NamedSharding
PartitionSpec = partitioning.PartitionSpec
PyTree = Any
Rules = Union[mapping.Rules, mapping.UnparsedRules]
ShapeDtypeStruct = jax.ShapeDtypeStruct
ThreadPool = multiprocessing.pool.ThreadPool

__all__ = [
    'initialize_from_orbax',
    'initialize_from_vit',
    'initialize_from_vmoe',
]


class AsyncCheckpointerWithStructure(orbax_checkpoint.AsyncCheckpointer):

  def structure(self, directory: epath.PathLike) -> Optional[Any]:
    """See superclass documentation."""
    directory = epath.Path(directory)
    try:
      return self._handler.structure(directory)  # pytype: disable=attribute-error
    except NotImplementedError:
      return


class PyTreeCheckpointHandlerWithStructure(
    orbax_checkpoint.PyTreeCheckpointHandler):

  def structure(self, directory):
    return self._read_aggregate_file(directory)  # pylint: disable=protected-access


def initialize_from_orbax(
    *,
    target: PyTree,
    directory: str,
    rules: Rules,
    mesh: Mesh,
    axis_resources_regexes: Optional[partitioning.AxisResourcesRegexes] = None,
    **map_state_dict_kwargs) -> PyTree:
  """Initializes the target from an Orbax checkpoint.

  Args:
    target: PyTree to initialize. This should not be used again once this
      function returns. Use the returned object instead.
    directory: Directory containing the checkpoint to use for initialization.
    rules: Rules used for mapping variable names from the checkpoint to the
      target.
    mesh: Device mesh used to partition the target PyTree.
    axis_resources_regexes: Optional regexes matching the checkpoint array names
      and specifying how the array read from the checkpoint is partitioned.
      Notice that this is different from the target partitioning, which is
      specified in the target leaves (jax.Array or jax.ShapeDtypeStruct).
    **map_state_dict_kwargs: Additional keyword arguments passed to the
      `mapping.map_state_dict` function.

  Returns:
    A PyTree as the input `target` with (some of) the values loaded from the
    checkpoint.
  """

  # Orbax doesn't provide a way of restoring the shape and dtype of the arrays,
  # so we must use our own code to read the .zarray files that contain that
  # information.
  def _get_shape_dtype_struct(item):
    k, v = item
    if isinstance(v, str) and v.startswith('PLACEHOLDER://'):
      v = v[len('PLACEHOLDER://'):]
      with tf.io.gfile.GFile(os.path.join(directory, v, '.zarray'), 'r') as fp:
        zarray = json.load(fp)
        dtype = zarray['dtype']
        shape = tuple(zarray['shape'])
        return k, jax.ShapeDtypeStruct(shape, jax.numpy.dtype(dtype))
    return k, v

  ckptr = AsyncCheckpointerWithStructure(PyTreeCheckpointHandlerWithStructure())

  # Restore the structure of the checkpoint.
  structure = ckptr.structure(directory)
  # Restore shape and dtype of all arrays. We use a large thread pool since
  # there are typically hundreds/thousands of small files to read (.zarray).
  flat_structure = flax.traverse_util.flatten_dict(
      flax.serialization.to_state_dict(structure),
      sep='/', keep_empty_nodes=True)
  flat_structure = dict(ThreadPool(4096).map(_get_shape_dtype_struct,
                                             flat_structure.items()))
  structure = flax.serialization.from_state_dict(
      structure, flax.traverse_util.unflatten_dict(flat_structure, sep='/'))
  # Get the PartitionSpec corresponding to the arrays to restore, according to
  # the given regular expressions.
  axis_resources = partitioning.tree_axis_resources_from_regexes(
      tree=structure, axis_resources_regexes=axis_resources_regexes or ())

  def _array_restore_args(value, spec):
    if isinstance(value, jax.ShapeDtypeStruct):
      sharding = NamedSharding(mesh, spec)
      return orbax_checkpoint.ArrayRestoreArgs(
          dtype=value.dtype, sharding=sharding, global_shape=value.shape)
    else:
      return orbax_checkpoint.RestoreArgs()

  restore_args = jax.tree_util.tree_map(
      _array_restore_args, structure, axis_resources)
  ckpt = ckptr.restore(directory, restore_args=restore_args)
  return mapping.map_state_dict(ckpt, target, rules, **map_state_dict_kwargs)


def initialize_from_vit(
    *,
    target: PyTree,
    filepath: str,
    rules: Rules,
    mesh: Mesh,
    axis_resources_regexes: Optional[partitioning.AxisResourcesRegexes] = None,
    **map_state_dict_kwargs,
) -> PyTree:
  """Initializes the target from a VisionTransformer checkpoint.

  Args:
    target: PyTree to initialize. This should not be used again once this
      function returns. Use the returned object instead.
    filepath: Filepath of the checkpoint to use for initialization.
    rules: Rules used for mapping variable names from the checkpoint to the
      target.
    mesh: Device mesh used to partition the target PyTree.
    axis_resources_regexes: Optional regexes matching the checkpoint array names
      and specifying how the array read from the checkpoint is partitioned.
      Notice that this is different from the target partitioning, which is
      specified in the target leaves (jax.Array or jax.ShapeDtypeStruct).
    **map_state_dict_kwargs: Additional keyword arguments passed to the
      `mapping.map_state_dict` function.

  Returns:
    A PyTree as the input `target` with (some of) the values loaded from the
    checkpoint.
  """
  # Get the parameters from the checkpoint.
  ckpt = vit_jax_checkpoint.load(filepath)
  # Numpy saves bfloat16 as a void type. Fix that.
  def fix_dtype(x):
    if hasattr(x, 'dtype') and x.dtype.type is np.void:
      assert x.itemsize == 2, 'Unknown dtype!'
      return x.view(jax.numpy.bfloat16)
    else:
      return x
  ckpt = jax.tree_util.tree_map(fix_dtype, ckpt)
  # Copy the checkpoint arrays from host to device, using the partitioning given
  # by axis_resources_regexes. If None, the parameters are simply fully
  # replicated across all devices.
  if axis_resources_regexes is None:
    axis_resources = jax.tree_util.tree_map(
        lambda _: PartitionSpec(), ckpt)
  else:
    axis_resources = partitioning.tree_axis_resources_from_regexes(
        tree=ckpt, axis_resources_regexes=axis_resources_regexes)
  with mesh:
    ckpt = pjit.pjit(
        fun=lambda x: x, in_shardings=None, out_shardings=axis_resources
    )(ckpt)
  # Map the arrays in the ckpt tree to the target tree.
  return mapping.map_state_dict(ckpt, target, rules, **map_state_dict_kwargs)


def initialize_from_vmoe(
    *,
    target: PyTree,
    prefix: str,
    rules: Rules,
    mesh: Mesh,
    axis_resources_regexes: Optional[partitioning.AxisResourcesRegexes] = None,
    thread_pool: Optional[ThreadPool] = None,
    **map_state_dict_kwargs,
) -> PyTree:
  """Initializes the target from a V-MoE checkpoint.

  Args:
    target: PyTree to initialize. This should not be used again once this
      function returns. Use the returned object instead.
    prefix: Filepath of the checkpoint to use for initialization.
    rules: Rules used for mapping variable names from the checkpoint to the
      target.
    mesh: Device mesh used to partition the target PyTree.
    axis_resources_regexes: Optional regexes matching the checkpoint array names
      and specifying how the array read from the checkpoint is partitioned.
      Notice that this is different from the target partitioning, which is
      specified in the target leaves (jax.Array or jax.ShapeDtypeStruct).
    thread_pool: Optional thread pool used to restore checkpoints. This can
      significantly speed-up the time to restore a sharded checkpoint.
    **map_state_dict_kwargs: Additional keyword arguments passed to the
      `mapping.map_state_dict` function.

  Returns:
    A PyTree as the input `target` with (some of) the values loaded from the
    checkpoint.
  """
  # Read only the index and create a PyTree with the same structure but with
  # jax.ShapeDtypeStruct leaves containing the partitioning information for
  # each array read from the checkpoint.
  index = vmoe_checkpoint.restore_checkpoint(prefix + '.index')
  version = index.get('version', vmoe_checkpoint.Version.UNKNOWN)
  shapes = jax.tree_util.tree_map(lambda x: x.global_shape, index['index'])
  if version == vmoe_checkpoint.Version.UNKNOWN:
    target_state_dict = vmoe_serialization.to_state_dict(target)
    if (jax.tree_util.tree_structure(target_state_dict) !=
        jax.tree_util.tree_structure(shapes)):
      raise ValueError(
          'Initialization from V-MoE checkpoints created before 2022/06/22 '
          'is only possible when the structure of the checkpoint and target '
          'trees match.')
    axis_resources = partitioning.tree_axis_resources_from_regexes(
        tree=target, axis_resources_regexes=axis_resources_regexes or ())
    shapes = vmoe_serialization.from_state_dict(target, shapes)
  else:
    axis_resources = partitioning.tree_axis_resources_from_regexes(
        tree=shapes, axis_resources_regexes=axis_resources_regexes or ())
  axis_resources = jax.tree_util.tree_map(
      lambda spec: NamedSharding(mesh, spec), axis_resources)
  ckpt_struct = jax.tree_util.tree_map(
      lambda x, y: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=y),
      shapes, axis_resources)
  # Restore parameters from the checkpoint.
  ckpt = vmoe_checkpoint.restore_checkpoint_partitioned(
      prefix=prefix,
      tree=ckpt_struct,
      thread_pool=thread_pool)
  return mapping.map_state_dict(ckpt, target, rules, **map_state_dict_kwargs)
