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

"""Functions to initialize a TrainState from pre-trained checkpoints."""
import multiprocessing.pool
from typing import Any, Optional, Union

import jax
from jax.experimental import pjit
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

__all__ = ['initialize_from_vit', 'initialize_from_vmoe']


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
    ckpt = pjit.pjit(fun=lambda x: x, in_axis_resources=None,
                     out_axis_resources=axis_resources)(ckpt)
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
  shapes = jax.tree_map(lambda x: x.global_shape, index['index'])
  if version == vmoe_checkpoint.Version.UNKNOWN:
    if (jax.tree_structure(vmoe_serialization.to_state_dict(target)) !=
        jax.tree_structure(shapes)):
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
