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

"""Core Sparse MoE utils using pjit.

Many thanks to Parker Schuh and Sudip Roy, for helping with the einsum
implementation, and to Jonathan Heek for helping writing the lift transform.

The following abbreviations are sometimes used to name the size of different
axes in the arrays.

G = num_groups. It must be a multiple of num_experts.
S = group_size.
E = num_experts.
C = capacity.
K = num_selected_experts. It must be <= num_experts.
"""
import abc
import math
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple

from absl import logging
import flax.core.lift
import flax.linen.partitioning
import flax.linen.transforms
import flax.struct
import jax
import jax.numpy as jnp
import vmoe.partitioning


Array = jnp.ndarray
CeilOrRound = Literal["ceil", "round"]
PartitionSpec = jax.sharding.PartitionSpec
with_sharding_constraint = vmoe.partitioning.with_sharding_constraint
_add_axis_to_metadata = flax.linen.partitioning._add_axis_to_metadata  # pylint: disable=protected-access


class BaseDispatcher(abc.ABC):
  """Base class for different dispatcher implementations.

  Dispatchers are in charge of preparing the data to be dispatched to the
  different experts, and then combining the outputs of each expert for each
  item. There are different ways of doing so with different memory / flops /
  runtime implications when running on actual hardware.

  In all cases, when dispatching data, they take an array of shape (G, S, ...).
  The groups (G) are dispatched independently of each other. The items in each
  group (S) will take place in the buffer (of capacity C) of items to be
  processed by each expert (E). The output is an array of shape (E, G * C, ...)
  with the elements to be processed by each expert.

  When combining data, they take an array of shape (E, G * C, ...) and output
  an array of shape (G, S, ...). Notice that the trailing dimensions (...) at
  combine might not be the same as the ones at dispatch (e.g. if the expert
  changes the shape of the data).
  """

  @abc.abstractmethod
  def dispatch(self, data: Array) -> Array:
    """Dispatches data to experts.

    Args:
      data: (G, S, ...) array with the data to dispatch to the experts.

    Returns:
      (E, G * C, ...) array with the data to be processed by each expert.
    """

  @abc.abstractmethod
  def combine(self, data: Array) -> Array:
    """Combines outputs from multiple experts.

    Args:
      data: (E, G * C, ...) array with the output data from each expert.

    Returns:
      (G, S, ...) array with the combined outputs from each expert.
    """


@flax.struct.dataclass
class DenseEinsumDispatcher(BaseDispatcher):
  """Dispatcher using Einsum, dispatching data to all experts.

  This is similar to EinsumDispatcher, but with the assumption that C = S.

  Attributes:
    combine_weights: (G, S, E) array with the combine weights for each item
      (G, S) for each expert (E).
    partition_spec: Optional. PartitionSpec used to constrain the sharding of
      the data arrays. By default (None), no sharding constraint is specified.
    einsum_precision: Optional. Precision used in all the einsums (e.g.
      combining the outputs of different experts).
  """
  combine_weights: Array
  partition_spec: Optional[PartitionSpec] = flax.struct.field(
      pytree_node=False, default=None)
  einsum_precision: jax.lax.Precision = flax.struct.field(
      pytree_node=False, default=jax.lax.Precision.DEFAULT)

  def dispatch(self, data: Array) -> Array:
    dispatch_weights = jnp.ones_like(self.combine_weights, dtype=jnp.bool_)
    data = jnp.einsum("GSE,GS...->GES...", dispatch_weights, data,
                      precision=self.einsum_precision)
    return _dispatch(data, self.partition_spec)

  def combine(self, data: Array) -> Array:
    """Combines data from experts according to combine_weights."""
    num_groups, _, _ = self.combine_weights.shape
    data = _receive(data, num_groups, self.partition_spec)
    return jnp.einsum("GSE,GES...->GS...", self.combine_weights, data,
                      precision=self.einsum_precision)


@flax.struct.dataclass
class EinsumDispatcher(BaseDispatcher):
  """Dispatcher using Einsum.

  Attributes:
    combine_weights: (G, S, E, C) array with the combine weights for each item
      (G, S) for each expert (E) and buffer position (C).
    dispatch_weights: Optional. (G, S, E, C) array with the dispatch weights of
      each item (G, S) for each expert (E) and buffer position (C).
    partition_spec: Optional. PartitionSpec used to constrain the sharding of
      the data arrays. By default (None), no sharding constraint is specified.
    einsum_precision: Optional. Precision used in all the einsums (e.g.
      combining the outputs of different experts).
  """
  combine_weights: Array
  dispatch_weights: Optional[Array] = None
  partition_spec: Optional[PartitionSpec] = flax.struct.field(
      pytree_node=False, default=None)
  einsum_precision: jax.lax.Precision = flax.struct.field(
      pytree_node=False, default=jax.lax.Precision.DEFAULT)

  def dispatch(self, data: Array) -> Array:
    dispatch_weights = (
        self.combine_weights > 0
        if self.dispatch_weights is None else self.dispatch_weights)
    data = jnp.einsum("GSEC,GS...->GEC...", dispatch_weights, data,
                      precision=self.einsum_precision)
    return _dispatch(data, self.partition_spec)

  def combine(self, data: Array) -> Array:
    """Combines data from experts according to combine_weights."""
    num_groups, _, _, _ = self.combine_weights.shape
    data = _receive(data, num_groups, self.partition_spec)
    return jnp.einsum("GSEC,GEC...->GS...", self.combine_weights, data,
                      precision=self.einsum_precision)


@flax.struct.dataclass
class ExpertIndicesDispatcher(BaseDispatcher):
  """Dispatcher using scatter/gather with (expert, buffer) indices.

  Attributes:
    indices: (G, S, K, 2) integer array with the (expert, buffer) indices of
      each item (G, S) and their K-selected experts. The tuple (expert, buffer)
      for each item is represented in the last dimension (of size 2).
    combine_weights: (G, S, K) array with the combine weights of each item
      (G, S) and their K-selected experts.
    num_experts: Number of experts.
    capacity: Capacity of each expert's buffer per group.
    partition_spec: Optional. PartitionSpec used to constrain the sharding of
      the data arrays. By default (None), no sharding constraint is specified.
    einsum_precision: Optional. Precision used in all the einsums (e.g.
      combining the outputs of different experts).
  """
  indices: Array  # (G, S, K, 2).
  combine_weights: Array   # (G, S, K).
  num_experts: int = flax.struct.field(pytree_node=False)
  capacity: int = flax.struct.field(pytree_node=False)
  partition_spec: Optional[PartitionSpec] = flax.struct.field(
      pytree_node=False, default=None)
  einsum_precision: jax.lax.Precision = flax.struct.field(
      pytree_node=False, default=jax.lax.Precision.DEFAULT)

  def dispatch(self, data: Array) -> Array:
    num_groups, _, num_selected_experts, _ = self.indices.shape
    _, _, *item_shape = data.shape
    data = jnp.repeat(data, num_selected_experts, axis=1)
    indices = self.indices.reshape(num_groups, -1, 2)
    shape = (self.num_experts, self.capacity, *item_shape)
    data = jax.vmap(lambda x, i: _scatter_nd(i, x, shape))(data, indices)
    return _dispatch(data, self.partition_spec)

  def combine(self, data: Array) -> Array:
    num_groups, _, _ = self.combine_weights.shape
    data = _receive(data, num_groups, self.partition_spec)
    data = jax.vmap(lambda x, i: x[i[:, :, 0], i[:, :, 1]])(data, self.indices)
    # Mask invalid gathered data.
    mask = jnp.logical_and(self.indices[..., 0] < self.num_experts,
                           self.indices[..., 1] < self.capacity)
    data = data * mask.reshape(mask.shape + (1,) * (data.ndim - 3))
    # Weighted sum of the outputs of the K-selected experts for each item.
    return jnp.einsum("GSK...,GSK->GS...", data, self.combine_weights,
                      precision=jax.lax.Precision.HIGHEST)


@flax.struct.dataclass
class Bfloat16Dispatcher(BaseDispatcher):
  """Dispatcher wrapper converting data to bfloat16 to save bandwidth."""
  dispatcher: BaseDispatcher

  def dispatch(self, data: Array) -> Array:
    dtype = data.dtype
    data = _cast_to_bfloat16(data)
    data = self.dispatcher.dispatch(data)
    return data.astype(dtype)

  def combine(self, data: Array) -> Array:
    dtype = data.dtype
    data = _cast_to_bfloat16(data)
    data = self.dispatcher.combine(data)
    return data.astype(dtype)


def compute_capacity(
    num_tokens: int,
    num_experts: int,
    capacity_factor: float,
    ceil_or_round: CeilOrRound = "ceil",
    multiple_of: Optional[int] = 4) -> int:
  """Returns the capacity per expert needed to distribute num_tokens among num_experts."""
  if ceil_or_round == "ceil":
    capacity = int(math.ceil(num_tokens * capacity_factor / num_experts))
  elif ceil_or_round == "round":
    capacity = int(round(num_tokens * capacity_factor / num_experts))
  else:
    raise ValueError(f"Unsupported {ceil_or_round=}")
  if capacity < 1:
    raise ValueError(f"The values num_tokens = f{num_tokens}, num_experts = "
                     f"{num_experts} and capacity_factor = {capacity_factor} "
                     f"lead to capacity = {capacity}, but it must be greater "
                     "than or equal to 1.")
  if multiple_of and multiple_of > 0:
    # Make capacity multiple of 4 to try to avoid padding.
    capacity += (-capacity) % multiple_of
  actual_capacity_factor = capacity * num_experts / num_tokens
  if abs(actual_capacity_factor - capacity_factor) > 1e-6:
    logging.warning(
        "The target capacity_factor is %f, but with num_tokens=%d and "
        "num_experts=%d the actual capacity_factor is %f.",
        capacity_factor, num_tokens, num_experts, actual_capacity_factor)
  return capacity


def get_dense_einsum_dispatcher(gates,
                                **dispatcher_kwargs) -> DenseEinsumDispatcher:
  # The dispatching algorithm is trivial, because all tokens are sent to
  # all experts this is coded implicitly in the DenseEinsumDispatcher class.
  return DenseEinsumDispatcher(combine_weights=gates, **dispatcher_kwargs)


def get_top_experts_per_item_dispatcher(
    gates: Array, name: str, num_selected_experts: int, batch_priority: bool,
    capacity: Optional[int] = None, capacity_factor: Optional[float] = None,
    capacity_ceil_or_round: CeilOrRound = "ceil",
    capacity_multiple_of: Optional[int] = 4,
    **dispatcher_kwargs) -> BaseDispatcher:
  """Returns a dispatcher implementing Top-Experts-Per-Item routing.

  For each item, the `num_selected_experts` experts with the largest gating
  score are selected in a greedy fashion. However, because each expert has a
  fixed `capacity`, if more items than `capacity` select a given expert some of
  the assignments will be ignored. All top-1 choices have priority over top-2
  choices and so on. In addition, the choices that are ignored also depend on
  `batch_priority`. If it is False, the "Vanilla" algorithm is used, meaning
  that items in earlier positions of the array have priority. If it is True, the
  "Batch Priority Routing" algorithm (see https://arxiv.org/abs/2106.05974) is
  used, which gives more priority to the items whose largest score is greater.

  Args:
    gates: (S, E) array with the gating values for each (item, expert).
      These values will also be used as combine_weights for the selected pairs.
    name: String with the type of dispatcher to use (supported values are
      "einsum" and "indices").
    num_selected_experts: Maximum number of experts to select per each item (K).
    batch_priority: Whether to use batch priority routing or not.
    capacity: If given, maximum number of items processed by each expert.
      Either this or `capacity_factor` must be given.
    capacity_factor: If given, sets the `capacity` to this factor of S * K / E.
      Either this or `capacity` must be given.
    capacity_ceil_or_round: Compute the capacity by either ceiling or rounding
      (default = "ceil").
    capacity_multiple_of: If given, ensures that the capacity is multiple of
      this number by increasing it if necessary.
    **dispatcher_kwargs: Additional arguments for the dispatcher object.

  Returns:
    A dispatcher.
  """
  if (capacity is None) == (capacity_factor is None):
    raise ValueError(
        "You must specify either 'capacity' or 'capacity_factor', and not both."
        f" Current values are capacity = {capacity!r}, "
        f"capacity_factor = {capacity_factor!r}")
  if not capacity:
    group_size, num_experts = gates.shape
    capacity = compute_capacity(
        # Target number of tokens to split among the `num_experts` experts.
        num_tokens=group_size * num_selected_experts,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        ceil_or_round=capacity_ceil_or_round,
        multiple_of=capacity_multiple_of)

  fn_map = {
      "einsum": _get_top_experts_per_item_einsum_dispatcher,
      "indices": _get_top_experts_per_item_expert_indices_dispatcher,
  }
  if name not in fn_map:
    raise ValueError(f"Unknown dispatcher type: {name!r}")
  return fn_map[name](gates, num_selected_experts, capacity, batch_priority,
                      **dispatcher_kwargs)


def get_top_items_per_expert_dispatcher(
    gates: Array,
    name: str,
    capacity: Optional[int] = None,
    capacity_factor: Optional[float] = None,
    capacity_ceil_or_round: CeilOrRound = "ceil",
    capacity_multiple_of: Optional[int] = 4,
    **dispatcher_kwargs) -> Tuple[BaseDispatcher, Dict[str, Array]]:
  """Returns a dispatcher implementing Top-Items-Per-Expert routing.

  For each expert, the top `capacity` items with the largest gating score are
  selected in a greedy fashion. This ensures that all experts process exactly
  `capacity` items, so they have a perfectly balanced load. However, it ignores
  the standard assumption that each item must be processed by a fixed number of
  experts.

  Args:
    gates: (S, E) array with the gating values for each (item, expert).
      These values will also be used as combine_weights for the selected pairs.
    name: String with the type of dispatcher to use (supported values are
      "einsum").
    capacity: If given, maximum number of items processed by each expert.
      Either this or `capacity_factor` must be given.
    capacity_factor: If given, sets the `capacity` to this factor of S / E.
      Either this or `capacity` must be given.
    capacity_ceil_or_round: Compute the capacity by either ceiling or rounding
      (default = "ceil").
    capacity_multiple_of: If given, ensures that the capacity is multiple of
      this number by increasing it if necessary.
    **dispatcher_kwargs: Additional arguments for the dispatcher object.

  Returns:
    A dispatcher and a dictionary of metrics.
  """
  if (capacity is None) == (capacity_factor is None):
    raise ValueError(
        "You must specify either 'capacity' or 'capacity_factor', and not both."
        f" Current values are capacity = {capacity!r}, "
        f"capacity_factor = {capacity_factor!r}")
  if not capacity:
    group_size, num_experts = gates.shape
    capacity = compute_capacity(
        # Target number of tokens to split among the `num_experts` experts.
        num_tokens=group_size,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        ceil_or_round=capacity_ceil_or_round,
        multiple_of=capacity_multiple_of)

  fn_map = {
      "einsum": _get_top_items_per_expert_einsum_dispatcher,
      # TODO(jpuigcerver): Implement indices dispatcher for Top-C routing.
  }
  if name not in fn_map:
    raise ValueError(f"Unknown dispatcher type: {name!r}")
  return fn_map[name](gates, capacity, **dispatcher_kwargs)


def sparse_moe_spmd(target: flax.linen.transforms.Target,
                    variable_axes: Mapping[flax.core.lift.CollectionFilter,
                                           flax.core.lift.InOutAxis],
                    split_rngs: Mapping[flax.core.lift.PRNGSequenceFilter,
                                        bool],
                    has_aux: bool = False,
                    methods=None):
  """Lift transformation that wraps a target with a Sparse MoE using SPMD.

  SPMD stands for "Single Program, Multiple Data", meaning that all experts
  actually implement the same function (program), but use different data
  (inputs and parameters). Thus, a single target to "expertify" is given.

  When an instance of a Linen module wrapped with this transformation is called,
  it expects one additional argument at the beginning, a "dispatcher"
  (see `BaseDispatcher`). This "dispatcher" is used to prepare the arguments to
  be processed by each "expert". The "target" is wrapped with vmap and applied
  to different sets of parameters and inputs. Finally, the "dispatcher" combines
  the outputs of all experts applied to each given item.

  If the target has any auxiliary outputs (e.g. metrics) that should not be
  combined, these can be returned by using "has_aux = True".

  Args:
    target: A target to wrap with a Sparse MoE (e.g. a flax.linen.Module) with
      methods passed via the `methods` argument.
    variable_axes: Mapping indicating the axis along each variable collection is
      "expertified". Typically, this is something like {"params": 0}.
    split_rngs: Mapping indicating whether to split each of the PRNGKeys passed
      to the experts.
    has_aux: If the target returns any auxiliary output that should not be
      combined, set this to True.
    methods: Methods from the target to wrap with a Sparse MoE. By default,
      the "__call__" method will be wrapped.

  Returns:
    A transformed target.
  """

  def wrapper(expert_fn: Callable[..., Any]):

    def transformed(scopes, dispatcher, *inputs):
      # Prepare inputs to be processed by each expert.
      inputs = jax.tree_util.tree_map(dispatcher.dispatch, inputs)
      # Wrap the target with vmap, to pass different parameters and inputs to
      # each expert.
      outputs = flax.core.lift.vmap(
          expert_fn,
          in_axes=0,
          out_axes=0,
          variable_axes=variable_axes,
          split_rngs=split_rngs)(scopes, *inputs)
      # Combine outputs.
      if has_aux:
        outputs, aux = outputs
      outputs = jax.tree_util.tree_map(dispatcher.combine, outputs)
      return (outputs, aux) if has_aux else outputs

    return transformed

  return flax.linen.transforms.lift_transform(wrapper, target, methods=methods)


def sparse_moe_spmd_with_axes(
    target: flax.linen.transforms.Target,
    variable_axes: Mapping[flax.core.lift.CollectionFilter,
                           flax.core.lift.InOutAxis],
    split_rngs: Mapping[flax.core.lift.PRNGSequenceFilter, bool],
    partitioning_axis_names: Mapping[str, str],
    has_aux: bool = False,
    methods=None):
  """Lift transformation similar to sparse_moe_spmd with partitioned named axes."""
  variable_axes = dict(variable_axes)
  for name in partitioning_axis_names:
    variable_axes[f"{name}_axes"] = None

  lifted = sparse_moe_spmd(target, variable_axes, split_rngs, has_aux, methods)

  for collection_name, axis in variable_axes.items():
    if collection_name in partitioning_axis_names:
      lifted = _add_axis_to_metadata(
          lifted,
          axis_pos=axis,
          axis_name=partitioning_axis_names[collection_name],
          axis_col=f"{collection_name}_axes")

  return lifted


def _cast_to_bfloat16(x: Array) -> Array:
  return x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x


def _convert_partition_spec(spec):
  if spec is not None and not isinstance(spec, PartitionSpec):
    spec = (spec,) if isinstance(spec, str) else tuple(spec)
    spec = PartitionSpec(*spec)
  return spec


def _dispatch(data: Array, partition_spec: Optional[PartitionSpec]) -> Array:
  """Dispatches data to experts using all_to_all."""
  partition_spec = _convert_partition_spec(partition_spec)
  num_groups, num_experts, capacity, *item_shape = data.shape
  data = with_sharding_constraint(data, partition_spec)
  if num_groups % num_experts == 0:
    data = data.reshape(num_experts, -1, num_experts, capacity, *item_shape)
    data = jnp.swapaxes(data, 0, 2)
  else:
    data = jnp.swapaxes(data, 0, 1)
  data = data.reshape(-1, *item_shape)
  data = with_sharding_constraint(data, partition_spec)
  return data.reshape(num_experts, num_groups * capacity, *item_shape)


def _receive(data: Array, num_groups: int,
             partition_spec: Optional[PartitionSpec]) -> Array:
  """Receives data from experts using all_to_all."""
  partition_spec = _convert_partition_spec(partition_spec)
  num_experts, num_groups_time_capacity, *item_shape = data.shape
  capacity = num_groups_time_capacity // num_groups
  data = data.reshape(num_experts * num_groups, capacity, *item_shape)
  data = with_sharding_constraint(data, partition_spec)
  if num_groups % num_experts == 0:
    data = data.reshape(num_experts, -1, num_experts, capacity, *item_shape)
    data = jnp.swapaxes(data, 0, 2)
    data = data.reshape(num_groups, num_experts, capacity, *item_shape)
  else:
    data = data.reshape(num_experts, num_groups, capacity, *item_shape)
    data = jnp.swapaxes(data, 0, 1)
  data = with_sharding_constraint(data, partition_spec)
  return data


def _scatter_nd(indices, updates, shape):
  """Jax implementation of tf.scatter_nd.

  Notes:
  - The updates are cumulative, ie. if multiple indices point to the
    same position, the output value at this position is accumulated.
  - We rely on the fact that out-of-range indices will be quietly ignored and
    don't raise any error. This breaks what JAX index ops specify
    (https://jax.readthedocs.io/en/latest/jax.ops.html), but makes the code
    easier.

  Args:
    indices: An int matrix of (i, j, ...) indices with shape [B, ndim].
    updates: An array of data points with shape [B, ...].
    shape: An int vector with the dimensions of the output array of size [ndim].

  Returns:
    An array of shape `shape` with updated values at given indices.
  """
  # See: https://www.tensorflow.org/api_docs/python/tf/scatter_nd.
  zeros = jnp.zeros(shape, updates.dtype)
  key = tuple(jnp.moveaxis(indices, -1, 0))
  return zeros.at[key].add(updates)


def _get_top_experts_per_item_common(
    gates: Array, num_selected_experts: int,
    batch_priority: bool) -> Tuple[Array, Array, Array]:
  """Returns common arrays used by Top-Experts-Per-Item routing.

  Args:
    gates: (S, E) array with the gating values for each (item, expert).
      These values will also be used as combine_weights for the selected pairs.
    num_selected_experts: Maximum number of experts to select per item.
    batch_priority: Whether to use batch priority routing or not.

  Returns:
    - `combine_weights`, with shape (S, K) with the weights used to
      combine the outputs of the K-selected experts for each item.
    - `expert_index`, with shape (S, K) containing the expert_index for each of
      the K-selected experts for each item.
    - `buffer_index`, with shape (S, K, E) containing the buffer index for each
      item and selected expert.
  """
  group_size, num_experts = gates.shape
  combine_weights, expert_index = jax.lax.top_k(gates, num_selected_experts)
  if batch_priority:
    # Sort items according to their maximum routing weight. The permutation will
    # be reversed later, so no need to permute combine_weights here.
    perm = jnp.argsort(-combine_weights[:, 0])
    expert_index = expert_index[perm]
  # (K * S,). Make K the leading axis to ensure that top-1 choices have priority
  # over top-2 choices and so on. Flatten array for cumsum.
  expert_index = jnp.swapaxes(expert_index, 0, 1).ravel()
  # (K * S, E). Convert expert indices to a one-hot array.
  expert_one_hot = jax.nn.one_hot(expert_index, num_experts, dtype=jnp.int32)
  # (K * S, E) -> (K, S, E) -> (S, K, E). Use cumsum to compute the buffer idx
  # within each experts' buffer.
  buffer_index = jnp.cumsum(expert_one_hot, axis=0) * expert_one_hot - 1
  buffer_index = buffer_index.reshape(-1, group_size, num_experts)
  buffer_index = jnp.swapaxes(buffer_index, 0, 1)
  # (K, S) -> (S, K). Revert expert_index to the original shape.
  expert_index = jnp.swapaxes(expert_index.reshape(-1, group_size), 0, 1)
  if batch_priority:
    # Permute the items to their original order.
    inv_perm = jnp.argsort(perm)
    expert_index = expert_index[inv_perm]
    buffer_index = buffer_index[inv_perm]
  return combine_weights, expert_index, buffer_index


def _get_top_experts_per_item_einsum_dispatcher(
    gates: Array, num_selected_experts: int, capacity: int,
    batch_priority: bool, **dispatcher_kwargs) -> EinsumDispatcher:
  """Returns an EinsumDispatcher performing Top-Experts-Per-Item routing.

  Args:
    gates: (S, E) array with the gating values for each (item, expert).
      These values will also be used as combine_weights for the selected pairs.
    num_selected_experts: Maximum number of experts to select per each item.
    capacity: Maximum number of items processed by each expert.
    batch_priority: Whether to use batch priority routing or not.
    **dispatcher_kwargs: Additional arguments for the EinsumDispatcher.

  Returns:
    An EinsumDispatcher object.
  """
  _, _, buffer_idx = _get_top_experts_per_item_common(
      gates, num_selected_experts, batch_priority)
  # (S, K, E) -> (S, E). Select the only buffer index for each (item, expert).
  buffer_idx = jnp.max(buffer_idx, axis=1)
  # (S, E, C). Convert the buffer indices to a one-hot matrix. We rely on the
  # fact that indices < 0 or >= capacity will be ignored by the dispatcher.
  dispatch_weights = jax.nn.one_hot(buffer_idx, capacity, dtype=jnp.bool_)
  einsum_precision = dispatcher_kwargs.get("einsum_precision",
                                           jax.lax.Precision.DEFAULT)
  combine_weights = jnp.einsum(
      "SE,SEC->SEC", gates, dispatch_weights, precision=einsum_precision)
  return EinsumDispatcher(
      combine_weights=combine_weights,
      dispatch_weights=dispatch_weights,
      **dispatcher_kwargs)


def _get_top_experts_per_item_expert_indices_dispatcher(
    gates: Array, num_selected_experts: int, capacity: int,
    batch_priority: bool, **dispatcher_kwargs) -> ExpertIndicesDispatcher:
  """Returns an ExpertIndicesDispatcher performing Top-Experts-Per-Item routing.

  Args:
    gates: (S, E) array with the gating values for each (item, expert).
      These values will also be used as combine_weights for the selected pairs.
    num_selected_experts: Maximum number of experts to select per each item.
    capacity: Maximum number of items processed by each expert.
    batch_priority: Whether to use batch priority routing or not.
    **dispatcher_kwargs: Additional arguments for the ExpertIndicesDispatcher.

  Returns:
    An ExpertIndicesDispatcher object.
  """
  _, num_experts = gates.shape
  combine_weights, expert_idx, buffer_idx = _get_top_experts_per_item_common(
      gates, num_selected_experts, batch_priority)
  # (S, K, E) -> (S, K). Select the only buffer index for each (item, k_choice).
  buffer_idx = jnp.max(buffer_idx, axis=2)
  return ExpertIndicesDispatcher(
      indices=jnp.stack([expert_idx, buffer_idx], axis=-1),
      combine_weights=combine_weights,
      num_experts=num_experts,
      capacity=capacity,
      **dispatcher_kwargs)


def _get_top_items_per_expert_einsum_dispatcher(
    gates: Array, capacity: int,
    **dispatcher_kwargs) -> Tuple[EinsumDispatcher, Dict[str, Array]]:
  """Returns an EinsumDispatcher performing Top-Items-Per-Expert routing.

  Args:
    gates: (S, E) array with the gating values for each (item, expert).
      These values will also be used as combine_weights for the selected pairs.
    capacity: Maximum number of items processed by each expert.
    **dispatcher_kwargs: Additional arguments for the EinsumDispatcher.

  Returns:
    An EinsumDispatcher object and a dictionary of metrics.
  """
  group_size, num_experts = gates.shape
  # Get top items for each expert, and convert to one hot. (E, C) -> (S, E, C).
  top_items_gates, top_items_index = jax.lax.top_k(gates.transpose(), capacity)
  dispatch_weights = jax.nn.one_hot(top_items_index, group_size, axis=0,
                                    dtype=jnp.bool_)
  einsum_precision = dispatcher_kwargs.get("einsum_precision",
                                           jax.lax.Precision.DEFAULT)
  combine_weights = jnp.einsum("SE,SEC->SEC", gates, dispatch_weights,
                               precision=einsum_precision)
  dispatcher = EinsumDispatcher(dispatch_weights=dispatch_weights,
                                combine_weights=combine_weights,
                                **dispatcher_kwargs)
  # Compute metrics useful for monitoring training.
  num_experts_per_item = jnp.sum(dispatch_weights, axis=(1, 2), dtype=jnp.int32)
  metrics = {
      "num_experts_per_item_min": jnp.min(num_experts_per_item),
      "num_experts_per_item_max": jnp.max(num_experts_per_item),
      "min_selected_gate": jnp.min(top_items_gates),
      "max_selected_gate": jnp.max(top_items_gates),
  }
  log2_num_experts = int(math.log2(num_experts))
  for t in [2**i for i in range(log2_num_experts + 1)] + [num_experts]:
    ratio = jnp.sum(num_experts_per_item >= t) / group_size
    metrics[f"ratio_processed_items_by_at_least_{t}_experts"] = ratio

  return dispatcher, metrics
