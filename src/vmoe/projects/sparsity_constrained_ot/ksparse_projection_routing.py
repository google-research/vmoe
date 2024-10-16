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

"""Module with routing layers."""
import functools
from typing import Any, Mapping, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxopt
from jaxopt._src import projection  # temporary (until _regularized_transport becomes public in JAXopt)  ## pylint: disable=line-too-long
import optax
import vmoe.moe
from vmoe.projects.sparsity_constrained_ot import ot_routing
import vmoe.utils

Array = jnp.ndarray
BaseDispatcher = vmoe.moe.BaseDispatcher
DType = type(jnp.float32)
KwArgs = Mapping[str, Any]
Metrics = Mapping[str, Array]


class KSparseProjectionTransportTopExpertsPerItemRouter(
    ot_routing.BaseOTNoisyTopExpertsPerItemRouter):
  """A router that uses k-sparse optimal transportation (OT) to select k experts per item.

  First, a dense (i.e. the gating) layer computes logits for each pair of
  (token, expert). The logits of all tokens are transformed into a
  sparse transportation plan using a squared 2-norm regularized OT algorithm
  under the k-sparse constraint. In addition to be sparse, rows of The
  transportation matrix are probability vectors (non-negative and sums to 1)
  and columns sum to the same expected number of tokens.
  The router sends each token to all experts and then linearly
  combine the results using weights from the transportation plan.
  """

  adam_lr: float = 1e-2

  @nn.compact
  def __call__(self, inputs: Array) -> Tuple[BaseDispatcher, Metrics]:
    gates_ot, gates_softmax, metrics = self._compute_gates_softmax_and_metrics(
        inputs, self.num_experts)
    if self.use_softmax_combine_weights:
      dispatcher = self._create_dispatcher(gates_dispatch=gates_ot,
                                           gates_combine=gates_softmax)
    else:
      dispatcher = self._create_dispatcher(gates_dispatch=gates_ot)
    return dispatcher, metrics

  @nn.nowrap
  def _compute_gates_softmax_and_metrics(
      self, inputs: Array, num_experts: int) -> Tuple[Array, Array, Metrics]:
    if inputs.ndim != 3:
      raise ValueError(f"inputs.ndim must be 3, but it is {inputs.ndim}")
    if not num_experts >= self.num_selected_experts >= 1:
      raise ValueError(f"num_experts >= num_selected_experts >= 1, but got "
                       f"num_experts = {num_experts} and "
                       f"num_selected_experts = {self.num_selected_experts}.")
    dtype = self.dtype or inputs.dtype
    # Compute the gating logits for each pair of (item, expert).
    gates_logits = nn.Dense(features=num_experts,
                            use_bias=False,
                            dtype=dtype, name="dense")(inputs)
    group_size = gates_logits.shape[1]
    sparse_ot_fun = self._get_k_sparse_ot_func(group_size)
    gates_softmax = jax.nn.softmax(gates_logits)
    gates_ot = sparse_ot_fun(sim_matrix=gates_softmax)
    metrics = {"auxiliary_loss": 0.0}
    num_items_per_expert = jnp.count_nonzero(gates_ot, axis=1)
    metrics["num_items_per_expert_min"] = jnp.min(num_items_per_expert, axis=1)
    metrics["num_items_per_expert_max"] = jnp.max(num_items_per_expert, axis=1)
    metrics["num_items_per_expert_avg"] = jnp.mean(num_items_per_expert, axis=1)
    num_experts_per_item = jnp.count_nonzero(gates_ot, axis=2)
    metrics["num_experts_per_item_min"] = jnp.min(num_experts_per_item, axis=1)
    metrics["num_experts_per_item_max"] = jnp.max(num_experts_per_item, axis=1)
    metrics["num_experts_per_item_avg"] = jnp.mean(num_experts_per_item, axis=1)
    return gates_ot, gates_softmax, metrics  # pytype: disable=bad-return-type  # jax-ndarray

  @nn.nowrap
  def _get_k_sparse_ot_func(self, group_size):
    make_solver = functools.partial(
        jaxopt.OptaxSolver, maxiter=self.maxiter,
        opt=optax.adam(self.adam_lr))
    ot_algorithm = _get_sparsity_constrained_ot_algorithm(
        num_tokens=group_size,
        num_experts=self.num_experts,
        nnz_k=self.num_selected_experts,
        make_solver=make_solver,
        row_sparse=True)
    return jax.vmap(ot_algorithm)


class KSparseProjectionTransportTopItemsPerExpertRouter(
    ot_routing.BaseOTNoisyTopItemsPerExpertRouter):
  """A router that uses k-sparse optimal transportation (OT) to select k items per expert.

  First, a dense (i.e. the gating) layer computes logits for each pair of
  (token, expert). The logits of all tokens are transformed into a
  sparse transportation plan using a squared 2-norm regularized OT algorithm
  under the k-sparse constraint. In addition to be sparse, rows of The
  transportation matrix are probability vectors (non-negative and sums to 1)
  and columns sum to the same expected number of tokens.
  The router sends each token to all experts and then linearly
  combine the results using weights from the transportation plan.
  """

  adam_lr: float = 1e-2

  @nn.compact
  def __call__(self, inputs: Array) -> Tuple[BaseDispatcher, Metrics]:
    gates_ot, gates_softmax = self._compute_gates(
        inputs, self.num_experts)
    if self.use_softmax_combine_weights:
      dispatcher, metrics = self._create_dispatcher_and_metrics(
          gates_dispatch=gates_ot, gates_combine=gates_softmax)
    else:
      dispatcher, metrics = self._create_dispatcher_and_metrics(
          gates_dispatch=gates_ot)
    num_items_per_expert = jnp.count_nonzero(gates_ot, axis=1)
    metrics["num_items_per_expert_min"] = jnp.min(num_items_per_expert, axis=1)
    metrics["num_items_per_expert_max"] = jnp.max(num_items_per_expert, axis=1)
    metrics["num_items_per_expert_avg"] = jnp.mean(num_items_per_expert, axis=1)
    metrics["auxiliary_loss"] = 0.
    return dispatcher, metrics  # pytype: disable=bad-return-type

  @nn.nowrap
  def _compute_gates(
      self, inputs: Array, num_experts: int) -> Tuple[Array, Array]:
    if inputs.ndim != 3:
      raise ValueError(f"inputs.ndim must be 3, but it is {inputs.ndim}")
    dtype = self.dtype or inputs.dtype
    # Compute the gating logits for each pair of (item, expert).
    gates_logits = nn.Dense(features=num_experts,
                            use_bias=False,
                            dtype=dtype, name="dense")(inputs)
    group_size = gates_logits.shape[1]
    sparse_ot_fun = self._get_k_sparse_ot_func(group_size)
    gates_softmax = jax.nn.softmax(gates_logits)
    gates_ot = sparse_ot_fun(sim_matrix=gates_softmax)
    return gates_ot, gates_softmax

  @nn.nowrap
  def _get_k_sparse_ot_func(self, group_size):
    # The input array multi_group_logits has the shape
    # (num_groups, group_size, num_experts).
    dispatcher = self.dispatcher or {}
    nnz_k = vmoe.moe.compute_capacity(  # pylint: disable=protected-access
        # Target number of tokens to split among the `num_experts` experts.
        num_tokens=group_size,
        num_experts=self.num_experts,
        capacity_factor=dispatcher["capacity_factor"],
        ceil_or_round=dispatcher.get("capacity_ceil_or_round", "ceil"),
        multiple_of=dispatcher.get("capacity_multiple_of", 4))
    make_solver = functools.partial(
        jaxopt.OptaxSolver, maxiter=self.maxiter,
        opt=optax.adam(self.adam_lr))
    ot_algorithm = _get_sparsity_constrained_ot_algorithm(
        num_tokens=group_size,
        num_experts=self.num_experts,
        nnz_k=nnz_k,
        make_solver=make_solver,
        row_sparse=False)
    return jax.vmap(ot_algorithm)


def _get_sparsity_constrained_ot_algorithm(
    num_tokens, num_experts, nnz_k, make_solver, row_sparse):
  """Get the sparsity constrained OT algorithm.

  Args:
    num_tokens: number of tokens
    num_experts: number of experts
    nnz_k: number of nonzeros in a row or a column
    make_solver: the solver used in the optimization
    row_sparse: whether to use row-wise sparsity constraint
  Returns:
    The sparsity-constrained OT algorithm that converts a
    similarity matrix into a transportation plans.
  """

  # For each token, the sum of routing probabilities to all experts is 1.
  required_sums_along_experts = jnp.ones((num_tokens,))
  # The expected number of tokens assigned to each expert.
  expected_num_tokens_per_expert = num_tokens / num_experts
  # We expect to assign all experts the same amount of tokens.
  required_sums_along_tokens = jnp.full(
      (num_experts,), expected_num_tokens_per_expert)

  def maxop(x, marginal_b, gamma=1):
    return _max_l2_top_k(x,
                         k=nnz_k,
                         marginal_b=marginal_b,
                         gamma=gamma)
  maxop_vmap = jax.vmap(maxop, in_axes=(1, 0, None))
  max_grad_vmap = jax.vmap(jax.grad(maxop), in_axes=(1, 0, None))

  if row_sparse:
    marginal_a = required_sums_along_tokens
    marginal_b = required_sums_along_experts
  else:
    marginal_a = required_sums_along_experts
    marginal_b = required_sums_along_tokens

  def get_sparse_ot_plan(sim_matrix):
    if row_sparse:
      sim_matrix = sim_matrix.T
    sparse_ot_plan = projection._regularized_transport_semi_dual(  # pylint: disable=protected-access
        cost_matrix=-sim_matrix,
        marginals_a=marginal_a,
        marginals_b=marginal_b,
        make_solver=make_solver,
        max_vmap=maxop_vmap,
        max_grad_vmap=max_grad_vmap,
        gamma=1)
    if row_sparse:
      sparse_ot_plan = sparse_ot_plan.T
    return sparse_ot_plan
  return get_sparse_ot_plan


def _max_l2_top_k(x, k, marginal_b=1.0, gamma=1.0):
  scale = gamma * marginal_b
  p = jaxopt.projection.projection_sparse_simplex(x / scale, k)
  p = jax.lax.stop_gradient(p)
  z = jnp.dot(p, x) - 0.5 * scale * jnp.dot(p, p - 1)
  return z

