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


class SparseProjectionTransportTopItemsPerExpertRouter(
    ot_routing.BaseOTNoisyTopItemsPerExpertRouter):
  """A router that uses quadratically regularized optimal transportation [1] to select k items per expert.

  First, a dense (i.e. the gating) layer computes logits for each pair of
  (token, expert). The logits of all tokens are transformed into a
  sparse transportation plan using a squared 2-norm regularized OT algorithm
  under the k-sparse constraint. In addition to be sparse, rows of The
  transportation matrix are probability vectors (non-negative and sums to 1)
  and columns sum to the same expected number of tokens.
  The router sends each token to all experts and then linearly
  combine the results using weights from the transportation plan.

  [1] Blondel, Mathieu, Vivien Seguy, and Antoine Rolet. "Smooth and
  sparse optimal transport." AISTATS 2018.
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
    metrics["auxiliary_loss"] = 0.0
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
    sparse_ot_fun = self._get_sparse_ot_func(group_size)
    gates_softmax = jax.nn.softmax(gates_logits)
    gates_ot = sparse_ot_fun(sim_matrix=gates_softmax)
    return gates_ot, gates_softmax

  @nn.nowrap
  def _get_sparse_ot_func(self, group_size):
    # The input array multi_group_logits has the shape
    # (num_groups, group_size, num_experts).
    make_solver = functools.partial(
        jaxopt.OptaxSolver, maxiter=self.maxiter,
        opt=optax.adam(self.adam_lr))

    ot_algorithm = _get_squared_l2_ot_algorithm(
        num_tokens=group_size,
        num_experts=self.num_experts,
        make_solver=make_solver)
    return jax.vmap(ot_algorithm)


def _get_squared_l2_ot_algorithm(num_tokens, num_experts, make_solver):
  """Get the sparsity constrained OT algorithm.

  Args:
    num_tokens: number of tokens
    num_experts: number of experts
    make_solver: the solver used in the optimization
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

  def get_sparse_ot_plan(sim_matrix):
    sparse_ot_plan = projection.projection_transport(
        sim_matrix=sim_matrix,
        marginals=(required_sums_along_experts, required_sums_along_tokens),
        make_solver=make_solver)
    return sparse_ot_plan
  return get_sparse_ot_plan
