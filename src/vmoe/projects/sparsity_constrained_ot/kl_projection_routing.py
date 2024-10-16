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
from ott.core import sinkhorn
from ott.geometry import geometry
import vmoe.moe
from vmoe.projects.sparsity_constrained_ot import ot_routing

Array = jnp.ndarray
BaseDispatcher = vmoe.moe.BaseDispatcher
DType = type(jnp.float32)
KwArgs = Mapping[str, Any]
Metrics = Mapping[str, Array]


class KLProjectionNoisyTopExpertsPerItemRouter(
    ot_routing.BaseOTNoisyTopExpertsPerItemRouter):
  """Noisy TopExpertsPerItem router through the KL projection.

  First, a dense (i.e. the gating) layer computes logits for each pair of
  (item, expert). Noise is added to these logits. The logits of all tokens are
  converted into a transportation plan, either using the Sinkhorn's algorithm or
  LBFGS. Each row of the transportation plan is a
  probability vector (non-negative and sums to 1), and that each column of the
  transportation plan sums to the same expected number of tokens.
  This plan determines which items are dispatched to which
  experts.

  Because the routing algorithm is non-differentiable, the only way to train the
  parameters of the dense (a.k.a. gating layer) is through the weights used
  to combine the output of the experts, and through two auxiliary losses that
  depend on the output of the gating.
  """
  importance_loss_weight: float = 0.0
  load_loss_weight: float = 0.0
  use_lbfgs: bool = False
  use_implicit_differentiation: bool = True

  @nn.compact
  def __call__(self, inputs: Array) -> Tuple[BaseDispatcher, Metrics]:
    gates_plan, gates_softmax, metrics = self._compute_gates_softmax_and_metrics(
        inputs, self.num_experts)
    if self.use_softmax_combine_weights:
      dispatcher = self._create_dispatcher(gates_dispatch=gates_plan,
                                           gates_combine=gates_softmax)
    else:
      dispatcher = self._create_dispatcher(gates_dispatch=gates_plan)
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
    gates_logits = nn.Dense(features=num_experts, use_bias=False,
                            dtype=dtype, name="dense")(inputs)
    # Compute the auxiliary losses defined in Appendix A.2, from
    # https://arxiv.org/abs/2106.05974. Notice that the "Load Loss" can only be
    # computed if the router is stochastic (i.e. deterministic = False).
    # Notice that the auxiliary losses are computed on each group independently
    # (i.e. through the vmaps surrounding the calls).
    group_size = gates_logits.shape[1]
    kl_projection_algorithm = self._get_kl_projection_algorithm(group_size)
    gates_softmax = jax.nn.softmax(gates_logits)
    gates_plan = kl_projection_algorithm(sim_matrix=gates_logits)
    if self.importance_loss_weight == 0.0:
      # no need to compute the loss in this case
      importance_loss = 0.0
    else:
      importance_loss = jax.vmap(self._importance_auxiliary_loss)(gates_softmax)
    if self.deterministic or self.noise_std == 0.0:
      metrics = {
          "auxiliary_loss": ot_routing._weighted_sum(  ## pylint: disable=protected-access
              (self.importance_loss_weight, importance_loss)),
          "importance_loss": importance_loss,
      }
      return gates_plan, gates_softmax, metrics  # pytype: disable=bad-return-type  # jax-ndarray
    else:
      noise_std = (1.0 / num_experts) * self.noise_std
      logits_noise = noise_std * jax.random.normal(
          key=self.make_rng("gating"), shape=gates_logits.shape)
      gates_logits_noisy = gates_logits + logits_noise
      gates_softmax_noisy = jax.nn.softmax(gates_logits)
      gates_plan_noisy = kl_projection_algorithm(sim_matrix=gates_logits_noisy)
      if self.load_loss_weight == 0.0:
        # no need to compute the loss in this case
        load_loss = 0.0
      else:
        load_loss = jax.vmap(
            functools.partial(
                self._load_auxiliary_loss,
                num_selected_experts=self.num_selected_experts,
                noise_std=noise_std))(gates_logits, gates_logits_noisy)
      metrics = {
          "auxiliary_loss": ot_routing._weighted_sum(  ## pylint: disable=protected-access
              (self.importance_loss_weight, importance_loss),
              (self.load_loss_weight, load_loss)),
          "importance_loss": importance_loss,
          "load_loss": load_loss,
      }
      return gates_plan_noisy, gates_softmax_noisy, metrics  # pytype: disable=bad-return-type  # jax-ndarray

  @nn.nowrap
  def _get_kl_projection_algorithm(self, group_size):
    # The input array multi_group_logits has the shape
    # (num_groups, group_size, num_experts).
    # For each token, the sum of routing probabilities to all experts is 1.
    required_sums_along_experts = jnp.ones((group_size,))
    # The expected number of tokens assigned to each expert.
    expected_num_tokens_per_expert = group_size / self.num_experts
    # We expect to assign all experts the same amount of tokens.
    required_sums_along_tokens = jnp.full(
        (self.num_experts,), expected_num_tokens_per_expert)
    def get_plan(sim_matrix):
      # The input array cost_matrix has the
      # shape (group_size, num_experts).
      if self.use_lbfgs:
        plan = projection.kl_projection_transport(
            sim_matrix=sim_matrix,
            marginals=(required_sums_along_experts, required_sums_along_tokens),
            make_solver=self._make_lbfgs_solver)
      else:
        geom = geometry.Geometry(cost_matrix=-sim_matrix,
                                 epsilon=1.0)
        sinkhorn_output = sinkhorn.sinkhorn(
            geom,
            lse_mode=False,
            a=required_sums_along_experts,
            b=required_sums_along_tokens,
            implicit_differentiation=self.use_implicit_differentiation,
            min_iterations=self.maxiter,
            max_iterations=self.maxiter)
        plan = sinkhorn_output.matrix
      return plan
    return jax.vmap(get_plan)

  @nn.nowrap
  def _make_lbfgs_solver(self, fun):
    return jaxopt.LBFGS(fun=fun, tol=1e-3,
                        maxiter=self.maxiter,
                        implicit_diff=self.use_implicit_differentiation,
                        linesearch="zoom")


class KLProjectionNoisyTopItemsPerExpertRouter(
    ot_routing.BaseOTNoisyTopItemsPerExpertRouter):
  """Noisy TopItemsPerExpert router through the KL projection.

  First, a dense (i.e. the gating) layer computes logits for each pair of
  (item, expert). Noise is added to these logits. The logits of all tokens are
  converted into a transportation plan using the Sinkhorn's algorithm.
  Each row of the transportation plan is a probability vector
  (non-negative and sums to 1), and that each column of the
  transportation plan sums to the same expected number of tokens.
  This plan determines which items are dispatched to which
  experts.

  Instead of picking the Top-K experts with highest score for each item, and
  then ignore choices that exceed the capacity (C) of any given expert, here we
  pick the Top-C items with highest score for each expert.
  """
  use_implicit_differentiation: bool = True

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
    metrics["auxiliary_loss"] = 0.
    return dispatcher, metrics  # pytype: disable=bad-return-type

  @nn.nowrap
  def _compute_gates(self, inputs: Array, num_experts: int) -> Array:
    if inputs.ndim != 3:
      raise ValueError(f"inputs.ndim must be 3, but it is {inputs.ndim}")
    dtype = self.dtype or inputs.dtype
    # Compute the gating logits for each pair of (item, expert).
    gates_logits = nn.Dense(features=num_experts, use_bias=False,
                            dtype=dtype, name="dense")(inputs)
    group_size = gates_logits.shape[1]
    ot_algorithm = self._get_kl_projection_algorithm(group_size)
    if self.deterministic or self.noise_std == 0.0:
      gates_ot = ot_algorithm(sim_matrix=gates_logits)
      gates_softmax = jax.nn.softmax(gates_logits)
      return gates_ot, gates_softmax  # pytype: disable=bad-return-type  # jax-ndarray
    else:
      noise_std = (1.0 / num_experts) * self.noise_std
      logits_noise = noise_std * jax.random.normal(
          key=self.make_rng("gating"), shape=gates_logits.shape)
      gates_logits_noisy = gates_logits + logits_noise
      gates_ot_noisy = ot_algorithm(sim_matrix=gates_logits_noisy)
      gates_softmax_noisy = jax.nn.softmax(gates_logits_noisy)
      return gates_ot_noisy, gates_softmax_noisy  # pytype: disable=bad-return-type  # jax-ndarray

  @nn.nowrap
  def _get_kl_projection_algorithm(self, group_size):
    # The input array multi_group_logits has the shape
    # (num_groups, group_size, num_experts).
    # For each token, the sum of routing probabilities to all experts is 1.
    required_sums_along_experts = jnp.ones((group_size,))
    # The expected number of tokens assigned to each expert.
    expected_num_tokens_per_expert = group_size / self.num_experts
    # We expect to assign all experts the same amount of tokens.
    required_sums_along_tokens = jnp.full(
        (self.num_experts,), expected_num_tokens_per_expert)
    def get_plan(sim_matrix):
      # The input array cost_matrix has the
      # shape (group_size, num_experts).
      geom = geometry.Geometry(cost_matrix=-sim_matrix, epsilon=1.0)
      sinkhorn_output = sinkhorn.sinkhorn(
          geom,
          lse_mode=False,
          a=required_sums_along_experts,
          b=required_sums_along_tokens,
          implicit_differentiation=self.use_implicit_differentiation,
          min_iterations=self.maxiter,
          max_iterations=self.maxiter)
      plan = sinkhorn_output.matrix
      return plan
    return jax.vmap(get_plan)
