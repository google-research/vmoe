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

"""Tests for routing."""
from unittest import mock
from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from vmoe.projects.sparsity_constrained_ot import ksparse_projection_routing


class KSparseProjectionTransportTopExpertsPerItemRouterTest(absltest.TestCase):

  def test_forward_deterministic(self):
    """Tests that output is the same given two different gating PRNG seeds."""
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    layer = ksparse_projection_routing.KSparseProjectionTransportTopExpertsPerItemRouter(
        num_experts=4,
        maxiter=500,
        deterministic=True,
        dispatcher={
            'bfloat16': False,
            'capacity_factor': 1.0,
            'batch_priority': False,
            'name': 'einsum'}
        )
    # y's are dispatch weights, m's are metrics.
    y1, m1 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(0)})
    y2, m2 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(1)})
    y1 = y1.combine_weights
    y2 = y2.combine_weights
    chex.assert_trees_all_close(y1, y2)
    chex.assert_trees_all_close(m1, m2)

  def test_sparse_transportation(self):
    """Tests that the col-k-sparse transportation plans produced by sparse OT satisfy rowise and columnwise constraints.
    """
    group_size = 4
    num_experts = 2
    layer = ksparse_projection_routing.KSparseProjectionTransportTopExpertsPerItemRouter(
        num_experts=num_experts,
        deterministic=True,
        maxiter=500,
        dispatcher={
            'bfloat16': False,
            'batch_priority': False,
            'capacity_factor': 1.0,
            'name': 'einsum'}
        )
    sparse_ot_fun = layer._get_k_sparse_ot_func(group_size=group_size)
    gates_logits = jax.random.uniform(jax.random.PRNGKey(1),
                                      (1, group_size, layer.num_experts))
    gates_ot = sparse_ot_fun(sim_matrix=gates_logits)
    # check whether row-sums are 1
    chex.assert_trees_all_close(
        gates_ot[0].sum(axis=1), jnp.ones(group_size,), atol=1e-3)
    # check whether column-sums are group_size/num_experts
    chex.assert_trees_all_close(
        gates_ot[0].sum(axis=0),
        jnp.full((num_experts,), group_size/num_experts), atol=1e-1)
    # check whether nnz in each row are bounded above by num_experts
    max_num_experts_used = max(jnp.count_nonzero(gates_ot[0], axis=1))
    chex.assert_trees_all_close(min(num_experts - max_num_experts_used, 0), 0)

  # We mock _get_k_sparse_ot_func and _create_dispatcher.
  # This will return the logits, so that we can test whether they
  # are non-negative.
  @mock.patch.object(
      ksparse_projection_routing.KSparseProjectionTransportTopExpertsPerItemRouter,  # pylint: disable=line-too-long
      '_get_k_sparse_ot_func',
      return_value=lambda sim_matrix: sim_matrix)
  @mock.patch.object(
      ksparse_projection_routing.KSparseProjectionTransportTopExpertsPerItemRouter,  # pylint: disable=line-too-long
      '_create_dispatcher',
      side_effect=lambda gates_dispatch, gates_combine: gates_dispatch)
  def test_gates_noneg(self, unused_mock1, unused_mock2):
    """Tests that the logits are non-negative."""
    x = jnp.arange(8 * 4).reshape(1, 8, 4).astype(jnp.float32)
    shift_x = x - 10
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    num_experts = 4
    layer = ksparse_projection_routing.KSparseProjectionTransportTopExpertsPerItemRouter(
        num_experts=num_experts,
        maxiter=500,
        dispatcher={
            'bfloat16': False,
            'capacity_factor': 1.0,
            'name': 'einsum'}
        )
    logits, _ = layer.apply(variables,
                            shift_x, rngs={'gating': jax.random.PRNGKey(0)})
    # Logits should be non-negative
    chex.assert_trees_all_close(min(logits.min(), 0.0), 0.0)


class KSparseProjectionTransportTopItemsPerExpertRouterTest(absltest.TestCase):

  def test_forward_deterministic(self):
    """Tests that output is the same given two different gating PRNG seeds."""
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    layer = ksparse_projection_routing.KSparseProjectionTransportTopItemsPerExpertRouter(
        num_experts=4,
        maxiter=500,
        deterministic=True,
        dispatcher={
            'bfloat16': False,
            'capacity_factor': 1.0,
            'name': 'einsum'}
        )
    # y's are dispatch weights, m's are metrics.
    y1, m1 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(0)})
    y2, m2 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(1)})
    y1 = y1.combine_weights
    y2 = y2.combine_weights
    chex.assert_trees_all_close(y1, y2)
    chex.assert_trees_all_close(m1, m2)

  def test_sparse_transportation(self):
    """Tests that the row-k-sparse transportation plans produced by sparse OT satisfy rowise and columnwise constraints.
    """
    group_size = 8
    num_experts = 2
    layer = ksparse_projection_routing.KSparseProjectionTransportTopItemsPerExpertRouter(
        num_experts=num_experts,
        maxiter=500,
        deterministic=True,
        dispatcher={
            'bfloat16': False,
            'capacity_factor': 2.0,
            'name': 'einsum'}
        )
    sparse_ot_fun = layer._get_k_sparse_ot_func(group_size=group_size)
    gates_logits = jax.random.normal(jax.random.PRNGKey(1),
                                     (1, group_size, layer.num_experts))
    gates_ot = sparse_ot_fun(sim_matrix=gates_logits)
    # check whether row-sums are 1
    chex.assert_trees_all_close(
        gates_ot[0].sum(axis=1), jnp.ones(group_size,), atol=1e-3)
    # check whether column-sums are group_size/num_experts
    chex.assert_trees_all_close(
        gates_ot[0].sum(axis=0),
        jnp.full((num_experts,), group_size/num_experts), atol=1e-3)
    # check whether nnz in each column are bounded above by
    # capacity_factor * group_size/num_experts
    capacity = layer.dispatcher['capacity_factor'] * group_size/num_experts
    max_num_items_selected = max(jnp.count_nonzero(gates_ot[0], axis=0))
    chex.assert_trees_all_close(min(capacity - max_num_items_selected, 0), 0)

  # We mock _get_k_sparse_ot_func and get_dense_einsum_dispatcher.
  # This will return the logits, so that we can test whether they
  # are non-negative.
  @mock.patch.object(
      ksparse_projection_routing.KSparseProjectionTransportTopItemsPerExpertRouter,  # pylint: disable=line-too-long
      '_get_k_sparse_ot_func',
      return_value=lambda sim_matrix: sim_matrix)
  @mock.patch.object(
      ksparse_projection_routing.KSparseProjectionTransportTopItemsPerExpertRouter,  # pylint: disable=line-too-long
      '_create_dispatcher_and_metrics',
      side_effect=lambda gates_dispatch, gates_combine: (gates_dispatch, {}))
  def test_gates_noneg(self, unused_mock1, unused_mock2):
    """Tests that the logits are non-negative."""
    x = jnp.arange(8 * 4).reshape(1, 8, 4).astype(jnp.float32)
    shift_x = x - 10
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    num_experts = 4
    layer = ksparse_projection_routing.KSparseProjectionTransportTopItemsPerExpertRouter(
        num_experts=num_experts,
        maxiter=500,
        dispatcher={
            'bfloat16': False,
            'capacity_factor': 1.0,
            'name': 'einsum'}
        )
    logits, _ = layer.apply(variables,
                            shift_x, rngs={'gating': jax.random.PRNGKey(0)})
    # Logits should be non-negative
    chex.assert_trees_all_close(min(logits.min(), 0.0), 0.0)


if __name__ == '__main__':
  absltest.main()
