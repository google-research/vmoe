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
from vmoe.projects.sparsity_constrained_ot import sparse_projection_routing


class SparseProjectionTransportTopItemsPerExpertRouterTest(absltest.TestCase):

  def test_forward_deterministic(self):
    """Tests that output is the same given two different gating PRNG seeds."""
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    layer = sparse_projection_routing.SparseProjectionTransportTopItemsPerExpertRouter(
        num_experts=4,
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

  # We mock _get_sparse_ot_func and get_dense_einsum_dispatcher.
  # This will return the logits, so that we can test whether they
  # are non-negative.
  @mock.patch.object(
      sparse_projection_routing.SparseProjectionTransportTopItemsPerExpertRouter,  # pylint: disable=line-too-long
      '_get_sparse_ot_func',
      return_value=lambda sim_matrix: sim_matrix)
  @mock.patch.object(
      sparse_projection_routing.SparseProjectionTransportTopItemsPerExpertRouter,  # pylint: disable=line-too-long
      '_create_dispatcher_and_metrics',
      side_effect=lambda gates_dispatch, gates_combine: (gates_dispatch, {}))
  def test_gates_noneg(self, unused_mock1, unused_mock2):
    """Tests that the logits are non-negative."""
    x = jnp.arange(8 * 4).reshape(1, 8, 4).astype(jnp.float32)
    shift_x = x - 10
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    num_experts = 4
    layer = sparse_projection_routing.SparseProjectionTransportTopItemsPerExpertRouter(
        num_experts=num_experts,
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
