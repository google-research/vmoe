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
from vmoe.projects.sparsity_constrained_ot.kl_projection_routing import KLProjectionNoisyTopExpertsPerItemRouter
from vmoe.projects.sparsity_constrained_ot.kl_projection_routing import KLProjectionNoisyTopItemsPerExpertRouter


class KLProjectionNoisyTopExpertsPerItemRouterTest(absltest.TestCase):

  def test_forward_deterministic(self):
    """Tests that output is the same given two different gating PRNG seeds."""
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    for use_lbfgs in [True, False]:
      layer = KLProjectionNoisyTopExpertsPerItemRouter(
          num_experts=4,
          num_selected_experts=2,
          noise_std=1.0,
          deterministic=True,
          use_lbfgs=use_lbfgs,
          dispatcher={
              'batch_priority': False,
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

  def test_forward_not_deterministic(self):
    """Tests that output is different given two different gating PRNG seeds."""
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    for use_lbfgs in [True, False]:
      layer = KLProjectionNoisyTopExpertsPerItemRouter(
          num_experts=4,
          num_selected_experts=2,
          noise_std=2.0,
          load_loss_weight=1.0,
          deterministic=False,
          use_lbfgs=use_lbfgs,
          dispatcher={
              'batch_priority': False,
              'bfloat16': False,
              'capacity_factor': 1.0,
              'name': 'einsum'}
          )
      # y's are dispatch weights, m's are metrics.
      y1, m1 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(0)})
      y2, m2 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(1)})
      y1 = y1.combine_weights
      y2 = y2.combine_weights
      different_fn = lambda x, y: jnp.abs(x - y).sum() > 0.01
      error_msg_fn = lambda x, y: f'{x} is too close to {y}'
      chex.assert_trees_all_equal_comparator(different_fn, error_msg_fn, y1, y2)
      # Importance loss is applied before adding noise,
      # so it should be identical.
      chex.assert_trees_all_close(m1['importance_loss'], m2['importance_loss'])
      del m1['importance_loss']
      del m2['importance_loss']
      chex.assert_trees_all_equal_comparator(different_fn, error_msg_fn, m1, m2)

  # We mock _create_dispatcher to get the transportation plan.
  @mock.patch.object(
      KLProjectionNoisyTopExpertsPerItemRouter,
      '_create_dispatcher',
      side_effect=lambda gates_dispatch, gates_combine: gates_dispatch)
  def test_kl_projection_plan(self, unused_mock):
    """Test that that KL projection with a constant cost matrix gives a uniform transportation plan.
    """
    x = jnp.ones((1, 5, 4)).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    uniform_mat = jax.numpy.full((1, 5, 4), 0.25).astype(jnp.float32)
    for use_lbfgs in [True, False]:
      layer = KLProjectionNoisyTopExpertsPerItemRouter(
          num_experts=4,
          num_selected_experts=4,
          use_lbfgs=use_lbfgs,
          noise_std=0.0,
          deterministic=True)
      # Get the transportation plan
      plan, _, = layer.apply(variables, x,
                             rngs={'gating': jax.random.PRNGKey(0)})
      # The plan should be a constant matrix filled with values 0.25.
      chex.assert_trees_all_close(plan, uniform_mat)


class KLProjectionNoisyTopItemsPerExpertRouterTest(absltest.TestCase):

  def test_forward_deterministic(self):
    """Tests that output is the same given two different gating PRNG seeds."""
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    layer = KLProjectionNoisyTopItemsPerExpertRouter(
        num_experts=4,
        noise_std=1.0,
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

  def test_forward_not_deterministic(self):
    """Tests that output is different given two different gating PRNG seeds."""
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    layer = KLProjectionNoisyTopItemsPerExpertRouter(
        num_experts=4,
        noise_std=5.0,
        deterministic=False,
        dispatcher={
            'bfloat16': False,
            'capacity_factor': 1.0,
            'name': 'einsum'}
        )
    # y's are dispatch weights, m's are metrics.
    y1, _ = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(0)})
    y2, _ = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(1)})
    y1 = y1.combine_weights
    y2 = y2.combine_weights
    different_fn = lambda x, y: jnp.abs(x - y).sum() > 0.01
    error_msg_fn = lambda x, y: f'{x} is too close to {y}'
    chex.assert_trees_all_equal_comparator(different_fn, error_msg_fn, y1, y2)

  # We mock _create_dispatcher to get the transportation plan.
  @mock.patch.object(
      KLProjectionNoisyTopItemsPerExpertRouter,
      '_create_dispatcher_and_metrics',
      side_effect=lambda gates_dispatch, gates_combine: (gates_dispatch, {}))
  def test_kl_projection_plan(self, unused_mock):
    """Test that that KL projection with a constant cost matrix gives a uniform transportation plan.
    """
    x = jnp.ones((1, 5, 4)).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    uniform_mat = jax.numpy.full((1, 5, 4), 0.25).astype(jnp.float32)
    layer = KLProjectionNoisyTopItemsPerExpertRouter(
        num_experts=4,
        noise_std=0.0,
        deterministic=True)
    # Get the transportation plan
    plan, _, = layer.apply(variables, x,
                           rngs={'gating': jax.random.PRNGKey(0)})
    # The plan should be a constant matrix filled with values 0.25.
    chex.assert_trees_all_close(plan, uniform_mat)
if __name__ == '__main__':
  absltest.main()
