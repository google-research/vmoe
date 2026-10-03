# Copyright 2026 Google LLC.
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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from vmoe.nn import routing


class NoisyTopExpertsPerItemRouterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # We mock get_top_items_per_expert_dispatcher to avoid having to specify the
    # parameters of the dispatcher during testing. The output of the
    # NoisyTopItemsPerExpertRouter is supposed to be a dispatcher, but we will
    # simply return the `gates_softmax`, which is fine for testing purposes.
    self.mock_get_top_items_per_expert_dispatcher = self.enter_context(
        mock.patch.object(
            routing.vmoe.moe,
            'get_top_experts_per_item_dispatcher',
            side_effect=lambda x, **_: x))

  def test_gshard_auxiliary_loss(self):
    gates = jnp.asarray([[.5, .4, .1], [.3, .3, .4], [.1, .2, .7],
                         [.8, .2, .0]])
    output = routing.NoisyTopExpertsPerItemRouter._gshard_auxiliary_loss(gates)
    # mean_gates_per_expert = [1.7, 1.1, 1.2] / 4.
    # mean_top1_per_expert = ([1, 0, 0]+[0, 0, 1]+[0, 0, 1]+[1, 0, 0]) / 4 =
    # [2, 0, 2] / 4.
    expected_output = 1.0875
    self.assertAlmostEqual(expected_output, output, places=6)

  def test_importance_auxiliary_loss(self):
    gates = jnp.asarray([[.5, .4, .1], [.3, .3, .4], [.1, .2, .7],
                         [.8, .2, .0]])
    output = routing.NoisyTopExpertsPerItemRouter._importance_auxiliary_loss(
        gates)
    # sum_gates_per_expert = [1.7, 1.1, 1.2]
    # mean(sum_gates_per_expert) = 1.3333334
    # std(sum_gates_per_expert) = 0.2624669
    # coefficient of variation = 0.19685018
    # coefficient of variation ** 2 = 0.03874999
    expected_output = 0.03874999
    self.assertAlmostEqual(expected_output, output, places=6)

  def test_load_auxiliary_loss(self):
    # batch_size = 3, num_experts = 3, k = 2.
    logits = jnp.asarray([[.9, .06, .04], [.7, .18, .12], [.85, .05, .1]],
                         dtype=jnp.float32)
    noise = jnp.asarray(
        [[-.037, .026, -.018], [-.074, .045, -.015], [-.067, -.059, .073]],
        dtype=jnp.float32)
    logits_noisy = logits + noise
    output = routing.NoisyTopExpertsPerItemRouter._load_auxiliary_loss(
        logits, logits_noisy, noise_std=.1, num_selected_experts=2)
    # In this case there's a clear winner (first expert) which is selected whp.
    # This increases the variance across experts and, thus, the auxiliary loss.
    # p_mean = [0.999 0.277 0.234]
    # std_p_mean = 0.3512197
    # mean_p_mean = 0.5039385
    # coefficient of variation = 0.6969495
    # coefficient of variation ** 2 = 0.48573864536489236
    expected_output = 0.48573864536489236
    self.assertAlmostEqual(expected_output, float(output), places=6)

  # We mock get_top_experts_per_item_dispatcher to avoid having to specify the
  # parameters of the dispatcher during testing. The output of the
  # NoisyTopExpertsPerItemRouter is supposed to be a dispatcher, but we will
  # simply return the `gates_softmax`, which is fine for testing purposes.

  def test_forward_deterministic(self):
    """Tests that output is the same given two different gating PRNG seeds."""
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    layer = routing.NoisyTopExpertsPerItemRouter(
        num_experts=4,
        num_selected_experts=2,
        noise_std=1.0,
        deterministic=True)
    # y's are dispatch weights, m's are metrics.
    y1, m1 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(0)})
    y2, m2 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(1)})
    chex.assert_trees_all_close(y1, y2)
    chex.assert_trees_all_close(m1, m2)

  def test_forward_not_deterministic(self):
    """Tests that output is different given two different gating PRNG seeds."""
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    layer = routing.NoisyTopExpertsPerItemRouter(
        num_experts=4,
        num_selected_experts=2,
        noise_std=2.0,
        deterministic=False)
    # y's are dispatch weights, m's are metrics.
    y1, m1 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(0)})
    y2, m2 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(1)})
    different_fn = lambda x, y: jnp.abs(x - y).sum() > 0.01
    error_msg_fn = lambda x, y: f'{x} is too close to {y}'
    chex.assert_trees_all_equal_comparator(different_fn, error_msg_fn, y1, y2)
    # Importance loss is applied before adding noise, so it should be identical.
    chex.assert_trees_all_close(m1['importance_loss'], m2['importance_loss'])
    del m1['importance_loss']
    del m2['importance_loss']
    chex.assert_trees_all_equal_comparator(different_fn, error_msg_fn, m1, m2)


class NoisyTopItemsPerExpertRouterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # We mock get_top_items_per_expert_dispatcher to avoid having to specify the
    # parameters of the dispatcher during testing. The output of the
    # NoisyTopItemsPerExpertRouter is supposed to be a dispatcher, but we will
    # simply return the `gates_softmax`, which is fine for testing purposes.
    self.mock_get_top_items_per_expert_dispatcher = self.enter_context(
        mock.patch.object(
            routing.vmoe.moe, 'get_top_items_per_expert_dispatcher',
            side_effect=lambda x, **_: (x, {})))

  def test_forward_deterministic(self):
    """Tests that output is the same given two different gating PRNG seeds."""
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    layer = routing.NoisyTopItemsPerExpertRouter(
        num_experts=4,
        noise_std=1.0,
        deterministic=True)
    # y's are dispatch weights.
    y1, _ = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(0)})
    y2, _ = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(1)})
    chex.assert_trees_all_close(y1, y2)

  def test_forward_not_deterministic(self):
    """Tests that output is different given two different gating PRNG seeds."""
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    layer = routing.NoisyTopItemsPerExpertRouter(
        num_experts=4,
        noise_std=1.0,
        deterministic=False)
    # y's are dispatch weights.
    y1, _ = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(0)})
    y2, _ = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(1)})
    different_fn = lambda x, y: jnp.abs(x - y).sum() > 0.01
    error_msg_fn = lambda x, y: f'{x} is too close to {y}'
    chex.assert_trees_all_equal_comparator(different_fn, error_msg_fn, y1, y2)


class BalancedAuxiliaryGradientTest(parameterized.TestCase):
  """Balanced expert statistics have a smooth zero-variance loss."""

  @parameterized.parameters(1, 2, 4)
  def test_importance_value_gradient_and_hessian_at_balance(self, experts):
    gates = jnp.full((4, experts), 1.0 / experts)
    loss = routing.NoisyTopExpertsPerItemRouter._importance_auxiliary_loss
    for function in (jax.value_and_grad(loss), jax.jit(jax.value_and_grad(loss))):
      value, gradient = function(gates)
      self.assertEqual(float(value), 0.0)
      np.testing.assert_array_equal(gradient, np.zeros_like(gates))
    direction = jnp.arange(gates.size, dtype=gates.dtype).reshape(gates.shape)
    _, tangent = jax.jvp(jax.grad(loss), (gates,), (direction,))
    self.assertTrue(np.all(np.isfinite(tangent)))

  @parameterized.product(experts=(1, 3), selected=(1,))
  def test_balanced_load_has_finite_zero_gradients(self, experts, selected):
    logits = jnp.zeros((4, experts))

    def loss(values):
      return routing.NoisyTopExpertsPerItemRouter._load_auxiliary_loss(
          values, logits, noise_std=0.3, num_selected_experts=selected
      )

    value, gradient = jax.jit(jax.value_and_grad(loss))(logits)
    self.assertEqual(float(value), 0.0)
    np.testing.assert_array_equal(gradient, np.zeros_like(logits))

  def test_nonuniform_balanced_gates_and_unbalanced_reference(self):
    loss = routing.NoisyTopExpertsPerItemRouter._importance_auxiliary_loss
    balanced = jnp.array([[0.9, 0.1], [0.1, 0.9]])
    np.testing.assert_array_equal(jax.grad(loss)(balanced), np.zeros((2, 2)))
    gates = jnp.array(
        [[0.5, 0.4, 0.1], [0.3, 0.3, 0.4], [0.1, 0.2, 0.7], [0.8, 0.2, 0.0]]
    )
    totals = np.asarray(gates).sum(0)
    expected = np.mean((totals - totals.mean()) ** 2) / totals.mean() ** 2
    np.testing.assert_allclose(loss(gates), expected, rtol=1e-5)
    analytical = jax.grad(loss)(gates)
    eps = 1e-3
    for i in range(gates.shape[0]):
      for j in range(gates.shape[1]):
        plus = gates.at[i, j].add(eps)
        minus = gates.at[i, j].add(-eps)
        finite_difference = (loss(plus) - loss(minus)) / (2 * eps)
        np.testing.assert_allclose(
            analytical[i, j], finite_difference, rtol=1e-3, atol=1e-5
        )

  @parameterized.parameters(True, False)
  def test_real_router_balanced_initialization_allows_training_update(
      self, deterministic
  ):
    layer = routing.NoisyTopExpertsPerItemRouter(
        num_experts=3,
        num_selected_experts=2,
        deterministic=deterministic,
        dispatcher={'name': 'einsum', 'batch_priority': False, 'capacity': 4},
    )
    inputs = jnp.ones((3, 4, 2))
    params = {'dense': {'kernel': jnp.zeros((2, 3))}}
    key = jax.random.PRNGKey(13)

    def objective(p):
      dispatcher, metrics = layer.apply({'params': p}, inputs, rngs={'gating': key})
      dispatched = dispatcher.dispatch(inputs)
      combined = dispatcher.combine(dispatched * 2.0)
      return jnp.mean(combined**2) + metrics['auxiliary_loss'].mean()

    value, grads = jax.jit(jax.value_and_grad(objective))(params)
    self.assertTrue(np.isfinite(value))
    for gradient in jax.tree.leaves(grads):
      self.assertTrue(np.all(np.isfinite(gradient)))
    optimizer = optax.sgd(1e-3)
    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    updated = optax.apply_updates(params, updates)
    self.assertTrue(np.isfinite(objective(updated)))


if __name__ == '__main__':
  absltest.main()
