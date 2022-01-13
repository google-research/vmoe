# Copyright 2021 Google LLC.
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
from vmoe.nn import routing


class NoisyTopExpertsPerItemRouterTest(absltest.TestCase):

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
  @mock.patch.object(
      routing.vmoe.moe,
      'get_top_experts_per_item_dispatcher',
      side_effect=lambda x, **_: x)
  def test_forward_deterministic(self, unused_mock):
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

  @mock.patch.object(
      routing.vmoe.moe,
      'get_top_experts_per_item_dispatcher',
      side_effect=lambda x, **_: x)
  def test_forward_not_deterministic(self, unused_mock):
    """Tests that output is different given two different gating PRNG seeds."""
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    layer = routing.NoisyTopExpertsPerItemRouter(
        num_experts=4,
        num_selected_experts=2,
        noise_std=1.0,
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


if __name__ == '__main__':
  absltest.main()
