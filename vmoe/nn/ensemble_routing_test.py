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

"""Tests for ensemble routing.

We will use the following abbreviations:
  G = number of groups
  S = group size
  E = total number of experts
  K = number of selected experts
  M = number of ensemble members, also referred to as ensemble size.
  H = hidden size (a.k.a. number of dimensions of each token).
"""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from vmoe.nn import ensemble_routing as ens_routing
from vmoe.nn import routing

NoisyTopExpertsPerItemEnsembleRouter = ens_routing.NoisyTopExpertsPerItemEnsembleRouter


class NoisyTopExpertsPerItemEnsembleRouterTest(parameterized.TestCase):

  # We mock get_top_experts_per_item_dispatcher to avoid having to specify the
  # parameters of the dispatcher during testing. The output of the
  # NoisyTopExpertsPerItemEnsembleRouter is supposed to be a dispatcher, but we
  # will simply return the `gates_softmax`, which is fine for testing purposes.
  @parameterized.named_parameters(
      ('G=1, M=1', 1, 1, (1, 1)), ('G=1, M=2', 1, 2, (1, 2)),
      ('G=2, M=1', 2, 1, (2, 1)), ('G=2, M=2', 2, 2, (2, 2)))
  @mock.patch.object(
      routing.vmoe.moe,
      'get_top_experts_per_item_dispatcher',
      side_effect=lambda x, **_: x)
  def test_forward_deterministic(self, num_groups, ens_size,
                                 expected_output_shape, unused_mock):
    """Tests that output is the same given two different gating PRNG seeds."""
    num_experts = 4
    num_selected_experts = 2
    batch_size = 6
    batch_size_with_repeat = batch_size * ens_size
    dim_x = num_experts
    product_shape = dim_x * batch_size_with_repeat * num_groups
    x = jnp.arange(product_shape)
    x = x.reshape(num_groups, batch_size_with_repeat, dim_x).astype(jnp.float32)
    kernel = jnp.eye(dim_x).reshape(dim_x, ens_size, num_experts // ens_size)
    variables = {'params': {'dense': {'kernel': kernel}}}
    layer = NoisyTopExpertsPerItemEnsembleRouter(
        num_experts=num_experts,
        ensemble_size=ens_size,
        num_selected_experts=num_selected_experts,
        noise_std=1.0,
        deterministic=True)
    # y's are dispatch weights, m's are metrics.
    y1, m1 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(0)})
    y2, m2 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(1)})
    chex.assert_trees_all_close(y1, y2)
    chex.assert_trees_all_close(m1, m2)
    for loss in m1.values():
      self.assertEqual(loss.shape, expected_output_shape)

  @parameterized.named_parameters(
      ('G=1, M=1', 1, 1, (1, 1)), ('G=1, M=2', 1, 2, (1, 2)),
      ('G=2, M=1', 2, 1, (2, 1)), ('G=2, M=2', 2, 2, (2, 2)))
  @mock.patch.object(
      routing.vmoe.moe,
      'get_top_experts_per_item_dispatcher',
      side_effect=lambda x, **_: x)
  def test_forward_not_deterministic(self, num_groups, ens_size,
                                     expected_output_shape, unused_mock):
    """Tests that output is different given two different gating PRNG seeds."""
    num_experts = 4
    num_selected_experts = 2
    batch_size = 6
    batch_size_with_repeat = batch_size * ens_size
    dim_x = num_experts
    product_shape = dim_x * batch_size_with_repeat * num_groups
    x = jnp.arange(product_shape)
    x = x.reshape(num_groups, batch_size_with_repeat, dim_x).astype(jnp.float32)
    kernel = jnp.eye(dim_x).reshape(dim_x, ens_size, num_experts // ens_size)
    variables = {'params': {'dense': {'kernel': kernel}}}
    layer = NoisyTopExpertsPerItemEnsembleRouter(
        num_experts=num_experts,
        ensemble_size=ens_size,
        num_selected_experts=num_selected_experts,
        noise_std=1.0,
        deterministic=False)
    # y's are dispatch weights, m's are metrics.
    y1, m1 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(0)})
    y2, m2 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(1)})
    different_fn = lambda x, y: jnp.abs(x - y).sum() > 1e-3
    error_msg_fn = lambda x, y: f'{x} is too close to {y}'
    chex.assert_trees_all_equal_comparator(different_fn, error_msg_fn, y1, y2)
    # Importance loss is applied before adding noise, so it should be identical.
    chex.assert_trees_all_close(m1['importance_loss'], m2['importance_loss'])
    del m1['importance_loss']
    del m2['importance_loss']
    chex.assert_trees_all_equal_comparator(different_fn, error_msg_fn, m1, m2)
    for loss1, loss2 in zip(m1.values(), m2.values()):
      self.assertEqual(loss1.shape, expected_output_shape)
      self.assertEqual(loss2.shape, expected_output_shape)

  @mock.patch.object(
      routing.vmoe.moe,
      'get_top_experts_per_item_dispatcher',
      side_effect=lambda x, **_: x)
  def test_expert_partitioning_in_forward_deterministic(self, unused_mock):
    """Tests that experts are selected according to their partitioning."""
    ens_size = 2
    dim_x = 3
    num_experts = 4  # With ens_size=2, partition {0, 1}, {2, 3}.
    num_selected_experts = 1
    batch_size = 1
    batch_size_with_repeat = batch_size * ens_size
    num_groups = 1
    x = jnp.arange(batch_size_with_repeat * dim_x)
    x = x.reshape(num_groups, batch_size_with_repeat, dim_x).astype(jnp.float32)

    # The kernel is such that x * kernel leads to select expert 3 and 1 for
    # the first and second part of the batch. But the partitioning enforces a
    # selection in respectively {0, 1} and {2, 3}.
    kernel = [[[1.0, 1.0], [-1.0, -1.0]],
              [[-1.0, 1.0], [-1.0, -1.0]],
              [[-1.0, -1.0], [-1.0, 1.0]]]
    kernel = jnp.asarray(kernel)
    variables = {'params': {'dense': {'kernel': kernel}}}
    layer = NoisyTopExpertsPerItemEnsembleRouter(
        num_experts=num_experts,
        ensemble_size=ens_size,
        num_selected_experts=num_selected_experts,
        noise_std=1.0,
        deterministic=False)
    rngs = {'gating': jax.random.PRNGKey(0)}
    gates_softmax, _ = layer.apply(variables, x, rngs=rngs)
    selected_experts = jnp.argmax(gates_softmax, axis=-1)
    # Only one group, hence the first 0 index.
    self.assertIn(selected_experts[0, 0], (0, 1))
    self.assertIn(selected_experts[0, 1], (2, 3))

    selected_experts_no_partitioning = jnp.argmax(
        jnp.dot(x, kernel.reshape(dim_x, num_experts)), -1)
    # Only one group, hence the first 0 index.
    self.assertNotEqual(selected_experts_no_partitioning[0, 0],
                        selected_experts[0, 0])
    self.assertNotEqual(selected_experts_no_partitioning[0, 1],
                        selected_experts[0, 1])

  def test_diag_blocks(self):
    """Test the utils to manipulate the block-diagonal representations."""
    rows = 2
    cols = 3
    ens_size = 4

    diag_blocks1 = jnp.asarray(
        [k * jnp.ones((rows, cols)) for k in range(1, ens_size + 1)])
    full_block_diag_matrix1 = jax.scipy.linalg.block_diag(*diag_blocks1)

    diag_blocks2 = jnp.asarray(
        [-k * jnp.ones((rows, cols)) for k in range(1, ens_size + 1)])
    full_block_diag_matrix2 = jax.scipy.linalg.block_diag(*diag_blocks2)

    diag_blocks12 = jnp.asarray([diag_blocks1, diag_blocks2])
    full_block_diag_matrix12 = jnp.asarray(
        [full_block_diag_matrix1, full_block_diag_matrix2])

    def dist(u, v):
      return float(jnp.sum(jnp.abs(u - v)))

    out12 = ens_routing.reshape_from_diag_blocks(diag_blocks12)
    self.assertAlmostEqual(dist(out12, full_block_diag_matrix12), 0.0)

if __name__ == '__main__':
  absltest.main()
