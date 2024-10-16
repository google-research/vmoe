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

"""Tests for the Soft MoE implementation."""
from absl.testing import absltest
import chex
import flax.core
import jax
import jax.numpy as jnp
import numpy as np
import scipy.spatial
from vmoe.projects.soft_moe import router

pdist = scipy.spatial.distance.pdist
squareform = scipy.spatial.distance.squareform


class SoftRouterTest(absltest.TestCase):

  def test_cosine_psim_matrix(self):
    """Tests cosine pairwise similarity."""
    # A "vector" is one of the subarrays on axes (3, 4) after being flattened,
    # each "vector" has 6 elements.
    # The pairwise product is done along the axes (1, 2). That is, there are
    # 5 * 4 "vectors".
    # Subarrays along the first axis are processed independently.
    x = jax.random.normal(jax.random.PRNGKey(0), (6, 5, 4, 3, 2))
    y = router.cosine_psim(x, (0,), (3, 4))
    self.assertEqual(y.shape, (6, 5, 4, 5, 4))

    scipy_cosine_psim = lambda x: 1 - squareform(pdist(x, metric='cosine'))
    expected = np.stack([
        scipy_cosine_psim(x_i.reshape((5 * 4, -1))).reshape((5, 4, 5, 4))
        for x_i in x
    ])
    np.testing.assert_allclose(y, expected, rtol=1e-4)

  def test_forward(self):
    x = jnp.ones((1, 8, 16), dtype=jnp.float32)
    layer = router.SoftRouter(num_experts=5)
    variables = flax.core.freeze(layer.init(jax.random.PRNGKey(0), x))
    expected_variables_shape = flax.core.freeze({
        'params': {
            'mu': jax.ShapeDtypeStruct((16, 5, 2), jnp.float32),
            'scale': jax.ShapeDtypeStruct((), jnp.float32),
        }
    })
    chex.assert_trees_all_equal_shapes_and_dtypes(variables,
                                                  expected_variables_shape)
    dispatcher, _ = layer.apply(variables, x)
    self.assertIsInstance(dispatcher, router.Bfloat16Dispatcher)

  def test_metrics(self):
    """Tests the metrics of the router."""
    x = np.asarray([[
        [[1, 0], [0, 1], [0, 0]],
        [[0, 0], [1, 0], [0, 0]],
        [[0, 1], [0, 0], [1, 0]],
        [[0, 0], [0, 0], [0, 1]],
    ]], dtype=np.float32)
    mu = np.asarray([
        [[1, 0], [2, 1], [-1, 0]],
        [[0, 1], [0, 1], [0, -1]],
    ], dtype=np.float32)
    metrics = router.SoftRouter(num_experts=3).get_metrics(x, x, mu)
    expected = {
        'auxiliary_loss': 0.,
        'combine_weights_min_mean': 0.,
        'combine_weights_max_mean': 1.,
        'dispatch_weights_min_mean': 0.,
        'dispatch_weights_max_mean': 1.,
        # All "rows" in the matrix x are independent.
        'combine_weights_similarity_min': 0.,
        'combine_weights_similarity_max': 0.,
        'combine_weights_similarity_mean': 0.,
        # "Columns" {0, 3}, and {1, 4} in the matrix x are identical.
        'dispatch_weights_similarity_min': 0.,
        'dispatch_weights_similarity_max': 1.,
        'dispatch_weights_similarity_mean': 4 / 30,
        # Similarity of columns in mu.
        'mu_similarity_min': -1.,  # "Columns" {0, 4} and {1, 5}.
        'mu_similarity_max': 1.,   # "Columns" {0, 2}.
        'mu_similarity_mean': (1 / np.sqrt(2) - 2)  / 15,
    }
    chex.assert_trees_all_close(metrics, expected)


if __name__ == '__main__':
  absltest.main()
