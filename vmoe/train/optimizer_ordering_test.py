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
"""Weight decay must pair each gradient with its named parameter."""

from absl.testing import absltest
from absl.testing import parameterized
import flax.core
import jax
import jax.numpy as jnp
import numpy as np
import optax
from vmoe.train import optimizer


class WeightDecayOrderingTest(parameterized.TestCase):

  @parameterized.product(frozen=(False, True),
                         different_shapes=(False, True),
                         per_name=(False, True))
  def test_decay_matches_parameter_keys(self, frozen, different_shapes,
                                        per_name):
    params = {
        'z': {
            'weight': jnp.array([10., 20.]),
            'bias': jnp.array([7.])
        },
        'a': {
            'weight': jnp.arange(1., 4. if different_shapes else 3.)
        },
    }
    if frozen:
      params = flax.core.freeze(params)
    gradients = jax.grad(lambda p: sum(
        jnp.sum(x * x) for x in jax.tree_util.tree_leaves(p)))(params)
    decay = [('z/weight', .2), ('a/weight', .1)] if per_name else .1
    tx = optimizer.add_decayed_weights(decay)
    state = tx.init(params)
    expected = {
        'z': {
            'weight':
                2. * params['z']['weight'] +
                (.2 if per_name else .1) * params['z']['weight'],
            'bias': (2. if per_name else 2.1) * params['z']['bias']
        },
        'a': {
            'weight': 2.1 * params['a']['weight']
        },
    }
    for update in (tx.update, jax.jit(tx.update)):
      actual, _ = update(gradients, state, params)
      self.assertIsInstance(actual, type(gradients))
      for group in expected:
        for name in expected[group]:
          np.testing.assert_allclose(actual[group][name],
                                     expected[group][name],
                                     rtol=1e-6)

  @parameterized.parameters(False, True)
  def test_sgd_matches_keywise_reference(self, jit):
    params = {'z': jnp.array([10., 20.]), 'a': jnp.array([1., 2.])}
    tx = optimizer.create_optimizer(name='sgd',
                                    total_steps=3,
                                    learning_rate=.05,
                                    weight_decay=.1)
    state = tx.init(params)
    update = jax.jit(tx.update) if jit else tx.update
    for _ in range(3):
      gradients = jax.grad(lambda p: sum(jnp.sum(x * x) for x in p.values()))(
          params)
      expected = {
          k: p - .05 * (gradients[k] + .1 * p) for k, p in params.items()
      }
      updates, state = update(gradients, state, params)
      actual = optax.apply_updates(params, updates)
      for name in params:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-6)
      # Keep input insertion order distinct from JAX's flattened key order.
      params = {'z': actual['z'], 'a': actual['a']}


if __name__ == '__main__':
  absltest.main()
