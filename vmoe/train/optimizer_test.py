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

"""Tests for optimizer."""
import math

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from vmoe.train import optimizer


class AddDecayedWeightsTest(absltest.TestCase):

  def test_fixed_factor(self):
    init_fn, update_fn = optimizer.add_decayed_weights(0.1)
    state = init_fn('foo')
    self.assertIsInstance(state, optimizer.optax.AddDecayedWeightsState)
    grads = {'a': 0.1, 'b': 0.2}
    params = {'a': 0.1, 'b': 0.2}
    new_grads, new_state = update_fn(grads, state, params)
    self.assertIs(new_state, state)
    chex.assert_trees_all_close(
        new_grads, {'a': 0.1 + 0.1 * 0.1, 'b': 0.2 + 0.1 * 0.2}, rtol=1e-5)

  def test_variable_factor(self):
    weight_decay = [('a', 0.1), ('b', 0.2), ('.*z', 0.3)]
    grads = {'a': 0, 'bz': 0, 'cz': 0, 'd': 1}
    params = {'a': 1, 'bz': 2, 'cz': 3, 'd': 4}
    init_fn, update_fn = optimizer.add_decayed_weights(weight_decay)
    state = init_fn('foo')
    new_grads, _ = update_fn(grads, state, params)
    chex.assert_trees_all_close(
        new_grads, {'a': 0.1, 'bz': 0.4, 'cz': 0.9, 'd': 1}, rtol=1e-5)

  def test_raises(self):
    init_fn, update_fn = optimizer.add_decayed_weights(0.1)
    state = init_fn('foo')
    with self.assertRaisesRegex(ValueError, 'Not passing `params`'):
      update_fn({'a': 0.1}, state, None)


class CreateOptimizerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_sgd_clip_global_norm', {
          'name': 'sgd',
          'learning_rate': 0.1,
          'gradient_clip': {'global_norm': 1.},
      }, {'x': -0.54719, 'y': -1.54719}),
      ('_adam_clip_absolute_value', {
          'name': 'adam',
          'learning_rate': 0.1,
          'gradient_clip': {'absolute_value': 1.},
      }, {'x': -0.54719, 'y': -1.54719}),
      ('_sgd_weight_decay', {
          'name': 'sgd',
          'learning_rate': 0.1,
          'weight_decay': 0.0001,
      }, {'x': -0.54719, 'y': -1.54719}),
      ('_sgd_momentum_trainable', {
          'name': 'sgd',
          'momentum': 0.5,
          'learning_rate': 0.1,
          'trainable_pattern': 'x',
      }, {'x': 0.26782, 'y': 0.}),
  )
  def test(self, kwargs, expected):
    """Tests optimizers by minimizing the McCormick function for 200 steps.

    Check https://www.sfu.ca/~ssurjano/mccorm.html for details about the
    McCormick function.

    Args:
      kwargs: Optimizer kwargs.
      expected: Dictionary with the expected value of each variable.
    """

    @jax.jit
    def minimize():
      init_fn, update_fn = optimizer.create_optimizer(
          **kwargs, total_steps=200)

      def McCormick(params):  # pylint: disable=invalid-name
        x, y = params['x'], params['y']
        return jnp.sin(x + y) + jnp.square(x - y) - 1.5*x + 2.5*y + 1

      def step(_, state):
        params, tx_state = state
        grads = jax.grad(McCormick)(params)
        updates, new_tx_state = update_fn(grads, tx_state, params)
        new_params = optimizer.optax.apply_updates(params, updates)
        return new_params, new_tx_state

      params = jax.tree_map(jnp.asarray, {'x': 0., 'y': 0.})
      state = init_fn(params)
      return jax.lax.fori_loop(0, 200, step, (params, state))[0]

    chex.assert_trees_all_close(minimize(), expected, rtol=1e-4)

  def test_raises_unknown_optimizer(self):
    with self.assertRaisesRegex(ValueError, 'Unknown optimizer'):
      optimizer.create_optimizer(name='foo', learning_rate=0.1, total_steps=10)


class FreezeWeightsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_none', None, None, {'a': 1, 'b': 2, 'c': 3}),
      ('_trainable_str', '[ab]', None, {'a': 1, 'b': 2, 'c': 0}),
      ('_trainable_list', ['a', 'b'], None, {'a': 1, 'b': 2, 'c': 0}),
      ('_frozen_str', None, 'b|c', {'a': 1, 'b': 0, 'c': 0}),
      ('_frozen_list', None, ['b', 'c'], {'a': 1, 'b': 0, 'c': 0}),
  )
  def test(self, trainable, frozen, expected):
    init_fn, update_fn = optimizer.freeze_weights(trainable_pattern=trainable,
                                                  frozen_pattern=frozen)
    x = {'a': 1, 'b': 2, 'c': 3}
    y, _ = update_fn(x, init_fn(x))
    chex.assert_trees_all_close(y, expected)

  def test_raises(self):
    with self.assertRaisesRegex(ValueError, 'You cannot specify both'):
      optimizer.freeze_weights(
          trainable_pattern='foo', frozen_pattern='bar')


class GradientClippingTest(parameterized.TestCase):

  def test_raises(self):
    with self.assertRaisesRegex(ValueError, 'You must specify .* but not both'):
      optimizer.gradient_clipping(global_norm=0.1, absolute_value=0.2)

  @parameterized.named_parameters(
      ('_global_norm',
       {'global_norm': 2.0}, {'a': 2 / math.sqrt(5), 'b': 4 / math.sqrt(5)}),
      ('_absolute_value',
       {'absolute_value': 1.5}, {'a': 1, 'b': 1.5}),
  )
  def test(self, kwargs, expected):
    init_fn, update_fn = optimizer.gradient_clipping(**kwargs)
    x = {'a': 1., 'b': 2.}
    y, _ = update_fn(x, init_fn(x))
    chex.assert_trees_all_close(y, expected)


if __name__ == '__main__':
  absltest.main()
