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

"""Tests for attacks."""
import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from vmoe.projects.adversarial_attacks import attacks


class StatefulAttackTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_with_valid', [True, True, False],
       [.1, .2, .0], [.2, .2, .0], [1, 2, 0], [1, 2, 0]),
      ('_without_valid', None,
       [.1, .2, .3], [.2, .2, .2], [1, 2, 3], [1, 2, 2]),
  )
  def test(self, valid, expected_l_0, expected_l_m, expected_y_0, expected_y_m):
    combine_weights_0 = np.asarray([
        [[.3, .0], [.4, .1]],
        [[.0, .0], [.1, .2]],
        [[.0, .5], [.0, .4]]], dtype=np.float32)
    combine_weights_m = np.asarray([
        [[.3, .0], [.0, .1]],
        [[.0, .0], [.1, .0]],
        [[.0, .5], [.0, .4]]], dtype=np.float32)
    stateless_attack_fn = mock.MagicMock()
    stateless_attack_fn.side_effect = lambda x, _, rng: (x + 1., rng)
    compute_loss_predict_correct_cw_fn = mock.MagicMock()
    compute_loss_predict_correct_cw_fn.side_effect = [
        # Values in each tuple are: loss, prediction, correct, combine weights.
        (
            np.asarray(expected_l_0, dtype=np.float32),
            np.asarray(expected_y_0, dtype=np.int32),
            np.asarray([1, 1, 0], dtype=np.int32),
            {'foo': combine_weights_0},
        ),
        (
            np.asarray(expected_l_m, dtype=np.float32),
            np.asarray(expected_y_m, dtype=np.int32),
            np.asarray([0, 1, 0], dtype=np.int32),
            {'foo': combine_weights_m},
        ),
    ]
    state = attacks.AttackState.create(
        max_updates=2, router_keys=['foo'], rngs={})
    x = np.zeros((3,), dtype=np.float32)
    y = np.asarray([1, 2, 1], dtype=np.int32)
    valid = None if valid is None else np.asarray(valid)
    new_state, x_m, l_0, l_m, y_0, y_m, cw_0, cw_m = attacks.stateful_attack(
        state, x, y, valid, stateless_attack_fn=stateless_attack_fn,
        compute_loss_predict_correct_cw_fn=compute_loss_predict_correct_cw_fn)
    np.testing.assert_allclose(x_m, np.ones_like(x))
    np.testing.assert_allclose(l_0, expected_l_0)
    np.testing.assert_allclose(l_m, expected_l_m)
    np.testing.assert_allclose(y_0, expected_y_0)
    np.testing.assert_allclose(y_m, expected_y_m)
    # Check the values of the combine weights.
    expected_cw_0 = (
        combine_weights_0 * (1 if valid is None else valid[:, None, None]))
    expected_cw_m = (
        combine_weights_m * (1 if valid is None else valid[:, None, None]))
    np.testing.assert_allclose(cw_0['foo'], expected_cw_0)
    np.testing.assert_allclose(cw_m['foo'], expected_cw_m)
    # Check the values in the AttackState object.
    self.assertEqual(new_state.max_updates, 2)
    self.assertEqual(new_state.num_images, 3 if valid is None else 2)
    self.assertEqual(new_state.num_changes, 1 if valid is None else 0)
    np.testing.assert_array_equal(new_state.num_correct, [2, 1])
    np.testing.assert_allclose(new_state.sum_loss,
                               [.6, .6] if valid is None else [.3, .4])
    expected_sum_iou_experts = (1. + .5) / 2. + (.0 + .5) / 2.
    if valid is None:
      expected_sum_iou_experts += (1. + 1.) / 2.
    np.testing.assert_allclose(new_state.sum_iou_experts['foo'],
                               expected_sum_iou_experts)


class StatelessAttackPGDTest(parameterized.TestCase):

  # We maximize the squared norm of the difference x - y, given:
  # x = [[.1, .2, .0], [.0, .1, .3]]
  # y = [[.2, .0, .0], [.0, .1, .3]]

  #
  # Notice that:
  # d_loss/dx = [[-.2, .4, .0], [.0, .0, .0]]
  # d_aux/dx  = [[+.2, .4, .0], [.0, .1, .3]]
  #
  # Thus, the sign of the gradient is: [[-1, +1, 0], [0, 0, 0]].

  def test(self):
    x = np.asarray([[.1, .2, .0], [.0, .1, .3]], dtype=np.float32)
    y = np.asarray([[.2, .0, .0], [.0, .1, .3]], dtype=np.float32)

    def apply_fn(x, *, rngs):
      del rngs
      return x, {'auxiliary_loss': jax.numpy.square(x).sum()}

    def loss_fn(x, y, metrics):
      del metrics
      return jax.numpy.square(x - y).sum()

    pgd_attack_stateless_jitted = jax.jit(
        fun=functools.partial(
            attacks.stateless_attack_pgd, max_epsilon=0.1, num_updates=1,
            apply_fn=apply_fn, loss_fn=loss_fn))
    new_x, _ = pgd_attack_stateless_jitted(x, y, {})
    np.testing.assert_allclose(new_x, [[.0, .3, .0], [.0, .1, .3]])


class SumIntersectionOverUnionTest(absltest.TestCase):

  def test(self):
    # batch_size = 3, num_tokens = 2, num_experts = 4.
    a = np.asarray([
        [[.0, .0, .2, .4], [.1, .2, .3, .0]],
        [[.0, .2, .3, .0], [.5, .5, .3, .4]],
        [[.0, .0, .0, .0], [.0, .0, .0, .0]],  # Masked out example.
    ], dtype=np.float32)
    b = np.asarray([
        [[.0, .0, .4, .8], [.1, .2, .3, .1]],
        [[.0, .3, .0, .0], [.1, .0, .0, .0]],
        [[.0, .0, .0, .0], [.0, .0, .0, .0]],  # Masked out example.
    ], dtype=np.float32)
    value = jax.jit(attacks.sum_intersection_over_union)(a, b)
    expected_value = (1. + .75 + .5 + .25 + .0 + .0) / 2.
    self.assertAlmostEqual(value, expected_value)


if __name__ == '__main__':
  absltest.main()
