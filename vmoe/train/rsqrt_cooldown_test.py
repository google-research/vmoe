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
"""Regression coverage for learning rates after the cooldown endpoint."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import optax
from vmoe.train import optimizer
from vmoe.train import schedule


class RsqrtCooldownTest(parameterized.TestCase):

  @parameterized.product(warmup_steps=(0, 4),
                         cooldown_steps=(1, 3),
                         jit=(False, True))
  def test_cooldown_stays_at_zero(self, warmup_steps, cooldown_steps, jit):
    decay_steps, timescale, peak = 12, 2, 0.1
    fn = schedule.big_vision_rsqrt_schedule(peak_value=peak,
                                            decay_steps=decay_steps,
                                            timescale=timescale,
                                            warmup_steps=warmup_steps,
                                            cooldown_steps=cooldown_steps)
    evaluate = jax.jit(fn) if jit else fn
    counts = np.array([
        0, warmup_steps, decay_steps - cooldown_steps, decay_steps - 0.5,
        decay_steps, decay_steps + 1, 2 * decay_steps
    ],
                      dtype=np.float32)
    cooldown_start = decay_steps - cooldown_steps
    cooldown_peak = peak / np.sqrt(1 +
                                   (cooldown_start - warmup_steps) / timescale)
    expected = []
    for count in counts:
      if count < warmup_steps:
        value = peak * count / warmup_steps
      elif count < cooldown_start:
        value = peak / np.sqrt(1 + (count - warmup_steps) / timescale)
      else:
        value = cooldown_peak * max((decay_steps - count) / cooldown_steps, 0.)
      expected.append(value)
    actual = np.array([evaluate(jnp.asarray(count)) for count in counts])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(actual[-3:], 0.)

  @parameterized.parameters(False, True)
  def test_optimizer_stops_changing_parameters_after_cooldown(self, jit):
    tx = optimizer.create_optimizer(name='sgd',
                                    total_steps=6,
                                    momentum=0.9,
                                    weight_decay=0.1,
                                    learning_rate=dict(
                                        schedule='big_vision_rsqrt',
                                        peak_value=0.1,
                                        timescale=2,
                                        warmup_steps=1,
                                        cooldown_steps=2))
    params = {'w': jnp.array([1., -2.])}
    state = tx.init(params)
    update = jax.jit(tx.update) if jit else tx.update
    for step in range(10):
      grads = jax.grad(lambda p: jnp.sum(p['w']**2))(params)
      updates, state = update(grads, state, params)
      next_params = optax.apply_updates(params, updates)
      if step >= 6:
        np.testing.assert_array_equal(updates['w'], 0.)
        np.testing.assert_array_equal(next_params['w'], params['w'])
      params = next_params

  @parameterized.parameters(0, 4)
  def test_no_cooldown_retains_inverse_sqrt_tail(self, warmup_steps):
    fn = schedule.big_vision_rsqrt_schedule(peak_value=.1,
                                            decay_steps=12,
                                            timescale=2,
                                            warmup_steps=warmup_steps,
                                            cooldown_steps=0)
    for count in [12, 13, 24]:
      expected = .1 / np.sqrt(1 + (count - warmup_steps) / 2)
      np.testing.assert_allclose(fn(count), expected, rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
