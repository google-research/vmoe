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

"""Tests for schedule."""
from absl.testing import absltest
from absl.testing import parameterized
import chex
import numpy as np
from vmoe.train import schedule


class ScheduleTest(parameterized.TestCase):

  def test_create_learning_rate_schedule_constant(self):
    kwargs = {'schedule': 'constant', 'value': 0.1}
    fn = schedule.create_learning_rate_schedule(**kwargs, total_steps=5)
    values = np.asarray(list(map(fn, range(5))), dtype=np.float32)
    chex.assert_trees_all_close(values, np.asarray([0.1] * 5, dtype=np.float32))

  @parameterized.named_parameters(
      ('warmup_cosine_decay', 'warmup_cosine_decay', {}),
      ('warmup_linear_decay', 'warmup_linear_decay', {}),
      ('warmup_polynomial_decay', 'warmup_polynomial_decay', {'power': 0.5}),
  )
  def test_create_learning_rate_schedule(self, schedule_name, extra_kwargs):
    kwargs = {
        'schedule': schedule_name,
        'peak_value': 0.1,
        'warmup_steps': 4,
        'end_value': 0.01,
        **extra_kwargs,
    }
    fn = schedule.create_learning_rate_schedule(**kwargs, total_steps=8)
    values = np.asarray(list(map(fn, range(10))), dtype=np.float32)
    self.assertAlmostEqual(values[0], 0., places=5)
    self.assertAlmostEqual(values[4], 0.1, places=5)
    self.assertAlmostEqual(values[8], 0.01, places=5)
    self.assertAlmostEqual(values[9], 0.01, places=5)

  def test_create_learning_rate_warmup_exponential_decay(self):
    kwargs = {
        'schedule': 'warmup_exponential_decay',
        'peak_value': 0.1,
        'warmup_steps': 4,
        'transition_steps': 1,
        'decay_rate': 0.9,
    }
    fn = schedule.create_learning_rate_schedule(**kwargs, total_steps=8)
    values = np.asarray(list(map(fn, range(10))), dtype=np.float32)
    self.assertAlmostEqual(values[0], 0., places=5)
    self.assertAlmostEqual(values[4], 0.1, places=5)
    self.assertAlmostEqual(values[8], 0.1 * 0.9**4, places=5)
    self.assertAlmostEqual(values[9], 0.1 * 0.9**5, places=5)

  def test_create_learning_rate_schedule_raises(self):
    kwargs = {'schedule': 'foo_bar'}
    with self.assertRaisesRegex(ValueError, 'Unknown learning rate schedule'):
      schedule.create_learning_rate_schedule(**kwargs, total_steps=10)

  def test_warmup_polynomial_decay_schedule(self):
    fn = schedule.warmup_polynomial_decay_schedule(
        peak_value=0.1,
        end_value=0.01,
        power=0.5,
        warmup_steps=5,
        decay_steps=10)
    chex.assert_trees_all_close(
        np.asarray(list(map(fn, range(15))), dtype=np.float32),
        np.asarray([
            # Linear warmup
            0.0,
            0.02,
            0.04,
            0.06,
            0.08,
            0.1,
            # Sqrt decay (i.e. power = 0.5).
            (0.1 - 0.01) * (1 - 1 / 5)**0.5 + 0.01,
            (0.1 - 0.01) * (1 - 2 / 5)**0.5 + 0.01,
            (0.1 - 0.01) * (1 - 3 / 5)**0.5 + 0.01,
            (0.1 - 0.01) * (1 - 4 / 5)**0.5 + 0.01,
            # Extra steps beyond 'decay_steps'.
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
        ], dtype=np.float32),
        rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
