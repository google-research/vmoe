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

"""Tests for periodic_actions."""
import contextlib
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from vmoe.train import periodic_actions


class SingleProcessPeriodicActionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_should_not_run', 0, False),
      ('_should_run', 1, True),
  )
  def test(self, process_index, expected_output):
    action1 = mock.MagicMock(periodic_actions.PeriodicAction)
    action1.return_value = True
    action2 = periodic_actions.SingleProcessPeriodicAction(
        periodic_action=action1, process_index=1)
    with mock.patch.object(periodic_actions.jax, 'process_index',
                           return_value=process_index):
      self.assertEqual(action2(step=3), expected_output)


class ReportProgressTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_without_memory_profile', False,
       [{'foo': 0.4, 'bar/foo': 0.04}, {'foo': 0.8, 'bar/foo': 0.08}]),
      ('_with_memory_profile', True,
       [{'foo': 0.4, 'bar/foo': 0.04, 'host_memory_mb': mock.ANY},
        {'foo': 0.8, 'bar/foo': 0.08, 'host_memory_mb': mock.ANY}]),
  )
  def test(self, available, expected):
    writer = mock.MagicMock(periodic_actions.periodic_actions.MetricWriter)
    with _mock_memory_profiler(available):
      action = periodic_actions.ReportProgress(writer=writer, every_steps=4)
      for step in range(1, 10):
        action(step=step,
               scalar_metrics={'foo': step / 10, 'bar': {'foo': step / 100}})
    call_arg_list = writer.write_scalars.call_args_list
    self.assertLen(call_arg_list, 2 * 2)
    self.assertEqual(call_arg_list[0],
                     mock.call(4, {'steps_per_sec': mock.ANY}))
    self.assertEqual(call_arg_list[1], mock.call(4, expected[0]))
    self.assertEqual(call_arg_list[2],
                     mock.call(8, {'steps_per_sec': mock.ANY}))
    self.assertEqual(call_arg_list[3], mock.call(8, expected[1]))


@contextlib.contextmanager
def _mock_memory_profiler(available):
  original_value = periodic_actions.memory_profiler
  if available:
    periodic_actions.memory_profiler = mock.MagicMock()
  else:
    periodic_actions.memory_profiler = None
  try:
    yield
  finally:
    periodic_actions.memory_profiler = original_value


if __name__ == '__main__':
  absltest.main()
