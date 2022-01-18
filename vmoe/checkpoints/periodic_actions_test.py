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
import os
import random
import time
from unittest import mock

from absl.testing import absltest
from vmoe.checkpoints import periodic_actions


class PeriodicSaveCheckpointRemoveOldCheckpointsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.workdir = self.create_tempdir()
    for step in range(10):
      self.workdir.create_file('ckpt_with_no_step')
      self.workdir.create_file(f'not_a_ckpt_{step}')
      self.workdir.create_file(f'ckpt_main_{step}')
      self.workdir.create_file(f'ckpt_model_{step}-00000-of-00001')
      self.workdir.create_file(f'ckpt_host_{step}-00000-of-00002')
      self.workdir.create_file(f'ckpt_host_{step}-00001-of-00002')

  def test_remove_old_checkpoints_with_keep_multiple_of(self):
    periodic_actions.PeriodicSaveCheckpoint._remove_old_checkpoints(
        prefix=os.path.join(self.workdir.full_path, 'ckpt_'),
        keep_last=2,
        keep_multiple=5,
        thread_pool=None)
    expected = (['ckpt_with_no_step'] +
                [f'not_a_ckpt_{step}' for step in range(10)] +
                [f'ckpt_main_{step}' for step in [0, 5, 8, 9]] +
                [f'ckpt_model_{step}-00000-of-00001' for step in [0, 5, 8, 9]] +
                [f'ckpt_host_{step}-00000-of-00002' for step in [0, 5, 8, 9]] +
                [f'ckpt_host_{step}-00001-of-00002' for step in [0, 5, 8, 9]])
    self.assertCountEqual(expected, os.listdir(self.workdir.full_path))

  def test_remove_old_checkpoints_with_no_keep_multiples(self):
    periodic_actions.PeriodicSaveCheckpoint._remove_old_checkpoints(
        prefix=os.path.join(self.workdir.full_path, 'ckpt_'),
        keep_last=2,
        keep_multiple=0,
        thread_pool=None)
    expected = (['ckpt_with_no_step'] +
                [f'not_a_ckpt_{step}' for step in range(10)] +
                [f'ckpt_main_{step}' for step in [8, 9]] +
                [f'ckpt_model_{step}-00000-of-00001' for step in [8, 9]] +
                [f'ckpt_host_{step}-00000-of-00002' for step in [8, 9]] +
                [f'ckpt_host_{step}-00001-of-00002' for step in [8, 9]])
    self.assertCountEqual(expected, os.listdir(self.workdir.full_path))


class PeriodicSaveCheckpointTest(absltest.TestCase):

  @mock.patch.object(
      periodic_actions.checkpoints_partitioned, 'save_checkpoint')
  def test(self, mock_save_checkpoint):
    # When calling save_checkpoint, we do nothing but we'll wait a few seconds.
    def _save_checkpoint_side_effect(*args, thread_pool, **kwargs):
      del args
      del kwargs
      wait = lambda: time.sleep(random.randint(1, 3))
      return thread_pool.apply_async(wait)
    mock_save_checkpoint.side_effect = _save_checkpoint_side_effect
    # Run a few steps, calling saver on each step.
    prefix = os.path.join(self.create_tempdir().full_path, 'ckpt')
    saver = periodic_actions.PeriodicSaveCheckpoint(
        prefix=prefix,
        state_axis_resources={},
        every_steps=4)
    for step in range(1, 10):
      saver(step=step, state={})
    # Check that the saver was called twice, on steps 4 and 8.
    call_args_list = mock_save_checkpoint.call_args_list
    self.assertLen(call_args_list, 2)
    self.assertEqual(call_args_list[0],
                     mock.call(prefix=prefix + '_4', num_shards=0, mesh=None,
                               makedirs=False, overwrite=True, tree={},
                               axis_resources={}, thread_pool=mock.ANY))
    self.assertEqual(call_args_list[1],
                     mock.call(prefix=prefix + '_8', num_shards=0, mesh=None,
                               makedirs=False, overwrite=True, tree={},
                               axis_resources={}, thread_pool=mock.ANY))
    saver.__del__()

  def test_report_progress(self):
    mock_report_progress = mock.MagicMock(
        periodic_actions.periodic_actions.ReportProgress)
    # Run a few steps, calling saver on each step.
    prefix = os.path.join(self.create_tempdir().full_path, 'ckpt')
    saver = periodic_actions.PeriodicSaveCheckpoint(
        prefix=prefix,
        state_axis_resources={},
        every_steps=4,
        report_progress=mock_report_progress,
        report_progress_name='foo')
    for step in range(1, 10):
      saver(step=step, state={})
    call_args_list = mock_report_progress.timed.call_args_list
    self.assertLen(call_args_list, 1)
    self.assertEqual(call_args_list[0],
                     mock.call('foo', wait_jax_async_dispatch=False))


if __name__ == '__main__':
  absltest.main()
