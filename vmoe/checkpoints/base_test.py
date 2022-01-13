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

"""Tests for base."""
import os
import re
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from vmoe.checkpoints import base


class CheckpointsBaseTest(parameterized.TestCase):

  @parameterized.parameters(('prefix', 0, 1), ('prefix-00000-of-00001', 10, 20))
  def test_add_and_remove_shard_suffix(self, prefix, shard, shard_count):
    self.assertEqual(
        base.remove_shard_suffix(
            base.add_shard_suffix(prefix, shard, shard_count)), prefix)

  @parameterized.named_parameters(
      ('none', [], None),
      ('multiple', [4, 3], 'foo_4'),
  )
  @mock.patch.object(base, 'iterate_complete_steps_for_prefix')
  def test_find_latest_complete_checkpoint_for_prefix(
      self, mock_iterate_result, expected, mock_iterate):
    mock_iterate.return_value = mock_iterate_result
    self.assertEqual(
        base.find_latest_complete_checkpoint_for_prefix('foo'),
        expected)

  @parameterized.named_parameters(
      ('no_matches', [], [], None, False, []),
      # Return step 1 because 2 has tmp files.
      ('tmp_files_present',
       ['foo_1-0-of-1', 'foo_2'], ['.tmp.foo_2-0-of-1'], None, False, [1]),
      # Return step 1 because 2 is missing required suffix .bar.
      ('missing_suffix', ['foo_1.bar', 'foo_2'], [], ['.bar'], False, [1]),
      # Return step 1 and 2 in increasing/decreasing order.
      ('increasing', ['foo_1-0-of-1', 'foo_2-0-of-1'], [], [], False, [1, 2]),
      ('decreasing', ['foo_1-0-of-1', 'foo_2-0-of-1'], [], [], True, [2, 1]),
  )
  @mock.patch.object(base.gfile, 'glob')
  def test_iterate_complete_steps_for_prefix(
      self, files, tmp_files, suffixes, decreasing, expected, mock_glob):
    mock_glob.side_effect = [files, tmp_files]
    output = list(
        base.iterate_complete_steps_for_prefix('foo', suffixes, decreasing))
    self.assertEqual(output, expected)

  @mock.patch.object(base.gfile, 'glob')
  def test_iterate_complete_steps_for_prefix_raises_no_step(self, mock_glob):
    mock_glob.return_value = ['foo-0-of-1']
    with self.assertRaisesRegex(ValueError, 'does not contain a step number'):
      _ = list(base.iterate_complete_steps_for_prefix('foo'))

  def test_save_checkpoint_async(self):
    workdir = self.create_tempdir().full_path
    filepath = os.path.join(workdir, 'ckpt')
    base.save_checkpoint_async(filepath, np.asarray(1)).get()
    self.assertEqual(['ckpt'], list(os.listdir(workdir)))

  def test_save_and_restore_multiple_checkpoints(self):
    workdir = self.create_tempdir().full_path
    # Save checkpoints.
    filepath_tree_map = {
        workdir + '/checkpoint_1': np.arange(3),
        workdir + '/checkpoint_2': jnp.arange(16, dtype=jnp.bfloat16),
        workdir + '/checkpoint_3': {'a': np.arange(5), 'b': np.arange(6)},
    }
    base.save_multiple_checkpoints_async(filepath_tree_map).get()
    # Restore checkpoints without specifying the tree structure.
    restored_filepath_tree_map = base.restore_multiple_checkpoints(
        {key: None for key in filepath_tree_map})
    jax.tree_multimap(np.testing.assert_array_equal, restored_filepath_tree_map,
                      filepath_tree_map)
    # Restore checkpoints specifying the tree structure.
    restored_filepath_tree_map = base.restore_multiple_checkpoints(
        jax.tree_structure(filepath_tree_map).unflatten([1, 2, 3, 4]))
    jax.tree_multimap(np.testing.assert_array_equal, restored_filepath_tree_map,
                      filepath_tree_map)
    # Check that the bfloat16 is restored properly.
    self.assertEqual(
        restored_filepath_tree_map[workdir + '/checkpoint_2'].dtype,
        jnp.bfloat16)

  def test_remove_checkpoints(self):
    workdir = self.create_tempdir()
    filepaths = [
        workdir.create_file('checkpoint_1').full_path,
        workdir.create_file('checkpoint_2').full_path,
        workdir.create_file('checkpoint_3').full_path,
        workdir.create_file('checkpoint_4').full_path
    ]

    def match_fn(filepath):
      # Matches checkpoints from even steps.
      m = re.match('^(.*)_([0-9]+)$', filepath)
      return m is not None and int(m.group(2)) % 2 == 0

    base.remove_checkpoints(filepaths, match_fn)
    self.assertCountEqual(
        os.listdir(workdir.full_path), ['checkpoint_1', 'checkpoint_3'])


if __name__ == '__main__':
  absltest.main()
