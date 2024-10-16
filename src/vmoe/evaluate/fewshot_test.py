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

"""Tests for fewshot."""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import clu.data
import jax
import numpy as np
import tensorflow as tf
from vmoe.evaluate import fewshot


class ComputeFewshotMetricsTest(parameterized.TestCase):

  # In both these examples the data is intrinsically unidimensional. There are
  # two classes, 6 training examples and 2 test examples. The first two examples
  # of the training set have class id=0, the next two have class id=1, and the
  # last two are masked out.
  #
  # After normalization, ignoring redundant examples and dims, the training
  # data reduces to [-0.63245553, -1.26491106, 1.26491106, 0.63245553].
  #
  # With no regularization penalty, the first test case simply reduces to
  # solving a linear regression problem, which results in two linear
  # discriminant functions (one per class):
  # fn_0(x) = -0.9486893 * x, and fn_1(x) = +0.9486893 * x.
  #
  # The second test case reduces to solving a ridge regression problem, which
  # results in the linear discriminant functions:
  # fn_0(x) = -0.2710640 * x, and fn_1(x) = +0.2710640 * x.
  #
  # The test cases (after normalization) are [-0.63245553, -1.26491106], with
  # class ids [0, 1]. This results in one correct classification (the first) and
  # one mistake (the second).
  @parameterized.named_parameters(
      ('ndim_lt_npoints_l2_reg_0.0', 1, 0.0),
      ('ndim_gt_npoints_l2_reg_10.0', 10, 10.0),
  )
  def test(self, ndim, l2_reg):
    # Fake representation_fn for testing purposes.
    def fake_representation_fn(state, variables, images, labels, mask):
      del variables
      return state, images, labels, mask
    # Training and test examples, all have implicit ndim=1, the rest of dims
    # have a constant value of 0.
    tr_feats = np.asarray([[-1.], [-2.], [2.], [1.], [1e9], [1e9]])
    tr_feats = np.pad(tr_feats, [(0, 0), (0, ndim - 1)])
    te_feats = np.asarray([[-1.], [-2.]])
    te_feats = np.pad(te_feats, [(0, 0), (0, ndim - 1)])
    tr_data = [{
        'image': np.asarray(tr_feats, dtype=np.float32),
        'label': np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int32),
        fewshot.VALID_KEY: np.asarray([1, 1, 1, 1, 0, 0], dtype=np.bool_),
    }]
    te_data = [{
        'image': np.asarray(te_feats, dtype=np.float32),
        'label': np.asarray([0, 1], dtype=np.int32),
        fewshot.VALID_KEY: np.asarray([1, 1], dtype=np.bool_),
    }]
    results = fewshot._compute_fewshot_metrics(
        representation_fn=fake_representation_fn,
        shots=[2],
        l2_regs=[l2_reg],
        num_classes=2,
        state=mock.create_autospec(fewshot.FewShotState, autospec=True),
        variables={},
        tr_iter=tr_data,
        te_iter=te_data,
    )
    chex.assert_trees_all_close(results, {(2, l2_reg): 0.5})


class FewShotPeriodicActionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Mock tfds.builder(), which is used to get the number of classes of a given
    # dataset.
    self.mock_tfds_builder = self.enter_context(
        mock.patch.object(fewshot.tfds, 'builder', autospec=True))
    mock_label_info = mock.MagicMock()
    mock_label_info.num_classes = 10
    mock_info = mock.MagicMock()
    mock_info.info.features = {'label': mock_label_info}
    self.mock_tfds_builder.return_value = mock_info

    # 4 batches of 16 images each. The last 4 images are fake.
    images = np.tile(np.arange(4 * 16).reshape((4, 16, 1, 1, 1)),
                     (1, 1, 32, 32, 3)).astype(np.float32)
    labels = np.mod(np.arange(4 * 16).reshape((4, 16)).astype(np.int32), 10)
    valid = np.concatenate([
        np.ones((3, 16)),
        np.concatenate([np.ones((1, 12)), np.zeros((1, 4))], axis=1),
    ], axis=0).astype(bool)
    dataset = tf.data.Dataset.from_tensor_slices({
        'image': images,
        'label': labels,
        fewshot.VALID_KEY: valid,
    })
    self.mock_get_dataset = self.enter_context(
        mock.patch.object(
            fewshot.vmoe.data.input_pipeline,
            'get_dataset',
            side_effect=lambda *a, **kw: clu.data.TfDatasetIterator(  # pylint: disable=g-long-lambda
                dataset, checkpoint=False)))

  @classmethod
  def _apply_fn(cls, variables, images, rngs=None):
    del variables, rngs
    features = images.reshape((images.shape[0], -1))[:, :16]
    metrics = {}
    return features, metrics

  def test_periodic_action_metric_writer(self):
    mock_metric_writer = mock.MagicMock(fewshot.metric_writers.MetricWriter)
    periodic_action = fewshot.FewShotPeriodicAction(
        metric_writer=mock_metric_writer,
        datasets={'foo': ('tfds_name', 'train_split', 'test_split')},
        apply_fn=self._apply_fn,
        shots=[2, 5],
        l2_regs=[0.01],
        main_task=('foo', 2),
        rng_keys=(),
        every_steps=4)
    with jax.sharding.Mesh(np.asarray(jax.local_devices()), ('d',)):
      for step in range(1, 10):
        periodic_action(step=step, variables={})
    call_args_list = mock_metric_writer.write_scalars.call_args_list
    self.assertLen(call_args_list, 2)
    expected_metrics = {
        '0/foo/2shot': mock.ANY,
        'fewshot/2shot_best_l2': mock.ANY,
        'fewshot/5shot_best_l2': mock.ANY,
        'fewshot/foo/duration_secs': mock.ANY,
        'fewshot/foo/2shot': mock.ANY,
        'fewshot/foo/5shot': mock.ANY,
        'fewshot/duration_secs': mock.ANY,
    }
    self.assertEqual(call_args_list[0], mock.call(4, expected_metrics))
    self.assertEqual(call_args_list[1], mock.call(8, expected_metrics))

  def test_periodic_action_metric_writer_multiple_seeds(self):
    mock_metric_writer = mock.MagicMock(fewshot.metric_writers.MetricWriter)
    periodic_action = fewshot.FewShotPeriodicAction(
        metric_writer=mock_metric_writer,
        datasets={'foo': ('tfds_name', 'train_split', 'test_split')},
        apply_fn=self._apply_fn,
        shots=[2, 5],
        l2_regs=[0.01],
        main_task=('foo', 2),
        rng_keys=(),
        every_steps=4,
        seeds_per_step=2)
    with jax.sharding.Mesh(np.asarray(jax.local_devices()), ('d',)):
      for step in range(1, 10):
        periodic_action(step=step, variables={})
    call_args_list = mock_metric_writer.write_scalars.call_args_list
    self.assertLen(call_args_list, 2)
    expected_metrics = {
        '0/foo/2shot': mock.ANY,
        'fewshot/2shot_best_l2': mock.ANY,
        'fewshot/5shot_best_l2': mock.ANY,
        'fewshot/foo-seed-0/duration_secs': mock.ANY,
        'fewshot/foo-seed-0/2shot': mock.ANY,
        'fewshot/foo-seed-0/5shot': mock.ANY,
        'fewshot/foo-seed-1/duration_secs': mock.ANY,
        'fewshot/foo-seed-1/2shot': mock.ANY,
        'fewshot/foo-seed-1/5shot': mock.ANY,
        'fewshot/duration_secs': mock.ANY,
    }
    self.assertEqual(call_args_list[0], mock.call(4, expected_metrics))
    self.assertEqual(call_args_list[1], mock.call(8, expected_metrics))

  def test_periodic_action_report_progress(self):
    mock_metric_writer = mock.MagicMock(fewshot.metric_writers.MetricWriter)
    mock_report_progress = mock.MagicMock(
        fewshot.periodic_actions.ReportProgress)
    periodic_action = fewshot.FewShotPeriodicAction(
        metric_writer=mock_metric_writer,
        datasets={'foo': ('tfds_name', 'train_split', 'test_split')},
        apply_fn=self._apply_fn,
        shots=[2, 5],
        l2_regs=[0.01],
        rng_keys=(),
        every_steps=4,
        report_progress=mock_report_progress,
        report_progress_name='fewshot')
    with jax.sharding.Mesh(np.asarray(jax.local_devices()), ('d',)):
      for step in range(1, 10):
        periodic_action(step=step, variables={})
    call_args_list = mock_report_progress.timed.call_args_list
    self.assertLen(call_args_list, 1)
    self.assertEqual(call_args_list[0],
                     mock.call('fewshot', wait_jax_async_dispatch=False))


if __name__ == '__main__':
  absltest.main()
