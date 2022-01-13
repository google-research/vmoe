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

"""Tests for evaluator."""
from unittest import mock

from absl.testing import absltest
import jax
from jax.experimental import maps
import numpy as np
import tensorflow as tf
from vmoe.evaluate import evaluator

PartitionSpec = evaluator.PartitionSpec


class EvalStateTest(absltest.TestCase):

  def test(self):
    state = evaluator.EvalState(sum_correct=1, num=2, sum_loss=3, rngs={})
    new_state = state.update(correct=1, num=2, loss=3, rngs={'foo': 1})
    self.assertEqual(new_state.num, 4)
    self.assertEqual(new_state.sum_correct, 2)
    self.assertEqual(new_state.sum_loss, 6)
    self.assertEqual(new_state.rngs, {'foo': 1})


class EvaluatorTest(absltest.TestCase):

  @classmethod
  def _apply_fn(cls, _, x, rngs):
    del rngs
    return x, 0.

  @classmethod
  def _loss_fn(cls, x, _):
    return jax.numpy.sum(x, axis=tuple(range(1, x.ndim)))

  @classmethod
  def _label_pred_fn(cls, x):
    return jax.numpy.argmax(x, axis=-1)

  @classmethod
  def _create_dataset_and_expected_state(cls):
    n = 16 * jax.local_device_count()
    labels = tf.random.categorical(logits=[[0., 0.]], num_samples=n)[0]
    output = tf.random.categorical(logits=[[0., 0.]], num_samples=n)[0]
    loss = tf.random.uniform((n,), minval=0.0, maxval=10.0)
    fake = tf.concat([tf.ones((4,)), tf.zeros((n - 8,)), tf.ones((4,))], 0)
    dataset = tf.data.Dataset.from_tensor_slices(
        {
            'image': tf.one_hot(output, depth=2)[:, :] * loss[:, None],
            'labels': tf.one_hot(labels, depth=2),
            '_fake': fake,
        })
    dataset = dataset.batch(8, drop_remainder=True)
    expected_eval_state = evaluator.EvalState(
        num=(n - tf.reduce_sum(fake)).numpy(),
        sum_correct=tf.reduce_sum(
            tf.cast(labels == output, tf.float32) * (1 - fake)).numpy(),
        # Final loss will be the "loss" vector, since x[i,:] values are either
        # 0 or loss[i].
        sum_loss=tf.reduce_sum(loss * (1 - fake)).numpy(),
        rngs={})
    return dataset, expected_eval_state

  def test_evaluate_dataset(self):
    # Create random test dataset.
    dataset, expected_eval_state = self._create_dataset_and_expected_state()

    eval_step_pjit = evaluator.make_eval_step_pjit(
        params_axis_resources={},  # Model has no parameters.
        input_axis_resources=PartitionSpec('d',),
        apply_fn=self._apply_fn,
        loss_fn=self._loss_fn,
        label_pred_fn=self._label_pred_fn,
        rng_keys=[])
    # Run the evaluation.
    with maps.mesh(np.asarray(jax.local_devices()), ('d',)):
      eval_state = evaluator.evaluate_dataset(
          eval_step_pjit=eval_step_pjit,
          dataset=dataset,
          params={},  # Model has no parameters.
          rng_keys=[])
    # Check that the values of the EvalState are correct.
    self.assertAlmostEqual(eval_state.num, expected_eval_state.num)
    self.assertAlmostEqual(eval_state.sum_correct,
                           expected_eval_state.sum_correct)
    self.assertAlmostEqual(eval_state.sum_loss, expected_eval_state.sum_loss,
                           places=4)

  def test_evaluate_multiple_datasets(self):
    dataset1, _ = self._create_dataset_and_expected_state()
    dataset2, _ = self._create_dataset_and_expected_state()
    datasets = {'dataset1': dataset1, 'dataset2': dataset2}
    metric_writer = mock.MagicMock(evaluator.metric_writers.MetricWriter)
    with maps.mesh(np.asarray(jax.local_devices()), ('d',)):
      action = evaluator.EvaluateMultipleDatasets(
          apply_fn=self._apply_fn,
          loss_fn=self._loss_fn,
          label_pred_fn=self._label_pred_fn,
          params_axis_resources={},
          input_axis_resources=PartitionSpec('d'),
          datasets=datasets,
          metric_writer=metric_writer,
          rng_keys=[],
          every_steps=4)
      for step in range(1, 10):
        action(step=step, params={})
      call_args_list = metric_writer.write_scalars.call_args_list
      self.assertLen(call_args_list, 6)
      self.assertEqual(call_args_list[0],
                       mock.call(4, {'dataset1/compile_secs': mock.ANY}))
      self.assertEqual(call_args_list[1],
                       mock.call(4,
                                 {'dataset1/duration_secs': mock.ANY,
                                  'dataset1/loss': mock.ANY,
                                  'dataset1/prec@1': mock.ANY}))
      self.assertEqual(call_args_list[2],
                       mock.call(4, {'dataset2/compile_secs': mock.ANY}))
      self.assertEqual(call_args_list[3],
                       mock.call(4,
                                 {'dataset2/duration_secs': mock.ANY,
                                  'dataset2/loss': mock.ANY,
                                  'dataset2/prec@1': mock.ANY}))
      self.assertEqual(call_args_list[4],
                       mock.call(8,
                                 {'dataset1/duration_secs': mock.ANY,
                                  'dataset1/loss': mock.ANY,
                                  'dataset1/prec@1': mock.ANY}))
      self.assertEqual(call_args_list[5],
                       mock.call(8,
                                 {'dataset2/duration_secs': mock.ANY,
                                  'dataset2/loss': mock.ANY,
                                  'dataset2/prec@1': mock.ANY}))

  def test_evaluate_multiple_datasets_report_progress(self):
    mock_report_progress = mock.MagicMock(
        evaluator.periodic_actions.ReportProgress)
    dataset, _ = self._create_dataset_and_expected_state()
    datasets = {'dataset': dataset}
    metric_writer = mock.MagicMock(evaluator.metric_writers.MetricWriter)
    with maps.mesh(np.asarray(jax.local_devices()), ('d',)):
      action = evaluator.EvaluateMultipleDatasets(
          apply_fn=self._apply_fn,
          loss_fn=self._loss_fn,
          label_pred_fn=self._label_pred_fn,
          params_axis_resources={},
          input_axis_resources=PartitionSpec('d'),
          datasets=datasets,
          metric_writer=metric_writer,
          rng_keys=[],
          every_steps=4,
          report_progress=mock_report_progress,
          report_progress_name='foo')
      for step in range(1, 10):
        action(step=step, params={})
    call_args_list = mock_report_progress.timed.call_args_list
    self.assertLen(call_args_list, 1)
    self.assertEqual(call_args_list[0],
                     mock.call('foo', wait_jax_async_dispatch=False))


if __name__ == '__main__':
  absltest.main()
