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

"""Tests for input_pipeline."""
import collections
from unittest import mock

from absl.testing import absltest
import ml_collections
import numpy as np
import tensorflow as tf
from vmoe.data import input_pipeline


class InputPipelineTest(absltest.TestCase):

  def test_get_data_process_fn(self):
    """Tests that data process strings are parsed correctly."""
    process_fn = input_pipeline.get_data_process_fn(
        'value_range(-1, 1)|copy("image", outkey="bar")')
    dataset = tf.data.Dataset.from_tensors({'image': tf.zeros((4,))})
    dataset = dataset.map(process_fn)
    x = next(iter(dataset))
    self.assertIsInstance(x, dict)
    self.assertSetEqual(set(x.keys()), {'image', 'bar'})
    np.testing.assert_allclose(x['image'].numpy(), -1.)
    np.testing.assert_allclose(x['bar'].numpy(), -1.)

  def test_get_data_process_fn_getattr_error(self):
    """Tests that an exception is raised when no processing op is not found."""
    with self.assertRaises(AttributeError):
      input_pipeline.get_data_process_fn('some_inexistent_process_op')

  def test_get_data_process_fn_wrong_dataset_error(self):
    """Tests that an exception is raised if the dataset is not made of dicts."""
    process_fn = input_pipeline.get_data_process_fn('copy("foo", "bar")')
    dataset = tf.data.Dataset.from_tensors(tf.convert_to_tensor([0.0]))
    with self.assertRaises(TypeError):
      dataset = dataset.map(process_fn)
      _ = next(iter(dataset))

  def test_get_data_range(self):
    _SplitInfo = collections.namedtuple('SplitInfo', ('num_examples',))
    builder = mock.MagicMock()
    builder.info.splits = {'x': _SplitInfo(num_examples=11)}
    data_ranges = [
        input_pipeline.get_data_range(builder, 'x', i, 4) for i in range(4)
    ]
    self.assertTupleEqual(data_ranges[0], ('x', 0, 3, False))
    self.assertTupleEqual(data_ranges[1], ('x', 3, 6, False))
    self.assertTupleEqual(data_ranges[2], ('x', 6, 9, False))
    self.assertTupleEqual(data_ranges[3], ('x', 9, 11, True))

  def test_get_data_num_examples(self):
    builder = mock.MagicMock()
    _SplitInfo = collections.namedtuple('SplitInfo', ('num_examples',))
    builder.info.splits = {'bar': _SplitInfo(num_examples=3)}
    with mock.patch.object(
        input_pipeline.tfds, 'builder', return_value=builder):
      config = ml_collections.ConfigDict()
      config.name = 'foo'
      config.split = 'bar'
      self.assertEqual(
          input_pipeline.get_data_num_examples(config=config), 3)

  def test_get_data_from_tfds_train(self):
    # Mock a builder that returns a dataset with three examples.
    builder = mock.MagicMock()
    _SplitInfo = collections.namedtuple('SplitInfo', ('num_examples',))
    builder.info.splits = {'bar': _SplitInfo(num_examples=3)}
    builder.as_dataset.return_value = tf.data.Dataset.range(3).map(
        lambda x: {'x': x})
    with mock.patch.object(
        input_pipeline.tfds, 'builder', return_value=builder):
      # Dataset config.
      config = ml_collections.ConfigDict()
      config.name = 'foo'
      config.split = 'bar'
      config.batch_size = 5
      config.process = 'copy("x", "y")'
      data = input_pipeline.get_data_from_tfds(variant='train', **config)
      # Training data can be iterated forever.
      data = [x for _, x in zip(range(10), iter(data))]
      self.assertLen(data, 10)
      # Training data doesn't need fake examples.
      self.assertSetEqual(set(data[0].keys()), {'x', 'y'})
      # Check that data is shuffled.
      self.assertGreater(len(set(tuple(x['x'].numpy()) for x in data)), 1)

  def test_get_data_from_tfds_eval(self):
    # Mock a builder that returns a dataset with three examples.
    builder = mock.MagicMock()
    _SplitInfo = collections.namedtuple('SplitInfo', ('num_examples',))
    builder.info.splits = {'bar': _SplitInfo(num_examples=3)}
    builder.as_dataset.return_value = tf.data.Dataset.range(3).map(
        lambda x: {'x': x})
    with mock.patch.object(
        input_pipeline.tfds, 'builder', return_value=builder):
      # Dataset config.
      config = ml_collections.ConfigDict()
      config.name = 'foo'
      config.split = 'bar'
      config.batch_size = 5
      config.process = 'copy("x", "y")'
      data = input_pipeline.get_data_from_tfds(variant='eval', **config)
      # Eval data is only iterated once, which in this case means 1 batch.
      data = [x for _, x in zip(range(10), iter(data))]
      self.assertLen(data, 1)
      # Eval data has a '_fake' field, since fake examples were padded.
      self.assertSetEqual(set(data[0].keys()), {'x', 'y', '_fake'})
      # Eval data is not shuffled, and the fake data corresponds to the first
      # element.
      self.assertTupleEqual(tuple(data[0]['x'].numpy()), (0, 1, 2, 0, 0))
      self.assertTupleEqual(tuple(data[0]['_fake'].numpy()),
                            (False, False, False, True, True))

  def test_get_datasets(self):
    config = ml_collections.ConfigDict({
        'train': {'name': 'dataset_foo'},
        'valid': {'name': 'dataset_bar'},
    })
    with mock.patch.object(
        input_pipeline, 'get_data_from_tfds', return_value=mock.MagicMock()):
      datasets = input_pipeline.get_datasets(config)
      self.assertSetEqual(set(datasets.keys()), {'train', 'valid'})

  def test_get_datasets_wrong_config_error(self):
    config = ml_collections.ConfigDict()
    config.variant = 1
    with mock.patch.object(
        input_pipeline, 'get_data_from_tfds', return_value=mock.MagicMock()):
      with self.assertRaisesRegex(TypeError, "The config for the 'variant'"):
        input_pipeline.get_datasets(config)

  def test_make_dataset_iterator(self):
    dataset = tf.data.Dataset.from_tensor_slices({
        'a': tf.convert_to_tensor([0.1, 0.2, 0.3]),
    })
    dataset_iter = input_pipeline.make_dataset_iterator(dataset)
    batches = list(dataset_iter)
    self.assertIsInstance(batches[0]['a'], np.ndarray)
    self.assertIsInstance(batches[1]['a'], np.ndarray)
    self.assertIsInstance(batches[2]['a'], np.ndarray)
    self.assertAlmostEqual(batches[0]['a'], 0.1)
    self.assertAlmostEqual(batches[1]['a'], 0.2)
    self.assertAlmostEqual(batches[2]['a'], 0.3)


if __name__ == '__main__':
  absltest.main()
