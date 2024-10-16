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

"""Tests for input_pipeline."""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import ml_collections
import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from vmoe.data import input_pipeline

builder = input_pipeline.vmoe.data.builder


class GetDataProcessFnTest(absltest.TestCase):

  def test(self):
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

  def test_getattr_error(self):
    """Tests that an exception is raised when no processing op is not found."""
    with self.assertRaises(AttributeError):
      input_pipeline.get_data_process_fn('some_inexistent_process_op')

  def test_wrong_dataset_error(self):
    """Tests that an exception is raised if the dataset is not made of dicts."""
    process_fn = input_pipeline.get_data_process_fn('copy("foo", "bar")')
    dataset = tf.data.Dataset.from_tensors(tf.convert_to_tensor([0.0]))
    with self.assertRaises(TypeError):
      dataset = dataset.map(process_fn)
      _ = next(iter(dataset))


class GetDataNumExamplesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_get_tfds_builder = self.enter_context(
        mock.patch.object(builder, '_get_tfds_builder', autospec=True))
    dummy_mnist = tfds.testing.DummyMnist(data_dir=self.create_tempdir())
    dummy_mnist.download_and_prepare()
    self.mock_get_tfds_builder.return_value = dummy_mnist

  @parameterized.named_parameters(
      ('_mnist', {'name': 'mnist', 'split': 'test'}, 20),
      ('_mnist_subsplit', {'name': 'mnist', 'split': 'test[1:6]'}, 5),
  )
  def test(self, config, expected_value):
    config = ml_collections.ConfigDict(config)
    self.assertEqual(
        input_pipeline.get_data_num_examples(config=config), expected_value)


class GetDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Make a DatasetBuilder that returns a dataset with three examples.
    mock_builder = mock.create_autospec(builder.DatasetBuilder, autospec=True)
    mock_builder.num_examples = 3
    mock_builder.as_dataset.return_value = tf.data.Dataset.range(3).map(
        lambda x: {'x': x})
    mock_builder.get_num_fake_examples.return_value = 2

    self.mock_get_dataset_builder = self.enter_context(
        mock.patch.object(builder, 'get_dataset_builder', autospec=True))
    self.mock_get_dataset_builder.return_value = mock_builder

  def test_train(self):
    # Dataset config.
    config = ml_collections.ConfigDict()
    config.name = 'foo'
    config.split = 'bar'
    config.batch_size = 5
    config.process = 'copy("x", "y")'
    data = input_pipeline.get_dataset(variant='train', **config)
    # Training data can be iterated forever.
    data = [x for _, x in zip(range(10), iter(data))]
    self.assertLen(data, 10)
    # Training data doesn't need fake examples.
    self.assertSetEqual(set(data[0].keys()), {'x', 'y'})
    # Check that data is shuffled.
    self.assertGreater(len(set(tuple(x['x']) for x in data)), 1)

  def test_eval(self):
    # Dataset config.
    config = ml_collections.ConfigDict()
    config.name = 'foo'
    config.split = 'bar'
    config.batch_size = 5
    config.process = 'copy("x", "y")'
    data = input_pipeline.get_dataset(variant='eval', **config)
    # Eval data is only iterated once, which in this case means 1 batch.
    data = [x for _, x in zip(range(10), iter(data))]
    self.assertLen(data, 1)
    # Eval data has a '__valid__' field, since fake examples were added.
    self.assertSetEqual(set(data[0].keys()),
                        {'x', 'y', input_pipeline.VALID_KEY})
    # Eval data is not shuffled, and the fake data corresponds to the first
    # element.
    self.assertTupleEqual(tuple(data[0]['x']), (0, 1, 2, 0, 0))
    self.assertTupleEqual(tuple(data[0][input_pipeline.VALID_KEY]),
                          (True, True, True, False, False))


class GetDatasetTests(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_get_dataset = self.enter_context(
        mock.patch.object(input_pipeline, 'get_dataset', autospec=True,
                          return_value=mock.MagicMock()))

  def test(self):
    extra_kwargs = {'batch_size': 1, 'process': 'foo|bar'}
    config = ml_collections.ConfigDict({
        'train': {'name': 'dataset_foo', 'split': 'tr', **extra_kwargs},
        'valid': {'name': 'dataset_bar', 'split': 'va', **extra_kwargs},
    })
    datasets = input_pipeline.get_datasets(config)
    self.assertSetEqual(set(datasets.keys()), {'train', 'valid'})

  def test_wrong_config_error(self):
    config = ml_collections.ConfigDict({'variant': 1})
    with self.assertRaisesRegex(TypeError, "The config for the 'variant'"):
      input_pipeline.get_datasets(config)


if __name__ == '__main__':
  absltest.main()
