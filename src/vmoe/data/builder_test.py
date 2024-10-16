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

"""Tests for builder."""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow_datasets.public_api as tfds
from vmoe.data import builder


class GetDatasetBuilderTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_tfds', {'name': 'mnist', 'split': 'test'}, builder.TfdsBuilder),
  )
  def test(self, kwargs, expected_cls):
    self.assertIsInstance(builder.get_dataset_builder(**kwargs), expected_cls)


class TfdsBuilderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_get_tfds_builder = self.enter_context(
        mock.patch.object(builder, '_get_tfds_builder', autospec=True))
    dummy_mnist = tfds.testing.DummyMnist(data_dir=self.create_tempdir())
    dummy_mnist.download_and_prepare()
    self.mock_get_tfds_builder.return_value = dummy_mnist

  @parameterized.named_parameters(
      ('train', 'train', 20),
      ('train_10', 'train[5:15]', 10),
  )
  def test_num_examples(self, split, expected_value):
    self.assertEqual(
        builder.TfdsBuilder(name='mnist', split=split).num_examples,
        expected_value)

  def test_as_dataset(self):
    ds = builder.TfdsBuilder(name='mnist', split='train[5:10]').as_dataset()
    self.assertLen(list(ds), 5)

  @parameterized.named_parameters(
      ('batch_size_4_process_0', 4, 0, 1),
      ('batch_size_4_process_1', 4, 1, 1),
      ('batch_size_4_process_2', 4, 2, 2),
  )
  def test_get_num_fake_examples(self, batch_size_per_process, process_index,
                                 expected_value):
    with mock.patch.object(builder.jax, 'process_count', return_value=3):
      with mock.patch.object(
          builder.jax, 'process_index', return_value=process_index):
        b = builder.TfdsBuilder(name='mnist', split='test')
        self.assertEqual(b.get_num_fake_examples(batch_size_per_process),
                         expected_value)


if __name__ == '__main__':
  absltest.main()
