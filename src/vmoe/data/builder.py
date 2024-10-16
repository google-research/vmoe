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

"""Dataset builders."""
import abc
import dataclasses
import functools
from typing import Optional, Sequence, Tuple

import cachetools
import jax
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

AbstractSplit = tfds.core.splits.AbstractSplit
ReadInstruction = tfds.core.ReadInstruction


class DatasetBuilder(abc.ABC):
  """Abstract dataset builder."""

  @property
  @abc.abstractmethod
  def num_examples(self) -> int:
    """Total number of examples in the dataset split."""

  @abc.abstractmethod
  def as_dataset(self) -> tf.data.Dataset:
    """Returns a dataset as a tf.data.Dataset object."""

  @abc.abstractmethod
  def get_num_fake_examples(self, batch_size_per_process: int) -> int:
    """Number of fake examples processed in the current process."""


@dataclasses.dataclass
class TfdsBuilder(DatasetBuilder):
  """Dataset builder for TFDS datasets.

  Attributes:
    name: Name of the dataset in TFDS.
    split: String with the split to use (e.g. 'train', 'validation[:100]', etc).
    data_dir: Optional directory where the data is stored. If None, it uses the
      default TFDS data dir.
    manual_dir: Optional directory where the raw data is stored. This is
      necessary to prepare some datasets (e.g. 'imagenet2012'), since TFDS does
      not suppport downloading them directly.
    shuffle_seed: Optional seed for shuffling files.
    shuffle_files: If True, shuffles files to process data in random order.
    try_gcs: If True, tries to download data from TFDS' Google Cloud bucket.
    ignore_errors: bool, if True, the tf.data.Dataset will ignore all errors.
  """
  name: str
  split: str
  data_dir: Optional[str] = None
  manual_dir: Optional[str] = None
  shuffle_files: bool = False
  shuffle_seed: Optional[int] = None
  try_gcs: bool = False
  ignore_errors: bool = False

  @property
  def num_examples(self) -> int:
    return self._tfds_builder.info.splits[self.split].num_examples

  def as_dataset(self) -> tf.data.Dataset:
    split = tfds.split_for_jax_process(self.split, drop_remainder=False)
    read_config = tfds.ReadConfig(
        shuffle_seed=self.shuffle_seed, skip_prefetch=True, try_autocache=False)
    data = self._tfds_builder.as_dataset(
        split=split,
        decoders={'image': tfds.decode.SkipDecoding()},
        shuffle_files=self.shuffle_files,
        read_config=read_config)
    if self.ignore_errors:
      data = data.ignore_errors()
    return data

  def get_num_fake_examples(self, batch_size_per_process: int) -> int:
    builder = self._tfds_builder
    num_examples = [
        builder.info.splits[split].num_examples for split in tfds.even_splits(
            self.split, jax.process_count(), drop_remainder=False)
    ]
    return _get_num_fake_examples(jax.process_index(), batch_size_per_process,
                                  num_examples)

  @property
  def _tfds_builder(self):
    return _get_tfds_builder(self.name, self.data_dir, self.manual_dir,
                             self.try_gcs)


def _get_num_fake_examples(process_index: int, batch_size_per_process: int,
                           num_examples_per_process: Sequence[int]) -> int:
  """Returns the number of fake examples to use in a given process."""
  assert process_index < len(num_examples_per_process)
  num_examples_max = max(num_examples_per_process)
  num_examples_process = num_examples_per_process[process_index]
  # All processes must iterate over the same number of examples.
  num_fake_examples = num_examples_max - num_examples_process
  # The number of examples must be a multiple of batch_size_per_process.
  num_fake_examples += (-num_examples_max) % batch_size_per_process
  return num_fake_examples


@cachetools.cached(
    cache={},
    key=lambda name, data_dir, *_: cachetools.keys.hashkey(name, data_dir))
def _get_tfds_builder(name, data_dir, manual_dir, try_gcs):
  if 'from_directory:' in name:
    return tfds.builder_from_directory(data_dir)
  data_builder = tfds.builder(name=name, data_dir=data_dir, try_gcs=try_gcs)
  data_builder.download_and_prepare(
      download_config=tfds.download.DownloadConfig(manual_dir=manual_dir))
  return data_builder


def get_dataset_builder(*, name: str, **kwargs) -> DatasetBuilder:
  """Returns the builder to use for a given dataset split."""
  return TfdsBuilder(name=name, **kwargs)
