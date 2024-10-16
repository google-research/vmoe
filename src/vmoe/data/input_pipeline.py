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

"""Module with input pipeline functions.

Most of these were originally implemented by: Lucas Beyer, Alex Kolesnikov,
Xiaohua Zhai and other collaborators from Google Brain Zurich.
"""
from typing import Any, Callable, Dict, Optional, Union

from absl import logging
from clu.data import dataset_iterator
import jax
import ml_collections
import tensorflow as tf
import vmoe.data.builder
import vmoe.data.pp_ops
import vmoe.utils

DEFAULT_SHUFFLE_BUFFER = 50_000
VALID_KEY = '__valid__'
ArraySpec = dataset_iterator.ArraySpec
ArraySpecDict = Dict[str, ArraySpec]
Data = Dict[str, Any]
DatasetBuilder = vmoe.data.builder.DatasetBuilder
DatasetIterator = dataset_iterator.DatasetIterator
TfDatasetIterator = dataset_iterator.TfDatasetIterator


def get_datasets(
    config: ml_collections.ConfigDict) -> Dict[str, DatasetIterator]:
  """Returns a dictionary of datasets to use for different variants."""
  datasets = {}
  for variant, variant_config in config.items():
    if not isinstance(variant_config, ml_collections.ConfigDict):
      raise TypeError(
          f'The config for the {variant!r} variant is not a ConfigDict.')
    variant_config = variant_config.to_dict()
    _ = variant_config.pop('prefetch_device', None)
    datasets[variant] = get_dataset(variant=variant, **variant_config)
  return datasets


def get_dataset(
    *,
    variant: str,
    name: str,
    split: str,
    batch_size: int,
    process: str,
    cache: Optional[str] = None,
    num_parallel_calls: int = 128,
    pre_filter_fn: Optional[Callable[[Data], bool]] = None,
    prefetch: Optional[Union[int, str]] = None,
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,
    shuffle_seed: Optional[int] = None,
    **extra_builder_kwargs) -> DatasetIterator:
  """Returns a dataset iterator.

  Args:
    variant: Variant (e.g. 'train', 'validation', ...).
    name: Name of the dataset in TFDS.
    split: String with the split to use (e.g. 'train', 'validation[:100]', etc).
    batch_size: (Global) batch size to use. We assume that this batch size is
      evenly split among all devices.
    process: String representing the processing operations to perform (e.g.
      'decode|resize(128)|flip_lr'. Check the available ops in `pp_ops.py`).
    cache: If 'loaded' caches the dataset after loading it. If 'batched',
      caches it after batching. If `None`, no caching is done.
    num_parallel_calls: Process this number of examples in parallel.
    pre_filter_fn: If given, filters the dataset according to this function.
    prefetch: If given, prefetches this number of batches.
    shuffle_buffer: Size of the shuffle buffer. Only used for training.
    shuffle_seed: Optional seed for shuffling files and examples.
    **extra_builder_kwargs: Additional kwargs passed to the DatasetBuilder.

  Returns:
    A DatasetIterator.
  """
  if variant == 'train' and shuffle_seed is not None:
    logging.error('Deterministic training is not supported but you specified '
                  'shuffle_seed=%d for training. This can potentially lead to '
                  'data being repeated if restarts happen during training.',
                  shuffle_seed)
  builder = vmoe.data.builder.get_dataset_builder(
      name=name,
      split=split,
      shuffle_files=variant == 'train',
      shuffle_seed=shuffle_seed,
      **extra_builder_kwargs)
  # Compute the batch size per process.
  if (batch_size % jax.process_count() or batch_size % jax.device_count()):
    raise ValueError(f'batch_size must divide the process and device count, '
                     f'but got {batch_size}, {jax.process_count()}, '
                     f'and {jax.device_count()} respectively.')
  batch_size_per_process = batch_size // jax.process_count()
  data = builder.as_dataset()
  data = data.filter(pre_filter_fn) if pre_filter_fn is not None else data
  # Optionally, cache loaded data.
  if cache == 'loaded':
    data = data.cache()
  if variant == 'train':
    # Repeat training data forever.
    data = data.repeat()
    # Shuffle training data.
    data = data.shuffle(shuffle_buffer, seed=shuffle_seed)
    # Process
    process_fn = get_data_process_fn(process)
  else:
    # Other variants process each example only once and include VALID_KEY to
    # differentiate real vs. fake examples (that are added later).
    def _mask_fn(data):
      return data | {VALID_KEY: data.get(VALID_KEY, True)}
    process_fn = _compose_fns(get_data_process_fn(process), _mask_fn)
  # Process data.
  data = data.map(
      map_func=process_fn,
      num_parallel_calls=num_parallel_calls,
      deterministic=False)
  if variant != 'train':
    num_fake_examples = builder.get_num_fake_examples(batch_size_per_process)
    if num_fake_examples > 0:
      fake_elem = tf.nest.map_structure(
          lambda spec: tf.zeros(spec.shape, spec.dtype), data.element_spec)
      fake_data = tf.data.Dataset.from_tensors(fake_elem)
      fake_data = fake_data.repeat(num_fake_examples).cache()
      data = data.concatenate(fake_data)
  # Batch data.
  data = data.batch(batch_size_per_process, drop_remainder=True)
  # Optionally, cache data after batching.
  if cache == 'batched':
    data = data.cache()
  # Optionally, prefetch data.
  if prefetch == 'autotune':
    prefetch = tf.data.experimental.AUTOTUNE
  data = data.prefetch(prefetch) if prefetch else data
  # Note: checkpointing of TfDatasetIterators is very disk and time consuming,
  # For now, we disable checkpointing of the dataset iterator until we have a
  # better alternative.
  return TfDatasetIterator(data, checkpoint=False)


def get_data_num_examples(config: ml_collections.ConfigDict) -> int:
  """Returns the total number of examples of a dataset specified by a config."""
  config = config.to_dict()
  # These are kwarg keys used when creating the pipeline, not the builder.
  pipeline_keys = ('variant', 'batch_size', 'process', 'cache',
                   'num_parallel_calls', 'prefetch', 'prefetch_device',
                   'shuffle_buffer', 'pre_filter_fn')
  builder_kwargs = {k: v for k, v in config.items() if k not in pipeline_keys}
  builder = vmoe.data.builder.get_dataset_builder(**builder_kwargs)
  return builder.num_examples


def get_data_process_fn(process_str: str) -> Callable[[Data], Data]:
  """Transforms a processing string into a function.

  The minilanguage is as follows: "fn1|fn2|fn3(4, kw='a')"

  Args:
    process_str: String representing the data pipeline.

  Returns:
    A processing function to use with tf.data.Dataset.map().
  """
  ops = []
  for op_str in process_str.split('|'):
    op_fn, op_args, op_kwargs = vmoe.utils.parse_call(op_str, vmoe.data.pp_ops)
    ops.append(op_fn(*op_args, **op_kwargs))
  return _compose_fns(*ops)


def _compose_fns(*fns):

  def fn(data: Data):
    if not isinstance(data, dict):
      raise TypeError(f'Argument `data` must be a dict, not {type(data)}')
    for f in fns:
      data = f(data)
    return data

  return fn
