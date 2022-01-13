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

"""Module with input pipeline functions.

Most of these were originally implemented by: Lucas Beyer, Alex Kolesnikov,
Xiaohua Zhai and other collaborators from Google Brain Zurich.
"""
import ast
from typing import Any, Callable, Dict, Iterator

from absl import logging
import cachetools
import jax
import ml_collections
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import vmoe.data.pp_ops

DEFAULT_SHUFFLE_BUFFER = 50_000
Data = Dict[str, Any]


@cachetools.cached(
    cache={},
    key=lambda name, data_dir, _: cachetools.keys.hashkey(name, data_dir))
def _get_dataset_builder(name, data_dir, manual_dir):
  data_builder = tfds.builder(name=name, data_dir=data_dir)
  data_builder.download_and_prepare(
      download_config=tfds.download.DownloadConfig(manual_dir=manual_dir))
  return data_builder


def get_datasets(
    config: ml_collections.ConfigDict) -> Dict[str, tf.data.Dataset]:
  """Returns a dictionary of datasets to use for different variants."""
  datasets = {}
  for variant, variant_config in config.items():
    if not isinstance(variant_config, ml_collections.ConfigDict):
      raise TypeError(
          f'The config for the {variant!r} variant is not a ConfigDict.')
    datasets[variant] = get_data_from_tfds(config=variant_config,
                                           variant=variant)
  return datasets


def get_data_from_tfds(*, config: ml_collections.ConfigDict,
                       variant: str) -> tf.data.Dataset:
  """Returns a Tensorflow dataset from TFDS.

  Args:
    config: Dataset ConfigDict.
    variant: Variant (e.g. 'train', 'validation', ...).

  Returns:
    A tf.data.Dataset.
  """
  data_builder = _get_dataset_builder(config.name, config.get('tfds_data_dir'),
                                      config.get('tfds_manual_dir'))
  split_name, start, end, smaller_range = get_data_range(
      data_builder, config.split, jax.process_index(), jax.process_count())
  logging.info(
      'Process %d / %d will handle examples from %d to %d, from split %r of dataset %r.',
      jax.process_index(), jax.process_count(), start, end, split_name,
      config.name)
  data_range = tfds.core.ReadInstruction(split_name, start, end)
  read_config = tfds.ReadConfig(shuffle_seed=config.get('shuffle_seed'))
  data = data_builder.as_dataset(
      split=data_range,
      decoders={'image': tfds.decode.SkipDecoding()},
      shuffle_files=variant == 'train',
      read_config=read_config)
  # Optionally, cache loaded data.
  if config.get('cache') == 'loaded':
    data = data.cache()
  # Compute the batch size per process.
  if (config.batch_size % jax.process_count() or
      config.batch_size % jax.device_count()):
    raise ValueError(f'batch_size must divide the process and device count, '
                     f'but got {config.batch_size}, {jax.process_count()}, '
                     f'and {jax.device_count()} respectively.')
  batch_size_per_process = config.batch_size // jax.process_count()
  if variant == 'train':
    # Repeat training data forever.
    data = data.repeat()
    # Shuffle training data.
    shuffle_buffer = config.get('shuffle_buffer', DEFAULT_SHUFFLE_BUFFER)
    data = data.shuffle(shuffle_buffer, seed=config.get('shuffle_seed'))
  # Process data.
  data = data.map(
      map_func=get_data_process_fn(config.process),
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
      deterministic=False)
  if variant != 'train':
    # Other variants process each example only once.
    # All processes must iterate over the same number of examples, thus
    # processes with a smaller range need one extra padded example. After that,
    # the number of examples iterated has to be multiple of
    # batch_size_per_process, thus we add the extra fake examples if necessary.
    num_fake_examples = int(smaller_range)
    num_fake_examples += -(end - start + smaller_range) % batch_size_per_process
    add_fake_fn = lambda fake: (lambda x: {**x, '_fake': fake})
    data = data.map(add_fake_fn(False),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if num_fake_examples > 0:
      fake_data = data.take(1).map(add_fake_fn(True)).repeat(num_fake_examples)
      data = data.concatenate(fake_data)
  # Batch data.
  data = data.batch(batch_size_per_process, drop_remainder=True)
  # Optionally, cache data after batching.
  if config.get('cache') == 'batched':
    data = data.cache()
  # Optionally, prefetch data.
  prefetch = config.get('prefetch')
  if prefetch == 'autotune':
    prefetch = tf.data.experimental.AUTOTUNE
  data = data.prefetch(prefetch) if prefetch else data
  return data


def get_data_num_examples(config: ml_collections.ConfigDict) -> int:
  """Returns the total number of examples of a dataset specified by a config."""
  data_builder = tfds.builder(
      name=config.name, data_dir=config.get('tfds_data_dir'))
  data_builder.download_and_prepare(
      download_config=tfds.download.DownloadConfig(
          manual_dir=config.get('tfds_manual_dir')))
  # 1. Canonicalize input to absolute indices.
  abs_ri = tfds.core.ReadInstruction.from_spec(config.split).to_absolute(
      data_builder.info.splits)
  # 2. Make sure it's only 1 continuous block.
  assert len(abs_ri) == 1, 'Multiple non-continuous TFDS splits not supported'
  split_range = abs_ri[0]
  # 3. Get its start/end indices.
  num_examples = data_builder.info.splits[split_range.splitname].num_examples
  split_start = split_range.from_ or 0
  split_end = split_range.to or num_examples
  return split_end - split_start


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
    op_name, op_args, op_kwargs = _parse_process_op_str(op_str)
    op_fn = getattr(vmoe.data.pp_ops, op_name)(*op_args, **op_kwargs)
    ops.append(op_fn)

  def _process_fn(data: Data):
    if not isinstance(data, dict):
      raise TypeError(f'Argument `data` must be a dict, not {type(data)}')
    for op_fn in ops:
      data = op_fn(data)
    return data

  return _process_fn


def get_data_range(builder, split: str, process_id: int, process_count: int):
  """Returns a (sub)split adapted to a given process.

  The examples in the given `split` are partitioned into `process_count`
  subsets. If the total number of examples in the split is `total_examples`,
  all processes will handle at least `total_examples // process_count` of them.
  If the total number of examples is not a multiple of `process_count`, the
  first `total_examples % process_count` processes handle one extra example.

  Args:
    builder: TFDS dataset builder.
    split: String of the split to use (e.g. 'train'). It can be a partial
      contiguous split as well (e.g. 'train[10%:20%]' or 'train[:10000]').
    process_id: Id of the process to compute the range for.
    process_count: Number of processes.

  Returns:
    A tuple (split_name, start_index, end_index).
  """
  # 1. Canonicalize input to absolute indices.
  abs_ri = tfds.core.ReadInstruction.from_spec(split).to_absolute(
      builder.info.splits)
  # 2. Make sure it's only 1 continuous block.
  assert len(abs_ri) == 1, 'Multiple non-continuous TFDS splits not supported'
  full_range = abs_ri[0]
  # 3. Get its start/end indices.
  full_range_examples = builder.info.splits[full_range.splitname].num_examples
  full_start = full_range.from_ or 0
  full_end = full_range.to or full_range_examples
  # 4. Compute each host's subset.
  # Each host will handle at least `examples_per_host` examples. When the total
  # number of examples is not divisible by the number of processes, the first
  # `remainder` hosts will handle one extra example each.
  examples_per_host = (full_end - full_start) // process_count
  remainder = (full_end - full_start) % process_count
  start = full_start + examples_per_host * process_id
  start += process_id if process_id < remainder else remainder
  end = start + examples_per_host
  end += 1 if process_id < remainder else 0
  # If True, this range is one example smaller than other processes.
  smaller_range = remainder > 0 and process_id >= remainder
  return full_range.splitname, start, end, smaller_range


def make_dataset_iterator(dataset: tf.data.Dataset) -> Iterator[Dict[str, Any]]:
  """Returns an iterator over a TF Dataset."""

  def to_numpy(data):
    return jax.tree_map(lambda x: np.asarray(memoryview(x)), data)

  ds_iter = iter(dataset)
  ds_iter = map(to_numpy, ds_iter)
  return ds_iter


def _parse_process_op_str(string_to_parse):
  """Parses a process operation string.

  Args:
    string_to_parse: can be either an arbitrary name or function call
      (optionally with positional and keyword arguments).

  Returns:
    A tuple of input name, argument tuple and a keyword argument dictionary.
    Examples:
      "flip_lr" -> ("flip_lr", (), {})
      "onehot(25, on=1, off=-1)" -> ("onehot", (25,), {"on": 1, "off": -1})
  """
  expr = ast.parse(string_to_parse, mode='eval').body  # pytype: disable=attribute-error
  if not isinstance(expr, (ast.Call, ast.Name)):
    raise ValueError(
        f'The given string should be a name or a call, but a {type(expr)} was '
        f'parsed from the string {string_to_parse!r}')
  # Notes:
  # name="some_name" -> type(expr) = ast.Name
  # name="module.some_name" -> type(expr) = ast.Attribute
  # name="some_name()" -> type(expr) = ast.Call
  # name="module.some_name()" -> type(expr) = ast.Call
  if isinstance(expr, ast.Name):
    return string_to_parse, (), {}

  def _get_func_name(expr):
    if isinstance(expr, ast.Name):
      return expr.id
    else:
      raise ValueError(
          f'Type {type(expr)} is not supported in a function name, the string '
          f'to parse was {string_to_parse!r}')

  def _get_func_args_and_kwargs(call):
    args = tuple([ast.literal_eval(arg) for arg in call.args])
    kwargs = {
        kwarg.arg: ast.literal_eval(kwarg.value) for kwarg in call.keywords
    }
    return args, kwargs

  func_name = _get_func_name(expr.func)
  func_args, func_kwargs = _get_func_args_and_kwargs(expr)

  return func_name, func_args, func_kwargs
