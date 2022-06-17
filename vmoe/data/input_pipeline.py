# Copyright 2022 Google LLC.
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
from typing import Any, Callable, Dict, Iterator, Optional, Union

import jax
import ml_collections
import numpy as np
import tensorflow as tf
import vmoe.data.builder
import vmoe.data.pp_ops

DEFAULT_SHUFFLE_BUFFER = 50_000
VALID_KEY = '__valid__'
Data = Dict[str, Any]
DatasetBuilder = vmoe.data.builder.DatasetBuilder


def get_datasets(
    config: ml_collections.ConfigDict) -> Dict[str, tf.data.Dataset]:
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
    prefetch: Optional[Union[int, str]] = None,
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,
    shuffle_seed: Optional[int] = None,
    **extra_builder_kwargs) -> tf.data.Dataset:
  """Returns a Tensorflow dataset.

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
    prefetch: If given, prefetches this number of batches.
    shuffle_buffer: Size of the shuffle buffer. Only used for training.
    shuffle_seed: Optional seed for shuffling files and examples.
    **extra_builder_kwargs: Additional kwargs passed to the DatasetBuilder.

  Returns:
    A tf.data.Dataset.
  """
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
    process_fn = _compose_fns(get_data_process_fn(process),
                              lambda x: {**x, VALID_KEY: True})
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
  return data


def get_data_num_examples(config: ml_collections.ConfigDict) -> int:
  """Returns the total number of examples of a dataset specified by a config."""
  # These are kwarg keys used when creating the pipeline, not the builder.
  pipeline_keys = ('variant', 'batch_size', 'process', 'cache',
                   'num_parallel_calls', 'prefetch', 'prefetch_device',
                   'shuffle_buffer')
  builder_kwargs = {
      k: v for k, v in config.to_dict().items() if k not in pipeline_keys
  }
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
    op_name, op_args, op_kwargs = _parse_process_op_str(op_str)
    op_fn = getattr(vmoe.data.pp_ops, op_name)(*op_args, **op_kwargs)
    ops.append(op_fn)

  return _compose_fns(*ops)


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


def _compose_fns(*fns):

  def fn(data: Data):
    if not isinstance(data, dict):
      raise TypeError(f'Argument `data` must be a dict, not {type(data)}')
    for f in fns:
      data = f(data)
    return data

  return fn
