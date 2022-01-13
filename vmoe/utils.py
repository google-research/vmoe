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

"""Module with several util functions."""
import collections.abc
from typing import Any, Callable, Iterator, Tuple, Type

import jax


def partialclass(cls: Type[Any], *base_args, **base_kwargs):
  """Builds a subclass with partial application of the given args and keywords."""

  class _NewClass(cls):

    def __init__(self, *args, **kwargs):
      bound_args = base_args + args
      bound_kwargs = base_kwargs.copy()
      bound_kwargs.update(kwargs)
      super().__init__(*bound_args, **bound_kwargs)

  return _NewClass


class SafeZipIteratorError(RuntimeError):
  pass


class SafeZipIterator:
  """Lazy zip over multiple iterators, ensuring that all have the same length."""

  def __init__(self, *iterators):
    self.iterators = tuple(
        i if isinstance(i, collections.abc.Iterator) else iter(i)
        for i in iterators)

  def __iter__(self):
    return self

  def __next__(self) -> Tuple[Any, ...]:
    stop = None
    elements = []
    for i, iterator in enumerate(self.iterators):
      try:
        elements.append(next(iterator))
        if stop is not None:
          break
      except StopIteration:
        stop = i
    if stop is not None and elements:
      raise SafeZipIteratorError(
          f'The {stop}-th iterator raised StopIteration before the rest')
    if not elements:
      raise StopIteration
    return tuple(elements)


def safe_map(f: Callable[..., Any], *iterables) -> Iterator[Any]:
  for args in SafeZipIterator(*iterables):
    yield f(*args)


def safe_zip(*iterables) -> Iterator[Tuple[Any, ...]]:
  return SafeZipIterator(*iterables)


def tree_rngs_split(rngs, num_splits=2):
  """Splits a PyTree of PRNGKeys into num_splits PyTrees."""
  rngs = jax.tree_map(lambda rng: jax.random.split(rng, num_splits), rngs)
  slice_rngs = lambda rngs, i: jax.tree_map(lambda rng: rng[i], rngs)
  return tuple(slice_rngs(rngs, i) for i in range(num_splits))


def tree_shape_dtype_struct(tree):
  """Converts a PyTree with array-like objects to jax.ShapeDtypeStruct."""
  def fn(x):
    shape, dtype = x.shape, x.dtype
    # Useful to convert Tensorflow Tensors.
    dtype = dtype.as_numpy_dtype if hasattr(dtype, 'as_numpy_dtype') else dtype
    return jax.ShapeDtypeStruct(shape=shape, dtype=dtype)
  return jax.tree_map(fn, tree)
