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

"""Module with several util functions."""
import collections.abc
import functools
from typing import Any, Callable, Dict, Iterator, Tuple, Type

import jax
import jax.numpy as jnp

PRNGKey = jax.random.KeyArray


@functools.lru_cache()
def make_rngs(rng_keys: Tuple[str, ...], seed: int = 0) -> Dict[str, PRNGKey]:
  """Creates a dictionary of PRNGKeys from a tuple of key names and a seed."""

  @functools.partial(jax.jit, backend='cpu')
  def _make_rngs():
    if not rng_keys:
      return dict()
    rngs = jax.random.split(jax.random.PRNGKey(seed), len(rng_keys))
    return dict(zip(rng_keys, rngs))

  return _make_rngs()


def multiply_no_nan(x, y):
  """Multiplies x and y and returns 0 if x is 0, even if y is not finite."""
  # Note: This is equivalent to tf.math.multiply_no_nan, with safe gradients.
  x_ok = x != 0.
  safe_x = jnp.where(x_ok, x, 1.)
  safe_y = jnp.where(x_ok, y, 1.)
  return jnp.where(x_ok, jax.lax.mul(safe_x, safe_y), jnp.zeros_like(x))


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
