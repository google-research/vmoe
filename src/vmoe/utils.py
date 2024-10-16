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

"""Module with several util functions."""
import ast
import collections.abc
import importlib
import re
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, Type, Union
import jax
import jax.numpy as jnp

PRNGKey = jax.Array


def get_flops_and_seconds_per_device(
    compiled_fn
) -> Tuple[float | None, float | None]:
  """Returns the FLOPs and optimal seconds per device of a compiled function."""
  cost_analysis = compiled_fn.cost_analysis()[0]
  flops_per_device = cost_analysis.get('flops')
  seconds_per_device = cost_analysis.get('optimal_seconds')
  # Note: XLA returns negative FLOPs and optimal_seconds for some platforms
  # (e.g. GPUs).
  if flops_per_device is not None and flops_per_device <= 0:
    flops_per_device = None
  if seconds_per_device is not None and seconds_per_device <= 0:
    seconds_per_device = None
  return flops_per_device, seconds_per_device


def make_rngs(rng_keys: Tuple[str, ...], seed: int) -> Dict[str, PRNGKey]:
  if not rng_keys:
    return dict()
  rngs = jax.random.split(jax.random.PRNGKey(seed), len(rng_keys))
  return dict(zip(rng_keys, rngs))


def multiply_no_nan(x, y):
  """Multiplies x and y and returns 0 if x is 0, even if y is not finite."""
  # Note: This is equivalent to tf.math.multiply_no_nan, with safe gradients.
  x_ok = x != 0.
  safe_x = jnp.where(x_ok, x, 1.)
  safe_y = jnp.where(x_ok, y, 1.)
  return jnp.where(x_ok, jax.lax.mul(safe_x, safe_y), jnp.zeros_like(x))


def parse_call(string: str, default_module: Union[str, Any]):
  """Parses a string representing a call.

  Examples:
    - parse_call('foo', module): Returns (module.foo, (), {}).
    - parse_call('foo(25)', module): Returns (module.foo, (25,), {}).
    - parse_call('foo.bar.baz', module): Returns ('foo.bar.baz', (), {}).

  Args:
    string: This can be either a name (it assumes no arguments) or a call string
      including the positional and keyword arguments. The call cannot include
      nested calls (e.g. "foo.bar().baz()" is not allowed). The optional args
      must be Python literals.
    default_module: Default module to use to import the function.

  Returns:
    Returns the callable (e.g. a class or function), a tuple of positional args,
    and a dictionary of keyword arguments.
  """
  expr = ast.parse(string, mode='eval').body
  if isinstance(expr, ast.Call):
    # Parses the positional and keyword arguments in strings like:
    # "foo.bar.baz(a, b=c)".
    args = tuple([ast.literal_eval(arg) for arg in expr.args])
    kwargs = {
        kwarg.arg: ast.literal_eval(kwarg.value) for kwarg in expr.keywords
    }
    # Prepare to process the rest of the expression (e.g. "foo.bar.baz").
    expr = expr.func
  else:
    args, kwargs = (), {}
  if isinstance(expr, ast.Name):
    # After the (optional) call arguments, the expression is a name: e.g. "foo".
    module = default_module
    name = expr.id
  elif isinstance(expr, ast.Attribute):
    name = expr.attr
    expr = expr.value
    module = string[:expr.end_col_offset]
    # We check that the expression is something like:
    # "name.attribute_1.....attribute_n".
    # For instance, something like "foo.bar().baz" is not be accepted.
    while not isinstance(expr, ast.Name):
      if not isinstance(expr, ast.Attribute):
        raise ValueError(f'{string=!r} is not a supported callable string.')
      expr = expr.value
  else:
    raise ValueError(f'{string=!r} is not a supported callable string.')
  if isinstance(module, str):
    module = importlib.import_module(module)
  return getattr(module, name), args, kwargs


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
  rngs = jax.tree_util.tree_map(
      lambda rng: jax.random.split(rng, num_splits), rngs)
  slice_rngs = lambda rngs, i: jax.tree_util.tree_map(lambda rng: rng[i], rngs)
  return tuple(slice_rngs(rngs, i) for i in range(num_splits))


def make_match_fn_from_regex_list(
    regexes: Optional[Sequence[str]]) -> Optional[Callable[[str], bool]]:
  """Creates a function returning True iff a str matches any of the regexes."""

  if not regexes:
    return None
  if isinstance(regexes, str):
    regexes = [regexes]
  joined_regex = re.compile(
      '(?:' + '|'.join([f'(?:{r})' for r in regexes]) + ')')

  def fn(string: str) -> bool:
    return joined_regex.search(string) is not None
  return fn
