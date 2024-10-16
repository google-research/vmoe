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

"""Class that summarizes the arrays in a given PyTree using different transformations."""
import dataclasses
import itertools
import re
from typing import Any, Iterator, List, Literal, Sequence, Tuple, Union

from absl import logging
import flax
import jax
import jax.numpy as jnp

Array = jax.Array
Op = Literal['norm', 'mean', 'std', 'sum', 'min', 'max']
Transform = Union[Op, Tuple[Op, Tuple[int, ...]]]
Pattern = Union[str, re.Pattern]
# This type of annotation is only supported with Python >= 3.11:
# See https://peps.python.org/pep-0646/.
# PatternAndTransforms = Tuple[Pattern, *Tuple[Transform, ...]]
PatternAndTransforms = Any
Rule = Union[Pattern, PatternAndTransforms]

_TRANSFORMATION = {
    'mean': lambda x, axis: jnp.mean(x, axis, keepdims=True),
    'std': lambda x, axis: jnp.std(x, axis, keepdims=True),
    'sum': lambda x, axis: jnp.sum(x, axis, keepdims=True),
    'min': lambda x, axis: jnp.min(x, axis, keepdims=True),
    'max': lambda x, axis: jnp.max(x, axis, keepdims=True),
    # Frobenious norm on a (sub)array.
    'norm': lambda x, axis: jnp.sqrt(jnp.square(x).sum(axis, keepdims=True)),
}


@dataclasses.dataclass(frozen=True)
class TreeSummarizer:
  """Summarizes the arrays in a given PyTree using different transformations.

  Each rule has a pattern used to match the names of the arrays in the flattened
  PyTree. When an array is matched with a rule, the (optional) sequence of
  transformations are applied to that array. Each transformation is either a
  string denoting an operation (e.g. 'mean', 'norm', etc.) and (optionally) a
  sequence of integers representing the axes in which that operation is applied.

  After all the transformations, all the values in the resulting array are
  iterated as individual summary values. To prevent generating hundreds of
  summary values by accident, we raise an exception if the number of summary
  values for a given rule and array is larger than `max_summary_values`.

  Attributes:
    rules: Sequence of rules specifying how to summarize the arrays.
    max_summary_values: Raises an exception when a rule applied to an array
      generates more than this number of values.
  """
  rules: Sequence[Rule]
  max_summary_values: int = 1

  def __post_init__(self):
    # Compile patterns into regexes.
    rules = []
    for rule in self.rules:
      if isinstance(rule, (str, re.Pattern)):
        rules.append((re.compile(rule),))
      else:
        pattern, *transforms = rule
        rules.append((re.compile(pattern),) + tuple(transforms))
    object.__setattr__(self, 'rules', rules)

  def __call__(self, tree) -> Iterator[Tuple[str, Array]]:
    num_times_rules_matched = [0] * len(self.rules)
    tree = flax.traverse_util.flatten_dict(
        flax.serialization.to_state_dict(tree), sep='/')
    for key, value in tree.items():
      yield from self._summarize(key, value, num_times_rules_matched)
    for i, rule in enumerate(self.rules):
      if num_times_rules_matched[i] == 0:
        logging.warning('Rule %r was not matched with any leaf key.', rule)

  def _summarize(
      self, key: str, value: Array, num_times_rules_matched: List[int],
  ) -> Iterator[Tuple[str, Array]]:
    """Generates one or multiple summary values for the given input array."""
    for i, (pattern, *transforms) in enumerate(self.rules):
      pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
      if pattern.search(key):
        num_times_rules_matched[i] += 1
        suffix, summary = self._transform(value, transforms)
        if summary.size > self.max_summary_values:
          raise ValueError(
              f'Rule with pattern={pattern.pattern!r} and {transforms=} '
              f'generates {summary.size} > {self.max_summary_values} '
              'summary values. If this is intended, increase the '
              'max_yielded_value_per_array.')
        yield from self._yield(key + suffix, summary)

  @classmethod
  def _transform(
      cls, value: Array, transforms: Sequence[Transform]) -> Tuple[str, Array]:
    """Applies a sequence of transforms to the input value."""
    suffix = ''
    output = value
    for transform in transforms:
      op = transform if isinstance(transform, str) else transform[0]
      if isinstance(transform, str):
        axes = None
      else:
        axes = transform[1]
        axes = (axes,) if isinstance(axes, int) else tuple(axes)
      if op not in _TRANSFORMATION:
        raise ValueError(
            f"Operation {op=} was extracted from {transform=}, but this "
            "isn't supported.")
      output = _TRANSFORMATION[op](output, axes)
      suffix += f'_{op}'
      if axes is not None:
        suffix += f'{axes}'
    return suffix, output

  @classmethod
  def _yield(cls, prefix: str, value: Array) -> Iterator[Tuple[str, Array]]:
    for idx in itertools.product(*tuple(range(s) for s in value.shape)):
      idx_s = ','.join([str(i) for i, s in zip(idx, value.shape) if s > 1])
      suffix = '[' + idx_s + ']' if idx_s else ''
      yield prefix + suffix, value[idx]
