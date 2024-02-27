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

"""Rules and transformations used for mapping FLAX state dicts."""
import abc
import dataclasses
import math
import re
from typing import Any, List, Optional, Sequence, Tuple, Union

import flax.struct
import jax
from jax.experimental import shard_map
from jax.interpreters import pxla
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import scipy.ndimage
from vmoe import partitioning
from vmoe import utils

Array = jax.Array
UnparsedRules = Sequence[Union['Rule', Tuple[Any, ...]]]

shard_map = shard_map.shard_map

get_array_sharding_or_default = partitioning.get_array_sharding_or_default


@dataclasses.dataclass
class Rules:
  """Rules used for mapping serializable FLAX trees."""
  rules: List['Rule'] = dataclasses.field(default_factory=list)

  def find(self, key: str) -> Optional['Rule']:
    for rule in self.rules:
      if rule.pattern.search(key):
        return rule
    return None

  @classmethod
  def parse(cls, rules: UnparsedRules) -> 'Rules':
    r"""Parse a sequence of tuples into a Rules object.

    Each rule is generally a tuple of the form ('regex', 'repl', ...); where
    'regex' is a string representing a Python regular expression (e.g.
    r'layer_(\d+)/(kernel|bias)'), and 'repl' is a string representing a Python
    replacement (e.g. r'block_\1/\2'). For additional details on Python regular
    expressions, check https://docs.python.org/3/library/re.html.

    The replacement can also be None or the empty string, to discard the source
    arrays matching the pattern.

    In addition to mapping names and skipping arrays, we also support some
    transformations on the source arrays, by using rules with the form
    ('regex', 'repl', 'transform', *args). Some examples are:

      - (r'(mha/query/kernel)', r'\1', 'reshape', (8, 64, 32))
        reshape an array to the given shape.
      - (r'layer_\d+/kernel', r'scanned_layer/kernel', 'stack', 0)
        stack several arrays on the first axis. Layers are stacked in "natural"
        order of their source names.

    Arguments:
      rules: A sequence of tuples. Read the description above for more details.

    Returns:
      A Rules object.
    """
    parsed_rules = []
    for rule in rules:
      if isinstance(rule, Rule):
        parsed_rules.append(rule)
        continue
      pattern, replacement, *transform = rule
      pattern = re.compile(pattern)
      if not transform:
        if replacement:
          rule = RegexRule(pattern=pattern, replacement=replacement)
        else:
          rule = DropRule(pattern=pattern)
      elif transform[0] == 'reshape':
        assert len(transform) in (1, 2)
        # Note: shape = None will be replaced later by the target's array shape.
        shape = transform[1] if len(transform) == 2 else None
        rule = ReshapeRule(
            pattern=pattern, replacement=replacement, shape=shape)
      elif transform[0] == 'squeeze':
        assert len(transform) == 2
        assert isinstance(transform[1], int)
        rule = SqueezeRule(
            pattern=pattern, replacement=replacement, axis=transform[1])
      elif transform[0] == 'stack':
        assert len(transform) == 2
        assert isinstance(transform[1], int)
        rule = StackRule(
            pattern=pattern, replacement=replacement, axis=transform[1])
      elif transform[0] == 'expand_tile':
        assert len(transform) == 3
        rule = ExpandTileRule(
            pattern=pattern, replacement=replacement, axis=transform[1],
            reps=transform[2])
      elif transform[0] == 'vit_zoom':
        rule = VitZoomRule(pattern=pattern, replacement=replacement)
      elif transform[0] == 'zoom':
        rule = ZoomRule(pattern=pattern, replacement=replacement)
      else:
        raise ValueError(f'Rule {rule!r} cannot be parsed.')
      parsed_rules.append(rule)
    return Rules(rules=parsed_rules)


@dataclasses.dataclass
class Rule:
  pattern: re.Pattern[str]

  def __post_init__(self):
    if isinstance(self.pattern, str):
      self.pattern = re.compile(self.pattern)

  def rename(self, key: str) -> str:
    return key


@dataclasses.dataclass
class DropRule(Rule):
  """The matched arrays are dropped, i.e. not used in the target tree."""


@dataclasses.dataclass
class RegexRule(Rule):
  """Renames the matched arrays."""
  replacement: str

  def rename(self, key: str) -> str:
    return self.pattern.sub(self.replacement, key)

  def get_transformation(
      self, source: Array,
      target: Union[Array, jax.ShapeDtypeStruct]) -> 'Transformation':
    return CopyTransformation(target_shape_dtype=self.get_shape_dtype(target),
                              array=source)

  def get_shape_dtype(self, array: Array) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape=array.shape, dtype=array.dtype,
                                sharding=get_array_sharding_or_default(array))


@dataclasses.dataclass
class ExpandTileRule(RegexRule):
  """Expands the dimensions of the source array and tiles along them."""
  axis: Union[int, Tuple[int, ...]]
  reps: Union[int, Tuple[int, ...]]

  def get_transformation(self, source, target) -> 'ExpandTileTransformation':
    return ExpandTileTransformation(
        target_shape_dtype=self.get_shape_dtype(target), array=source,
        axis=self.axis, reps=self.reps)


@dataclasses.dataclass
class ReshapeRule(RegexRule):
  """Reshapes the source array into the given shape.

  Attributes:
    shape: Target shape. If shape is None, the shape of the matched target array
      is used.
  """
  shape: Optional[Tuple[int, ...]]

  def get_transformation(self, source, target) -> 'ReshapeTransformation':
    return ReshapeTransformation(
        target_shape_dtype=self.get_shape_dtype(target), array=source,
        shape=self.shape or target.shape)


@dataclasses.dataclass
class SqueezeRule(RegexRule):
  axis: int

  def get_transformation(self, source, target) -> 'SqueezeTransformation':
    return SqueezeTransformation(
        target_shape_dtype=self.get_shape_dtype(target), array=source,
        axis=self.axis)


@dataclasses.dataclass
class StackRule(RegexRule):
  """Stacks all the source arrays matching the regex on the given axis."""
  axis: int

  def get_transformation(self, source, target) -> 'StackTransformation':
    return StackTransformation(
        target_shape_dtype=self.get_shape_dtype(target), axis=self.axis,
        to_stack=[source])


@dataclasses.dataclass
class VitZoomRule(RegexRule):

  def get_transformation(self, source, target) -> 'VitZoomTransformation':
    return VitZoomTransformation(
        target_shape_dtype=self.get_shape_dtype(target), source=source,
        target=None if isinstance(target, jax.ShapeDtypeStruct) else target)


@dataclasses.dataclass
class ZoomRule(RegexRule):

  def get_transformation(self, source, target) -> 'ZoomTransformation':
    return ZoomTransformation(
        target_shape_dtype=self.get_shape_dtype(target), source=source,
        target=None if isinstance(target, jax.ShapeDtypeStruct) else target)


class Transformation(abc.ABC, flax.struct.PyTreeNode):
  target_shape_dtype: jax.ShapeDtypeStruct = flax.struct.field(
      pytree_node=False)

  @abc.abstractmethod
  def __call__(self) -> Array:
    """Returns the array resulting from the transformation."""


class CopyTransformation(Transformation):
  array: Array = flax.struct.field(pytree_node=True)

  def __call__(self) -> Array:
    return self.array


class ExpandTileTransformation(Transformation):
  """Expand the dimensions of an array and tile along those."""
  array: Array = flax.struct.field(pytree_node=True)
  axis: Union[int, Tuple[int, ...]] = flax.struct.field(pytree_node=False)
  reps: Union[int, Tuple[int, ...]] = flax.struct.field(pytree_node=False)

  def __call__(self) -> Array:
    array = jnp.expand_dims(self.array, axis=self.axis)
    return jnp.tile(array, self._make_reps(array.ndim))

  def _make_reps(self, ndim: int) -> Sequence[int]:
    axis = (self.axis,) if isinstance(self.axis, int) else self.axis
    reps = (self.reps,) if isinstance(self.reps, int) else self.reps
    output = [1,] * ndim
    for a, r in utils.safe_zip(axis, reps):
      output[a] = r
    return output


class ReshapeTransformation(Transformation):
  array: Array = flax.struct.field(pytree_node=True)
  shape: Tuple[int, ...] = flax.struct.field(pytree_node=False)

  def __call__(self) -> Array:
    return jnp.reshape(self.array, newshape=self.shape)


class SqueezeTransformation(Transformation):
  array: Array = flax.struct.field(pytree_node=True)
  axis: int = flax.struct.field(pytree_node=False)

  def __call__(self) -> Array:
    return jnp.squeeze(self.array, axis=self.axis)


class StackTransformation(Transformation):
  """Util class used to store arrays that need to be stacked."""
  axis: int = flax.struct.field(pytree_node=False)
  to_stack: List[Array] = flax.struct.field(pytree_node=True,
                                            default_factory=list)

  def append(self, x: Array) -> 'StackTransformation':
    return self.replace(to_stack=self.to_stack + [x])

  def __call__(self) -> Array:
    return jnp.stack(self.to_stack, axis=self.axis)


class ZoomTransformation(Transformation):
  """Resizes with linear interpolation the source array."""
  source: Array = flax.struct.field(pytree_node=True)
  target: Optional[Array] = flax.struct.field(pytree_node=True, default=None)

  @classmethod
  def _zoom(cls, source, callback_shape_dtype, zoom):
    # Wrap scipy.ndimage.zoom with a _pure_callback call.
    def _pure_callback_zoom(x):
      return jax.pure_callback(
          lambda xx: scipy.ndimage.zoom(xx, zoom, order=1),
          callback_shape_dtype,
          x,
          vectorized=False,
      )

    mesh = pxla.thread_resources.env.physical_mesh
    if mesh.empty:
      return _pure_callback_zoom(source)
    else:
      source = partitioning.with_sharding_constraint(source, P())
      return shard_map(
          _pure_callback_zoom,
          mesh,
          in_specs=P(*(None,) * source.ndim),
          out_specs=P(*(None,) * source.ndim),
          check_rep=False,
      )(source)

  def __call__(self) -> Array:
    source = self.source
    target = self.target_shape_dtype if self.target is None else self.target
    zoom = tuple(map(lambda n, o: n / o, target.shape, source.shape))
    callback_shape_dtypes = jax.ShapeDtypeStruct(
        shape=target.shape, dtype=source.dtype)
    return self._zoom(source, callback_shape_dtypes, zoom)


class VitZoomTransformation(ZoomTransformation):
  """Resizes with linear interpolation the source array, as done in ViT for positional embeddings."""

  @classmethod
  def _get_tok_and_grid_shape(cls, value):
    _, num_tokens, hidden_size = value.shape
    sqrt_tokens = int(math.sqrt(num_tokens))
    grid_shape = (1, sqrt_tokens, sqrt_tokens, hidden_size)
    if num_tokens not in (sqrt_tokens**2, sqrt_tokens**2 + 1):
      raise ValueError(f'{num_tokens=} is neither a perfect square nor a '
                       'perfect square + 1. This means that the grid used is '
                       "not squared. This isn't supported.")
    if sqrt_tokens**2 == num_tokens:
      return False, grid_shape
    else:
      return True, grid_shape

  def __call__(self) -> Array:
    source = self.source
    target = self.target_shape_dtype if self.target is None else self.target
    if source.ndim != 3 or source.shape[0] != 1:
      raise ValueError(f'{source.shape=} is not (1, ?, ?)')
    if target.ndim != 3 or target.shape[0] != 1:
      raise ValueError(f'{target.shape=} is not (1, ?, ?)')
    if source.shape[2] != target.shape[2]:
      raise ValueError(f'{source.shape[2]=} != {target.shape[2]}')
    # Determine whether the source model uses a token or not and its grid size.
    source_has_tok, source_grid_shape = self._get_tok_and_grid_shape(source)
    # Extract the source grid embedding.
    if source_has_tok:
      source_grid_emb = source[:, 1:, :].reshape(source_grid_shape)
    else:
      source_grid_emb = source[:, :, :].reshape(source_grid_shape)
    # Determine whether the target model uses a token or not and its grid size.
    target_has_tok, target_grid_shape = self._get_tok_and_grid_shape(target)
    zoom = tuple(map(lambda n, o: n / o, target_grid_shape, source_grid_shape))
    callback_shape_dtypes = jax.ShapeDtypeStruct(
        shape=target_grid_shape, dtype=source.dtype)
    output_grid_emb = self._zoom(source_grid_emb, callback_shape_dtypes, zoom)
    hidden_size = output_grid_emb.shape[-1]
    output_grid_emb = output_grid_emb.reshape((1, -1, hidden_size))
    if target_has_tok:
      # Concatenate the token embedding at the start of the grid embedding.
      if source_has_tok:
        output_tok_emb = source[:, :1, :]
      elif isinstance(target, jax.ShapeDtypeStruct):
        output_tok_emb = 0.02 * jax.random.normal(jax.random.PRNGKey(0),
                                                  (1, 1, hidden_size),
                                                  dtype=target.dtype)
      else:
        output_tok_emb = target[:, :1, :]
      return jnp.concatenate([output_tok_emb, output_grid_emb], axis=1)
    else:
      return output_grid_emb
