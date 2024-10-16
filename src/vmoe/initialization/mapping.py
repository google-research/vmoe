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

r"""Util for mapping serializable FLAX trees.

This util is useful for model initialization from checkpoints, or for converting
checkpoints from one format to another.

The map_state_dict function takes a source and a target FLAX trees, and a set of
rules used to map the two (flattened state dicts) trees. The rules specify a
mapping from source to target names. The value of a target leaf will be replaced
with the corresponding source leaf.

The rules can also include transformations to perform on the source arrays
(e.g. reshape, squeeze, stack, etc). All the transformations are done on-device
to allow mapping trees containing very large arrays potentially sharded across
many hosts and devices.

Here's an example illustrating how to use this:

```
target = map_state_dict(
    old_vit_optimizer_state,
    new_vmoe_train_state,
    [
        # Squeze extra dimensions in the old checkpoint.
        ('target/vit/(encoder/pos_embedding)', r'params/vmoe/\1', 'squeeze'),
        # Expand kernel and biases of MLPs with a new dimension and tile along
        # that dimension to initialize the experts.
        ('target/vit/(encoder/encoderblock_\d+)/(mlp/.*/kernel|bias)',
         r'params/vmoe/\1/moe/\2',
         'expand_tile', 0, 32),
        # The rest of parameters are simply copied without any transformation.
        ('(target/vit/(.*)', 'params/vmoe/\1'),
    ],
    # The source tree contains some variables that we don't want to use, show
    # only a warning for these and do not raise an exception.
    raise_if_source_unmatched=False,
    # The target tree contains some variables that we want to keep and are not,
    # matched against any source variable. Show only a warning for these and do
    # not raise an exception.
    raise_if_target_unmatched=False)
```
"""
import re
from typing import Dict, Iterable, List, Union

from absl import logging
import flax.core
import flax.struct
import flax.traverse_util
import jax
from jax.experimental import pjit
import jax.numpy as jnp
from vmoe import partitioning
from vmoe.checkpoints import serialization
from vmoe.initialization import rules as _rules

Array = jax.Array
EmptyNode = flax.traverse_util._EmptyNode  # pylint: disable=protected-access
Rules = _rules.Rules
Transformation = _rules.Transformation
UnparsedRules = _rules.UnparsedRules

_IDENTITY_RULE = _rules.RegexRule(pattern=re.compile('^(.*)$'),
                                  replacement=r'\1')
_SIGNED_FLOAT_RE = re.compile(
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')

__all__ = ['map_state_dict', 'Rules', 'UnparsedRules', 'Transformation']

get_array_sharding_or_default = partitioning.get_array_sharding_or_default


def map_state_dict(source,
                   target,
                   rules: Union[Rules, UnparsedRules],
                   raise_if_dtype_mismatched: bool = False,
                   raise_if_shape_mismatched: bool = True,
                   raise_if_source_unmatched: bool = True,
                   raise_if_target_unmatched: bool = True):
  """Maps a source tree into a target tree, given a set of rules."""
  source_flat = flax.traverse_util.flatten_dict(
      serialization.to_state_dict(source), sep='/', keep_empty_nodes=True)
  target_flat = flax.traverse_util.flatten_dict(
      serialization.to_state_dict(target), sep='/', keep_empty_nodes=True)
  txs = _flat_state_dict_mapping_build(source_flat, target_flat, rules,
                                       raise_if_source_unmatched,
                                       raise_if_target_unmatched)
  output = _flat_state_dict_mapping_apply(
      txs, target_flat, raise_if_dtype_mismatched, raise_if_shape_mismatched)
  output = flax.traverse_util.unflatten_dict(output, sep='/')
  return serialization.from_state_dict(target, output)


def _flat_state_dict_mapping_build(
    source_flat: Dict[str, Array],
    target_flat: Dict[str, Array],
    rules: Union[Rules, UnparsedRules],
    raise_if_source_unmatched: bool = True,
    raise_if_target_unmatched: bool = True) -> Dict[str, Transformation]:
  """Builds a dictionary of transformations (to run on device) to set the target values from the source ones."""
  if not isinstance(rules, Rules):
    rules = Rules.parse(rules)

  target_flat_used = {k: False for k in target_flat}

  output = {}
  for src_key in _natural_sort(source_flat.keys()):
    rule = rules.find(src_key)
    if rule is None:
      # No rule matches src_key.
      exception_msg = f'Source key {src_key!r} does not match any rule.'
      _raise_or_warn(KeyError,
                     exception_msg=exception_msg,
                     warn_msg=exception_msg + ' Assuming the identity rule.',
                     raise_exception=raise_if_source_unmatched)
      rule = _IDENTITY_RULE
    if isinstance(rule, _rules.DropRule):
      # The src_key should not be mapped to the new state dict.
      continue
    dst_key = rule.rename(src_key)
    if dst_key not in target_flat:
      raise KeyError(
          f'Source key {src_key!r} was matched with target key {dst_key!r}, '
          'but it does not exist.')
    if target_flat_used[dst_key] and not isinstance(rule, _rules.StackRule):
      raise ValueError(f'Target key {dst_key!r} was matched more than once.')
    target_flat_used[dst_key] = True
    src_value = source_flat[src_key]
    target_value = target_flat[dst_key]
    if isinstance(src_value, EmptyNode) != isinstance(target_value, EmptyNode):
      raise ValueError(
          f'One of the values represents an empty node, but not the other. '
          f'{src_key=} has type={type(src_value)!r} and {dst_key=} has '
          f'type={type(target_value)!r}.')
    if isinstance(src_value, EmptyNode):
      continue
    if isinstance(rule, _rules.StackRule):
      if dst_key in output:
        output[dst_key] = output[dst_key].append(src_value)
      else:
        output[dst_key] = rule.get_transformation(src_value, target_value)
    elif isinstance(rule, _rules.RegexRule):
      output[dst_key] = rule.get_transformation(src_value, target_value)
    else:
      raise TypeError(f'Unexpected type for rule: {rule!r}')

  for dst_key, used in target_flat_used.items():
    if not used and not isinstance(target_flat[dst_key], EmptyNode):
      exception_msg = f'Target key {dst_key!r} is unmatched.'
      warn_msg = exception_msg + (
          f' The original value with shape {target_flat[dst_key].shape} '
          f'and type {target_flat[dst_key].dtype} will be preserved.')
      _raise_or_warn(ValueError, exception_msg=exception_msg, warn_msg=warn_msg,
                     raise_exception=raise_if_target_unmatched)

  return output


def _flat_state_dict_mapping_apply(
    txs_flat: Dict[str, Transformation],
    target_flat: Dict[str, Array],
    raise_if_dtype_mismatched: bool = False,
    raise_if_shape_mismatched: bool = True):
  """Applies the transformations and updates the values in target_flat."""
  # Keep the target values that will not be updated.
  target_flat = {k: v for k, v in target_flat.items() if k not in txs_flat}

  def fn(txs_flat):
    for key, tx in txs_flat.items():
      value = tx()
      # Check dtype and raise an exception or try to convert.
      expected_dtype = tx.target_shape_dtype.dtype
      if value.dtype != expected_dtype:
        except_msg = (f'Target key {key!r} has mismatched dtypes. '
                      f'Actual = {value.dtype}, '
                      f'expected = {expected_dtype}.')
        warn_msg = except_msg + ' Converting to the expected dtype.'
        _raise_or_warn(ValueError, exception_msg=except_msg, warn_msg=warn_msg,
                       raise_exception=raise_if_dtype_mismatched)
        value = value.astype(expected_dtype)
      # Check shape and raise an exception or try to broadcast.
      expected_shape = tx.target_shape_dtype.shape
      if value.shape != expected_shape:
        except_msg = (f'Target key {key!r} has mismatched shapes. '
                      f'Actual = {value.shape!r}, '
                      f'expected = {expected_shape!r}.')
        warn_msg = except_msg + ' Broadcasting to the expected shape.'
        _raise_or_warn(ValueError, exception_msg=except_msg, warn_msg=warn_msg,
                       raise_exception=raise_if_shape_mismatched)
        value = jnp.broadcast_to(value, expected_shape)
      txs_flat[key] = value
    return txs_flat

  out_axis_resources = {
      k: v.target_shape_dtype.sharding for k, v in txs_flat.items()}
  output = pjit.pjit(
      fn,
      # No in_axis_resources. We assume that jax.Array is used, which includes
      # the specification of the partitioning.
      out_shardings=out_axis_resources,
      donate_argnums=(0,),
  )(txs_flat)
  target_flat.update(output)

  return target_flat


def _natural_sort(names: Iterable[str]) -> List[str]:
  """Natural sort for names with numerical substrings."""
  def maybe_num(s):
    if _SIGNED_FLOAT_RE.match(s):
      return float(s)
    else:
      return s
  def split_keys(s):
    return [maybe_num(c) for c in _SIGNED_FLOAT_RE.split(s)]
  return sorted(names, key=split_keys)


def _raise_or_warn(cls, exception_msg, raise_exception, warn_msg=None):
  if raise_exception:
    raise cls(exception_msg)
  else:
    logging.warning(warn_msg or exception_msg)
