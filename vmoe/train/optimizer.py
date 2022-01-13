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

"""Optimizers."""
import re
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import flax.core
import flax.traverse_util
import optax
import vmoe.train.schedule as schedule
import vmoe.utils as utils

PyTree = Any
MaskFn = Callable[[optax.Params], PyTree]
WeightDecay = Union[float, Sequence[Tuple[str, float]]]


def add_decayed_weights(
    weight_decay: Optional[WeightDecay]) -> optax.GradientTransformation:
  """Optionally adds parameters scaled by weight_decay.

  This is similar to `optax.add_decayed_weights`, but supports passing different
  weight decay factors for each parameter, by matching the parameter names with
  a regex. The factor of the first regex matched will be used.

  Examples:
    add_decayed_weights(1e-3)   # All parameters are decayed with factor = 1e-3.

    add_decayed_weights([
      ('Head/kernel', 0.1),         # Head kernel uses factor = 0.1.
      ('Encoder/*/kernel', 0.001),  # Kernels in the encoder factor = 0.001.
      # Parameters that do not match the above regexes are not decayed.
    ])

  Args:
    weight_decay: A float or a mapping from regexes to floats.

  Returns:
    An optax.GradientTransformation object.
  """
  if not weight_decay:
    return optax.identity()
  elif isinstance(weight_decay, (list, tuple)):
    weight_decay = [(re.compile(k), v) for k, v in weight_decay]
    def weight_decay_fn(key):
      for regex, value in weight_decay:
        if regex.search(key):
          return value
      return 0.
  else:
    def weight_decay_fn(unused_key):
      return weight_decay

  def init_fn(_):
    return optax.AddDecayedWeightsState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError('Not passing `params` when calling `update`.')
    frozen = isinstance(updates, flax.core.FrozenDict)
    updates = flax.traverse_util.flatten_dict(flax.core.unfreeze(updates))
    params = flax.traverse_util.flatten_dict(flax.core.unfreeze(params))
    updates = dict(utils.safe_map(
        lambda k, g, p: (k, g + weight_decay_fn('/'.join(k)) * p),
        updates.keys(),
        updates.values(),
        params.values()))
    updates = flax.traverse_util.unflatten_dict(updates)
    return flax.core.freeze(updates) if frozen else updates, state

  return optax.GradientTransformation(init_fn, update_fn)


def create_optimizer(
    *,
    name: str,
    total_steps: int,
    learning_rate: Union[float, Mapping[str, Any]],
    gradient_clip: Optional[Mapping[str, Any]] = None,
    weight_decay: Optional[WeightDecay] = None,
    frozen_pattern: Optional[Union[str, Sequence[str]]] = None,
    trainable_pattern: Optional[Union[str, Sequence[str]]] = None,
    **optim_kwargs) -> optax.GradientTransformation:
  """Creates an optax optimizer."""
  # Optionally, add gradient clipping.
  ops = [gradient_clipping(**(gradient_clip or {}))]
  # Optimizer-dependant scaling of gradients.
  # Note: we don't use optax aliases (e.g. optax.adam, optax.sgd, ...) because
  # we want to control explicitly how to add weight decay.
  if name == 'adam':
    ops.append(optax.scale_by_adam(**optim_kwargs))
  elif name == 'sgd':
    # Optionally, add momentum with SGD.
    ops.append(trace_momentum(**optim_kwargs))
  else:
    raise ValueError(f'Unknown optimizer: {name}')
  # Optionally, add weight decay to the gradients.
  ops.append(add_decayed_weights(weight_decay))
  # Scale gradients by learning rate.
  if isinstance(learning_rate, (float, int)):
    learning_rate = {'schedule': 'constant', 'value': learning_rate}
  lr_schedule = schedule.create_learning_rate_schedule(
      **learning_rate, total_steps=total_steps)
  ops.append(optax.scale_by_schedule(lambda count: -lr_schedule(count)))
  # Optionally, freeze some variables.
  ops.append(freeze_weights(
      frozen_pattern=frozen_pattern, trainable_pattern=trainable_pattern))
  # Chain all operations on the gradients.
  return optax.chain(*ops)


def freeze_weights(
    *,
    frozen_pattern: Optional[Union[str, Sequence[str]]],
    trainable_pattern: Optional[Union[str, Sequence[str]]],
) -> optax.GradientTransformation:
  """Optionally sets to zero gradients that match `frozen_weights`."""
  if not trainable_pattern and not frozen_pattern:
    return optax.identity()
  if trainable_pattern and not frozen_pattern:
    search_true, search_false = False, True
  elif frozen_pattern and not trainable_pattern:
    search_true, search_false = True, False
  else:
    raise ValueError(
        'You cannot specify both trainable_pattern and frozen_pattern. '
        f'trainable_pattern = {trainable_pattern!r}, '
        f'frozen_pattern = {frozen_pattern!r}')
  # Create a single regex from trainable_pattern/frozen_pattern.
  pattern = trainable_pattern or frozen_pattern
  if not isinstance(pattern, str):
    pattern = '|'.join(f'(?:{x})' for x in pattern)
  pattern = re.compile(pattern)

  def frozen_fn(params: optax.Params) -> PyTree:
    frozen = isinstance(params, flax.core.FrozenDict)
    flatparams = flax.traverse_util.flatten_dict(flax.core.unfreeze(params))
    output = {
        key: search_true if pattern.search('/'.join(key)) else search_false
        for key, value in flatparams.items()
    }
    output = flax.traverse_util.unflatten_dict(output)
    return flax.core.freeze(output) if frozen else output

  return optax.masked(optax.set_to_zero(), frozen_fn)


def gradient_clipping(
    global_norm: Optional[float] = None,
    absolute_value: Optional[float] = None) -> optax.GradientTransformation:
  """Optionally performs gradient clipping."""
  if global_norm and absolute_value:
    raise ValueError(
        'You must specify either `global_norm` or `absolute_value`, '
        f'but not both: global_norm = {global_norm!r}, '
        f'absolute_value = {absolute_value!r}')
  if global_norm:
    return optax.clip_by_global_norm(global_norm)
  if absolute_value:
    return optax.clip(absolute_value)
  return optax.identity()


def trace_momentum(momentum: Optional[float] = None, **kwargs):
  return optax.trace(decay=momentum, **kwargs) if momentum else optax.identity()
