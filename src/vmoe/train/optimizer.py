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

"""Optimizers."""
import re
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import flax.serialization
import flax.traverse_util
import jax.numpy as jnp
import optax
from vmoe import utils
from vmoe.train import schedule

DType = type(jnp.float32)
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
    flatupdates = flax.traverse_util.flatten_dict(
        flax.serialization.to_state_dict(updates), sep='/')
    flatparams = flax.traverse_util.flatten_dict(
        flax.serialization.to_state_dict(params), sep='/')
    flatupdates = dict(utils.safe_map(
        lambda k, g, p: (k, g + weight_decay_fn(k) * p),
        flatupdates.keys(),
        flatupdates.values(),
        flatparams.values()))
    updates = flax.serialization.from_state_dict(
        updates, flax.traverse_util.unflatten_dict(flatupdates, sep='/'))
    return updates, state

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
    gradient_scale: Optional[Sequence[Tuple[str, float]]] = None,
    **optim_kwargs) -> optax.GradientTransformation:
  """Creates an optax optimizer."""
  ops = []
  # Optionally, apply a scale factor to some gradients.
  # WARNING: Use this with caution. Notice that this is NOT equivalent to having
  # a specific learning rate per parameter, since the scale that you use here
  # will affect the state of the optimizers like momentum.
  ops.append(gradient_scaling(gradient_scale))
  # Optionally, add gradient clipping.
  ops.append(gradient_clipping(**(gradient_clip or {})))
  # Optimizer-dependant scaling of gradients.
  # Note: we don't use optax aliases (e.g. optax.adam, optax.sgd, ...) because
  # we want to control explicitly how to add weight decay.
  if name == 'adam':
    ops.append(optax.scale_by_adam(**optim_kwargs))
  elif name == 'big_vision_adafactor':
    ops.append(scale_by_big_vision_adafactor(**optim_kwargs))
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
  # Wrap scale with inject_hyperparams to keep the last learning rate in the
  # optimizer state.
  @optax.inject_hyperparams
  def _scale_by_learning_rate(learning_rate):
    return optax.scale(-learning_rate)
  ops.append(_scale_by_learning_rate(lr_schedule))
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
    flatparams = flax.traverse_util.flatten_dict(
        flax.serialization.to_state_dict(params), sep='/')
    output = {
        key: search_true if pattern.search(key) else search_false
        for key, value in flatparams.items()
    }
    return flax.serialization.from_state_dict(
        params, flax.traverse_util.unflatten_dict(output, sep='/'))

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


def gradient_scaling(
    scales: Sequence[Tuple[str, float]]) -> optax.GradientTransformation:
  """Optionally scales gradients by a given factor.

  Example:
    gradient_scaling([
      ('Encoder/Moe/Mlp', 8.),  # Scale MLP gradients in MoE by a factor of 8.
    ])

  Args:
    scales: A sequence of pairs (regex, value). Each gradient is scaled by the
      value paired with the first regex that matches (or 1.0 if none matches).

  Returns:
    An optax.GradientTransformation object.
  """
  if not scales:
    return optax.identity()

  scales = [(re.compile(k), v) for k, v in scales]

  def scale_fn(key):
    for regex, value in scales:
      if regex.search(key):
        return value
    return 1.

  def init_fn(_):
    return optax.EmptyState()

  def update_fn(updates, state, params=None):
    del params
    flatupdates = flax.traverse_util.flatten_dict(
        flax.serialization.to_state_dict(updates), sep='/')
    flatupdates = dict(
        utils.safe_map(lambda k, g: (k, g * scale_fn(k)),
                       flatupdates.keys(), flatupdates.values()))
    updates = flax.serialization.from_state_dict(
        updates, flax.traverse_util.unflatten_dict(flatupdates, sep='/'))
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_big_vision_adafactor(
    decay_rate: float = 0.8,
    decay_rate_max: float = 0.999,
    decay_offset: int = 0,
    eps: float = 1e-30,
    min_dim_size_to_factor: int = 32,
    momentum: float = 0.9,
    momentum_dtype: DType = jnp.float32,
) -> optax.GradientTransformation:
  """Custom Adafactor used to train V-MoE-15B in https://arxiv.org/abs/2106.05974.

  Forked from: https://github.com/google-research/big_vision.

  Important: In models with MoE layers, one must set min_dim_size_to_factor to
  at least num_experts + 1, to ensure the same behaviour as in the paper (i.e.
  the params are not factorized across the experts axis).
  TODO(jpuigcerver): Generalize this to allow custom factorization rules.

  Args:
    decay_rate: controls exponential decay schedule of squared grads.
    decay_rate_max: maximum decay rate for the exponentially weighted average
      of squared grads.
    decay_offset: for finetuning, one may set this to the starting step-number
        of the fine tuning phase.
    eps: regularization constant for squared gradient.
    min_dim_size_to_factor: only factor accumulator if two array dimensions
      are at least this size.
    momentum: decay rate for the exponentially weighted average of grads.
    momentum_dtype: dtype used to store the exponentially weighted average of
      grads.

  Returns:
    A GradientTransform.
  """
  def _decay_rate_pow(step, exponent):
    """Second-order moment decay schedule."""
    t = jnp.array(step, jnp.float32) + 1.0
    return jnp.minimum(decay_rate_max, 1.0 - t**(-exponent))

  scale_by_rms = optax.scale_by_factored_rms(
      factored=True,
      decay_rate=decay_rate,
      step_offset=decay_offset,
      min_dim_size_to_factor=min_dim_size_to_factor,
      epsilon=eps,
      decay_rate_fn=_decay_rate_pow)
  mom = (optax.ema(momentum, debias=False, accumulator_dtype=momentum_dtype)
         if momentum else optax.identity())
  return optax.chain(scale_by_rms, mom)


def trace_momentum(momentum: Optional[float] = None, **kwargs):
  return optax.trace(decay=momentum, **kwargs) if momentum else optax.identity()
