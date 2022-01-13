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

"""Functions to initialize a TrainState with pre-trained checkpoints."""
import multiprocessing.pool
import re
from typing import Any, Optional, Sequence, Tuple

from absl import logging
import flax.core
import flax.traverse_util
from jax.experimental import maps
from jax.experimental import pjit
import numpy as np
import scipy.ndimage
from vit_jax import checkpoint as vit_jax_checkpoint
from vmoe.checkpoints import partitioned as vmoe_partitioned_checkpoint


PyTree = Any
ThreadPool = multiprocessing.pool.ThreadPool


def initialize_from_vmoe_release(
    *,
    params: PyTree,
    axis_resources: PyTree,
    prefix: str,
    mapping: Sequence[Tuple[str, str]] = (),
    keep: Sequence[str] = (),
    thread_pool: Optional[ThreadPool] = None,
    mesh: Optional[maps.Mesh] = None,
) -> PyTree:
  """Initializes parameters from a V-MoE released checkpoint.

  V-MoE released checkpoints *do not* contain the full TrainState, only the
  parameters.

  Args:
    params: PyTree of parameters to initialize. This should not be used again
      once this function returns. Use the returned object instead.
    axis_resources: PyTree of the same structure as params, indicating how the
      arrays in `params` are partitioned across the logical mesh of devices.
    prefix: Filepath of the checkpoint to use for initialization.
    mapping: A sequence of pairs (regular expression, replacement) denoting
      how to replace the names of the parameters in `params` with the names of
      the parameters loaded from the checkpoint.
    keep: A sequence of regular expressions. If a name of a parameter in
      `params` matches any of these, its current value will be kept. For
      instance, you may want to use this to train a head from scratch while
      using a pre-trained checkpoint for the rest of parameters.
    thread_pool: Optional thread pool used to restore checkpoints. This can
      significantly speed-up the time to restore a sharded checkpoint.
    mesh: Logical mesh of devices used to run the model. This is used to copy
      the restored parameters to devices. If None, the current active mesh will
      be used.

  Returns:
    A PyTree as the input `params` with (some of) the values loaded from the
    checkpoint.
  """
  # Compile strings representing regexes.
  mapping = [(re.compile(regex), repl) for regex, repl in mapping]
  keep = (re.compile('|'.join(f'(?:{regex})' for regex in keep))
          if keep else None)
  is_frozen_dict = isinstance(params, flax.core.FrozenDict)
  # Map current axis_resources to the names expected in the checkpoint.
  axis_resources_flat = flax.traverse_util.flatten_dict(
      flax.core.unfreeze(axis_resources))
  ckpt_axis_resources = {}
  for key_tuple, value in axis_resources_flat.items():
    key = '/'.join(key_tuple)
    ckpt_key_tuple = tuple(_map_name(key, mapping).split('/'))
    ckpt_axis_resources[ckpt_key_tuple] = value
  ckpt_axis_resources = flax.traverse_util.unflatten_dict(ckpt_axis_resources)
  if isinstance(axis_resources, flax.core.FrozenDict):
    ckpt_axis_resources = flax.core.freeze(ckpt_axis_resources)
  # Restore parameters from the checkpoint.
  ckpt_params = vmoe_partitioned_checkpoint.restore_checkpoint(
      prefix=prefix,
      tree=None,
      axis_resources=ckpt_axis_resources,
      mesh=mesh,
      thread_pool=thread_pool)
  # Flatten dictionaries of parameters.
  params_flat = flax.traverse_util.flatten_dict(flax.core.unfreeze(params))
  ckpt_params = flax.core.unfreeze(ckpt_params)
  ckpt_params_flat = flax.traverse_util.flatten_dict(ckpt_params)
  ckpt_params_flat = {'/'.join(k): v for k, v in ckpt_params_flat.items()}
  ckpt_params_used = set()
  # Replace values in params with those restored from the checkpoint.
  for key_tuple, value in params_flat.items():
    key = '/'.join(key_tuple)
    # If the key matches one of the regexes in `keep`, continue to the next key.
    if keep and keep.search(key):
      continue
    ckpt_key = _map_name(key, mapping)
    ckpt_params_used.add(ckpt_key)
    ckpt_value = ckpt_params_flat[ckpt_key]
    if value.shape == ckpt_value.shape:
      params_flat[key_tuple] = ckpt_value
    elif key.endswith('/posembed_input/pos_embedding'):
      params_flat[key_tuple] = _zoom_position_embedding(
          ckpt_value, value)
    else:
      raise ValueError(f'Parameter {key!r} was mapped to {ckpt_key!r}, but '
                       f'their shapes are not equal: {value.shape} vs '
                       f'{ckpt_value.shape}.')
  ckpt_params_unused = set(ckpt_params_flat.keys()) - ckpt_params_used
  if ckpt_params_unused:
    logging.info('The following parameters were found in the checkpoint but '
                 'not used during initialization:\n\t%s',
                 '\n\t'.join(sorted(ckpt_params_unused)))
  params = flax.traverse_util.unflatten_dict(params_flat)
  del ckpt_params, ckpt_params_flat, params_flat
  if is_frozen_dict:
    params = flax.core.freeze(params)
  return _pjit_donate_to_device(params, axis_resources, mesh)


def initialize_from_vit(
    *,
    params: PyTree,
    axis_resources: PyTree,
    filepath: str,
    mapping: Sequence[Tuple[str, str]] = (),
    keep: Sequence[str] = (),
    broadcast: Sequence[str] = (),
    mesh: Optional[maps.Mesh] = None,
) -> PyTree:
  """Initializes parameters from a VisionTransformer checkpoint.

  Args:
    params: PyTree of parameters to initialize. This should not be used again
      once this function returns. Use the returned object instead.
    axis_resources: PyTree of the same structure as params, indicating how the
      arrays in `params` are partitioned across the logical mesh of devices.
    filepath: Filepath of the checkpoint to use for initialization.
    mapping: A sequence of pairs (regular expression, replacement) denoting
      how to replace the names of the parameters in `params` with the names of
      the parameters loaded from the checkpoint.
    keep: A sequence of regular expressions. If a name of a parameter in
      `params` matches any of these, its current value will be kept. For
      instance, you may want to use this to train a head from scratch while
      using a pre-trained checkpoint for the rest of parameters.
    broadcast: A sequence of regular expressions. If a name of a parameter in
      `params` matches any of these, and the shape does not match with that of
      the checkpoint, we'll try to broadcast from the checkpoint shape to the
      new shape before raising an exception.
    mesh: Logical mesh of devices used to run the model. This is used to copy
      the restored parameters to devices. If None, the current active mesh will
      be used.

  Raises:
    KeyError: If a parameter name does not exist in the checkpoint, and it
      doesn't match one of the regular expressions in `keep`.
    ValueError: If a parameter shape is not compatible with the corresponding
      parameter from the checkpoint.

  Returns:
    A PyTree as the input `params` with (some of) the values loaded from the
    checkpoint.
  """
  # Compile strings representing regexes.
  mapping = [(re.compile(regex), repl) for regex, repl in mapping]
  keep = (re.compile('|'.join(f'(?:{regex})' for regex in keep))
          if keep else None)
  broadcast = (re.compile('|'.join(f'(?:{regex})' for regex in broadcast))
               if broadcast else None)
  is_frozen_dict = isinstance(params, flax.core.FrozenDict)
  # Get the parameters from the current training state.
  if is_frozen_dict:
    params = params.unfreeze()
  # Get the parameters from the checkpoint.
  ckpt_params = vit_jax_checkpoint.load(filepath)
  if isinstance(ckpt_params, flax.core.FrozenDict):
    ckpt_params = ckpt_params.unfreeze()
  # Flatten dictionaries of parameters.
  params_flat = flax.traverse_util.flatten_dict(params)
  ckpt_params_flat = flax.traverse_util.flatten_dict(ckpt_params)
  ckpt_params_flat = {'/'.join(k): v for k, v in ckpt_params_flat.items()}
  ckpt_params_used = set()
  for key_tuple, value in params_flat.items():
    key = '/'.join(key_tuple)
    # If the key matches one of the regexes in `keep`, continue to the next key.
    if keep and keep.search(key):
      continue
    ckpt_key = _map_name(key, mapping)
    if ckpt_key not in ckpt_params_flat:
      raise KeyError(f'Parameter {key!r} was mapped to {ckpt_key!r}, but it '
                     f'was not found in the checkpoint {filepath!r}')
    ckpt_params_used.add(ckpt_key)
    ckpt_value = ckpt_params_flat[ckpt_key]
    # TODO(jpuigcerver): Support parameters partitioned arbitrarily.
    # This may not work if parameters are partitioned across an axis that
    # existed in the checkpoint, since value.shape is the local shape (in the
    # current process), not the global shape of the array. However, in our
    # current use case, we only partition the axis corresponding to the experts,
    # which does not exist in the VisionTransformer checkpoints.
    if value.shape == ckpt_value.shape:
      params_flat[key_tuple] = np.asarray(ckpt_value).astype(str(value.dtype))
    elif key.endswith('/posembed_input/pos_embedding'):
      params_flat[key_tuple] = _zoom_position_embedding(
          ckpt_value, value)
    elif (broadcast and broadcast.search(key) and
          _is_broadcastable(ckpt_value.shape, value.shape)):
      params_flat[key_tuple] = np.broadcast_to(ckpt_value, value.shape)
    else:
      raise ValueError(f'Parameter {key!r} was mapped to {ckpt_key!r}, but '
                       f'it is not broadcastable or their shapes are not '
                       f'compatible: {value.shape} vs {ckpt_value.shape}.')
  ckpt_params_unused = set(ckpt_params_flat.keys()) - ckpt_params_used
  if ckpt_params_unused:
    logging.info('The following parameters were found in the checkpoint but '
                 'not used during initialization:\n\t%s',
                 '\n\t'.join(sorted(ckpt_params_unused)))
  params = flax.traverse_util.unflatten_dict(params_flat)
  del ckpt_params, ckpt_params_flat, params_flat
  if is_frozen_dict:
    params = flax.core.freeze(params)
  return _pjit_donate_to_device(params, axis_resources, mesh)


def _is_broadcastable(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> bool:
  for a, b in zip(shape1[::-1], shape2[::-1]):
    if a == 1 or b == 1 or a == b:
      pass
    else:
      return False
  return True


def _map_name(name: str, map_regexes: Sequence[Tuple[re.Pattern, str]]) -> str:
  for regex, repl in map_regexes:
    new_name, n = regex.subn(repl, name)
    if n > 0:
      return new_name
  return name


def _pjit_donate_to_device(data, axis_resources, mesh):
  mesh = mesh or maps.thread_resources.env.physical_mesh
  with maps.mesh(mesh.devices, mesh.axis_names):
    return pjit.pjit(
        fun=lambda x: x,
        in_axis_resources=(axis_resources,),
        out_axis_resources=axis_resources,
        donate_argnums=(0,))(data)


def _zoom_position_embedding(source, target):
  """Zooms (resizes with interpolation) position embeddings.

  Args:
    source: Array with the source position embeddings to use.
    target: Array with the target position embeddings to replace with `source`.

  Returns:
    An array with the same shape as target.
  """
  if source.ndim != 3 or source.shape[0] != 1:
    raise ValueError(f'The source position embedding must have shape (1, ?, ?) '
                     f'but has {source.shape}')
  if target.ndim != 3 or target.shape[0] != 1:
    raise ValueError(f'The target position embedding must have shape (1, ?, ?) '
                     f'but has {target.shape}')
  if source.shape[2] != target.shape[2]:
    raise ValueError('hidden_size of source and target does not match: '
                     f'{source.shape[2]} vs. {target.shape[2]}')

  def _get_tok_and_grid_emb(value):
    _, num_tokens, hidden_size = value.shape
    sqrt_tokens = int(np.sqrt(num_tokens))
    grid_shape = (sqrt_tokens, sqrt_tokens, hidden_size)
    if sqrt_tokens**2 == num_tokens:
      return None, value[0, :, :].reshape(grid_shape)
    elif sqrt_tokens**2 + 1 == num_tokens:
      return value[0, :1, :], value[0, 1:, :].reshape(grid_shape)
    else:
      raise ValueError(f"{num_tokens} tokens found, which is neither a perfect "
                       "square nor a perfect square + 1. This means that the "
                       "grid used is not squared. This isn't supported.")

  source_tok_emb, source_grid_emb = _get_tok_and_grid_emb(source)
  target_tok_emb, target_grid_emb = _get_tok_and_grid_emb(target)
  zoom = tuple(
      map(lambda n, o: n / o, target_grid_emb.shape, source_grid_emb.shape))
  source_grid_emb = scipy.ndimage.zoom(source_grid_emb, zoom, order=1)
  output = source_grid_emb.reshape((-1,) + source_grid_emb.shape[2:])
  if target_tok_emb is not None:
    output = np.concatenate([
        source_tok_emb if source_tok_emb is not None else target_tok_emb,
        output,
    ], axis=0)
  return np.expand_dims(output, axis=0)
