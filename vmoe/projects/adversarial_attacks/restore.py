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

"""Utils to restore a model from a config file to allow for adversarial attacks."""
import functools
from typing import Dict, Tuple

from absl import logging
import flax.core
import flax.traverse_util
import jax
import jax.numpy as jnp
import ml_collections
import optax
from vmoe import checkpoints
from vmoe import moe
from vmoe import partitioning
from vmoe import utils
from vmoe.nn import routing
import vmoe.nn.models


SUPPORTED_ROUTERS = (
    routing.NoisyTopExpertsPerItemRouter, routing.NoisyTopItemsPerExpertRouter)


def compute_loss_predict_cw_fn(x, y, rngs, *, apply_fn, loss_fn):
  """Computes the per-example loss, prediction, correctness and combine weights (if any)."""
  batch_size = x.shape[0]
  (logits, metrics), intermediates = apply_fn(
      x, rngs=rngs, capture_intermediates=router_filter)
  loss = loss_fn(logits, y, metrics)
  # Compute the top-1 prediction and whether or not it's correct.
  pred = jnp.argmax(logits, axis=-1)
  correct = jnp.sum(jax.nn.one_hot(pred, logits.shape[-1]) * y, axis=-1)
  # This is a dict mapping from each MoE layer to a binary array of shape
  # (batch_size, num_tokens, num_experts).
  combine_weights = get_combine_weights(intermediates)
  combine_weights = jax.tree_util.tree_map(
      lambda m: m.reshape(batch_size, -1, m.shape[1]), combine_weights)
  return loss, pred, correct, combine_weights


def create_mesh(num_expert_partitions: int) -> partitioning.Mesh:
  if num_expert_partitions > jax.device_count():
    logging.error(
        'The value of num_expert_partitions is %d, but the current number of '
        'devices is %d. Setting num_expert_partitions to %d',
        num_expert_partitions, jax.device_count(),
        jax.device_count())
    num_expert_partitions = jax.device_count()
  return partitioning.get_auto_logical_mesh(num_expert_partitions,
                                            jax.devices())


def get_combine_weights(intermediates) -> Dict[str, jnp.ndarray]:
  """Returns the experts selected for each token in each layer, as a one-hot matrix."""
  if 'intermediates' not in intermediates:
    return {}
  intermediates = flax.core.unfreeze(intermediates['intermediates'])
  intermediates = flax.traverse_util.flatten_dict(intermediates, sep='/')
  output = {}
  for key, value in intermediates.items():
    dispatcher = value[0][0]
    if isinstance(dispatcher, moe.Bfloat16Dispatcher):
      dispatcher = dispatcher.dispatcher
    if isinstance(dispatcher, moe.EinsumDispatcher):
      combine_weights = dispatcher.combine_weights
      # (G, S, E, C) -> (G, S, E).
      combine_weights = jnp.sum(combine_weights, axis=3)
    elif isinstance(dispatcher, moe.ExpertIndicesDispatcher):
      # (G, S, K) -> (G, S, K, E).
      indices_one_hot = jax.nn.one_hot(
          dispatcher.indices[..., 0], dispatcher.num_experts, dtype=jnp.float32)
      combine_weights = (
          indices_one_hot * dispatcher.combine_weights[:, :, :, None])
      # (G, S, K, E) -> (G, S, E).
      combine_weights = jnp.sum(combine_weights, axis=2)
    else:
      raise TypeError(f'Unknown dispatcher type: {type(dispatcher)}')
    # (G, S, E) -> (num_tokens, E).
    output[key] = combine_weights.reshape(-1, combine_weights.shape[-1])
  return output


def get_loss_fn(name: str, **kwargs):
  """Returns the loss function."""
  def default_sigmoid_xent(logits, labels, **kw):
    return jnp.sum(
        optax.sigmoid_binary_cross_entropy(logits, labels, **kw), axis=-1)
  loss_fns = {
      'softmax_xent': optax.softmax_cross_entropy,
      'sigmoid_xent': default_sigmoid_xent,
  }
  if name in loss_fns:
    return functools.partial(loss_fns[name], **kwargs)
  else:
    raise ValueError(f'Unknown loss: {name=!r}')


def get_loss_including_auxiliary_fn(
    name: str, *, attack_auxiliary_loss: bool = False, **kwargs):
  """Returns the loss function, including the auxiliary term."""
  loss_fn = get_loss_fn(name, **kwargs)

  def fn(logits, labels, metrics):
    loss = loss_fn(logits, labels)
    if attack_auxiliary_loss:
      loss = loss + jnp.mean(metrics['auxiliary_loss'])
    return loss

  return fn


def restore_from_config(
    config: ml_collections.ConfigDict, checkpoint_prefix: str,
    image_shape: Tuple[int, int, int, int], mesh: partitioning.Mesh):
  """Restores a model from a ConfigDict."""
  # Create FLAX module from the given configuration.
  model_config = config.model
  model_class, args, kwargs = utils.parse_call(model_config.name,
                                               vmoe.nn.models)
  kwargs = ml_collections.ConfigDict(kwargs)
  kwargs.update(model_config)
  flax_module = model_class(*args, **kwargs.to_dict(), deterministic=True)
  # Obtain variables shapes. This does not initialize the model or does any
  # computations.
  extra_rng_keys = tuple(config.get('extra_rng_keys', ()))
  def _init():
    rngs = utils.make_rngs(('params',) + tuple(extra_rng_keys), 0)
    images = jnp.zeros(image_shape, dtype=jnp.float32)
    variables = flax_module.init(rngs, images)
    _, intermediates = flax_module.apply(
        variables, images, capture_intermediates=router_filter)
    if 'intermediates' in intermediates:
      intermediates = flax.core.unfreeze(intermediates['intermediates'])
      intermediates = flax.traverse_util.flatten_dict(intermediates, sep='/')
    else:
      intermediates = {}
    return variables, intermediates
  variables_shape, intermediates_shape = jax.eval_shape(_init)
  router_keys = set(intermediates_shape.keys())
  # Find how the variables are partitioned.
  variables_axis_resources = partitioning.tree_axis_resources_from_regexes(
      tree=variables_shape,
      axis_resources_regexes=config.params_axis_resources)
  variables_axis_resources = jax.tree_util.tree_map(
      lambda s: jax.sharding.NamedSharding(mesh, s), variables_axis_resources)
  variables_shape = jax.tree_util.tree_map(
      lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s),
      variables_shape, variables_axis_resources)
  # TODO(jpuigcerver): Generalize this to models with other vars than params.
  variables = flax.core.freeze({
      'params':
          checkpoints.restore_checkpoint_partitioned(
              prefix=checkpoint_prefix,
              tree=variables_shape['params']),
  })
  loss_fn = get_loss_including_auxiliary_fn(**config.get('loss'))
  return (flax_module, variables, variables_axis_resources, loss_fn,
          router_keys, extra_rng_keys)


def router_filter(mdl, _) -> bool:
  return isinstance(mdl, SUPPORTED_ROUTERS)
