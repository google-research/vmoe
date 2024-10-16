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

"""Soft MoE implemented as a router merging tokens as inputs/outputs of the experts.

Results using this algorithm presented in the paper:
 - "From Sparse to Soft Mixture of Experts" (https://arxiv.org/abs/2308.00951).
"""
from typing import Dict, Optional, Tuple

from absl import logging
import flax.linen as nn
import jax
import jax.numpy as jnp
from vmoe import moe

Array = jnp.ndarray
BaseDispatcher = moe.BaseDispatcher
Bfloat16Dispatcher = moe.Bfloat16Dispatcher
DType = type(jnp.float32)
EinsumDispatcher = moe.EinsumDispatcher
Initializer = jax.nn.initializers.Initializer


def normalize(x: Array, axis: int = -1, eps: float = 1e-6) -> Array:
  m = jax.lax.rsqrt(jnp.square(x).sum(axis=axis, keepdims=True) + eps)
  return x * m


class SoftRouter(nn.Module):
  """Soft router merging tokens as inputs/outputs of the experts."""
  num_experts: int
  num_slots: Optional[int] = None
  capacity_factor: Optional[float] = 1.0
  noise_std: float = 0.0
  deterministic: bool = False
  dtype: Optional[DType] = None
  mu_init: Initializer = jax.nn.initializers.lecun_normal()
  scale_init: Initializer = jax.nn.initializers.ones
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT
  partition_spec: Optional[jax.sharding.PartitionSpec] = None
  compute_similarity_metrics: bool = True

  @nn.compact
  def __call__(self, inputs: Array) -> Tuple[BaseDispatcher, Dict[str, Array]]:
    # Normalize inputs to have unit norm.
    dtype = self.dtype or inputs.dtype
    inputs = normalize(inputs.astype(dtype), axis=-1)
    # Create num_experts * num_slots parameters, normalized to have unit norm.
    _, group_size, dim = inputs.shape
    if self.num_slots is None:
      num_slots = moe.compute_capacity(
          group_size, self.num_experts, self.capacity_factor,
          ceil_or_round='round', multiple_of=1)
    else:
      num_slots = self.num_slots
      actual_capacity_factor = self.num_experts * num_slots / group_size
      pre = f'{self.capacity_factor=} ignored. ' if self.capacity_factor else ''
      logging.info(
          '%sWith num_tokens=%d, num_experts=%d and num_slots=%d, the actual '
          'capacity_factor is %f.', pre, group_size, self.num_experts,
          self.num_slots, actual_capacity_factor)
    mu = self.param('mu', self.mu_init, (dim, self.num_experts, num_slots))
    mu = normalize(mu.astype(dtype), axis=0)
    self.sow('intermediates', 'mu_unit', mu)
    # Scale inputs/mu before computing the logits.
    scale = self.param('scale', self.scale_init, ()).astype(dtype)
    if inputs.size < mu.size:
      inputs = inputs * scale
    else:
      mu = mu * scale
    # Notation:
    # g = number of groups (typically batch size).
    # m = number of items per group (typically sequence length).
    # n = number of experts.
    # p = number of slots per expert.
    # n * p = number of total slots.
    # Compute router logits between pairs of items (m) and total slots (n * p),
    # independently on each group (g).
    logits = jnp.einsum('gmd,dnp->gmnp', inputs, mu, precision=self.precision)
    logits = self.add_noise(logits)
    # Each slot takes a convex combination of the inputs.
    dispatch_weights = jax.nn.softmax(logits, axis=1)
    # Each item takes a convex combination of all the outputs of each slot.
    combine_weights = jax.nn.softmax(logits, axis=(2, 3))
    dispatcher = EinsumDispatcher(
        combine_weights, dispatch_weights,
        partition_spec=self.partition_spec, einsum_precision=self.precision)
    dispatcher = Bfloat16Dispatcher(dispatcher)
    metrics = self.get_metrics(combine_weights, dispatch_weights, mu)
    return dispatcher, metrics

  @nn.nowrap
  def add_noise(self, logits: Array) -> Array:
    if not self.deterministic and self.noise_std > 0:
      normal = jax.random.normal(self.make_rng('gating'),
                                 logits.shape, logits.dtype)
      logits = logits + self.noise_std * normal
    return logits

  @nn.nowrap
  def get_metrics(
      self,
      combine_weights: Array,
      dispatch_weights: Array,
      mu: Array) -> Dict[str, Array]:
    g, m, n, p = combine_weights.shape
    metrics = {
        'auxiliary_loss': jnp.zeros((), dtype=jnp.float32),
        'combine_weights_min_mean': combine_weights.min(axis=(2, 3)).mean(),
        'combine_weights_max_mean': combine_weights.max(axis=(2, 3)).mean(),
        'dispatch_weights_min_mean': dispatch_weights.min(axis=1).mean(),
        'dispatch_weights_max_mean': dispatch_weights.max(axis=1).mean(),
    }
    if self.compute_similarity_metrics:
      # Cosine similarity between combine weights of all pairs of tokens
      # (each group independently). Shape is (g, m, m).
      cw_sim = cosine_psim(
          combine_weights, batch_axes=(0,), contract_axes=(2, 3),
          precision=self.precision)
      cw_id = jnp.eye(m)[None, :, :]
      # Mask "diagonal" elements in the matrix with the highest value.
      cw_sim_masked = (1 - cw_id) * cw_sim
      cw_sim_masked += cw_id * cw_sim_masked.max()
      metrics['combine_weights_similarity_min'] = cw_sim_masked.min()
      metrics['combine_weights_similarity_max'] = cw_sim_masked.max()
      # Compute the mean without taking into account the "diagonal".
      metrics['combine_weights_similarity_mean'] = (
          (cw_sim.sum() - g * m) / (g * m * (m - 1)))

      # Cosine similarity between dispatch weights of all pairs of slots
      # (each group independently). Shape is (g, n, p, n, p).
      dw_sim = cosine_psim(
          dispatch_weights, batch_axes=(0,), contract_axes=(1,),
          precision=self.precision)
      dw_id = jnp.eye(n * p)[None, :, :].reshape((1, n, p, n, p))
      # Mask "diagonal" elements in the matrix with the highest value.
      dw_sim_masked = (1 - dw_id) * dw_sim
      dw_sim_masked += dw_id * dw_sim_masked.max()
      metrics['dispatch_weights_similarity_min'] = dw_sim_masked.min()
      metrics['dispatch_weights_similarity_max'] = dw_sim_masked.max()
      # Compute the mean without taking into account the "diagonal".
      metrics['dispatch_weights_similarity_mean'] = (
          (dw_sim.sum() - g * n * p) / (g * n * p * (n * p - 1)))

      # Cosine similarity between all pairs of mu's.
      assert mu.ndim == 3 and mu.shape[1:] == (n, p)
      mu_sim = cosine_psim(mu, batch_axes=(), contract_axes=(0,))
      mu_id = jnp.eye(n * p)[None, :, :].reshape((n, p, n, p))
      mu_sim_masked = (1 - mu_id) * mu_sim
      mu_sim_masked += mu_id * mu_sim_masked.max()
      metrics['mu_similarity_min'] = mu_sim_masked.min()
      metrics['mu_similarity_max'] = mu_sim_masked.max()
      metrics['mu_similarity_mean'] = (
          (mu_sim.sum() - n * p) / (n * p * (n * p - 1)))
    return metrics


def cosine_psim(
    x: Array,
    batch_axes: Tuple[int, ...],
    contract_axes: Tuple[int, ...],
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    eps: float = 1e-9) -> Array:
  """Compute the pairwise cosine similarity of an array with arbitrary dims."""
  # Normalize array along the contract axes.
  m = jax.lax.rsqrt(jnp.square(x).sum(axis=contract_axes, keepdims=True) + eps)
  x = x * m
  # Compute dot product between all pair of vectors of each batch.
  dot_dim_nums = ((contract_axes, contract_axes), (batch_axes, batch_axes))
  return jax.lax.dot_general(x, x, dot_dim_nums, precision)
