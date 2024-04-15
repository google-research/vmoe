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

"""Module with routing layers."""
import functools
from typing import Any, Mapping, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import vmoe.moe

Array = jnp.ndarray
BaseDispatcher = vmoe.moe.BaseDispatcher
DType = type(jnp.float32)
KwArgs = Mapping[str, Any]
Metrics = Mapping[str, Array]


class NoisyTopExpertsPerItemRouter(nn.Module):
  """Noisy TopExpertsPerItem router used in https://arxiv.org/abs/2106.05974.

  First, a dense (i.e. the gating) layer computes logits for each pair of
  (item, expert). Noise is added to these logits. The logits are normalized
  using a softmax over the expert dimension. This score will be used to
  determine which items are dispatched to which experts and how the outputs of
  the experts are combined.

  Because the routing algorithm is non-differentiable, the only way to train the
  parameters of the dense (a.k.a. gating layer) is through the weights used
  to combine the output of the experts, and through two auxiliary losses that
  depend on the output of the gating.
  """
  num_experts: int
  num_selected_experts: int = 1
  noise_std: float = 1.0
  gshard_loss_weight: float = 0.0
  importance_loss_weight: float = 1.0
  load_loss_weight: float = 1.0
  dispatcher: Optional[KwArgs] = None
  deterministic: bool = False
  dtype: Optional[DType] = None

  @nn.compact
  def __call__(self, inputs: Array) -> Tuple[BaseDispatcher, Metrics]:
    gates_softmax, metrics = self._compute_gates_softmax_and_metrics(
        inputs, self.num_experts)
    dispatcher = self._create_dispatcher(gates_softmax)
    return dispatcher, metrics

  @nn.nowrap
  def _compute_gates_softmax_and_metrics(
      self, inputs: Array, num_experts: int) -> Tuple[Array, Metrics]:
    if inputs.ndim != 3:
      raise ValueError(f"inputs.ndim must be 3, but it is {inputs.ndim}")
    if not num_experts >= self.num_selected_experts >= 1:
      raise ValueError(f"num_experts >= num_selected_experts >= 1, but got "
                       f"num_experts = {num_experts} and "
                       f"num_selected_experts = {self.num_selected_experts}.")
    dtype = self.dtype or inputs.dtype
    # Compute the gating logits for each pair of (item, expert).
    gates_logits = nn.Dense(features=num_experts, use_bias=False,
                            dtype=dtype, name="dense")(inputs)
    # Compute the auxiliary losses defined in Appendix A.2, from
    # https://arxiv.org/abs/2106.05974. Notice that the "Load Loss" can only be
    # computed if the router is stochastic (i.e. deterministic = False).
    # Notice that the auxiliary losses are computed on each group independently
    # (i.e. through the vmaps surrounding the calls).
    gates_softmax = jax.nn.softmax(gates_logits)
    importance_loss = jax.vmap(self._importance_auxiliary_loss)(gates_softmax)
    if self.deterministic or self.noise_std == 0.0:
      gshard_loss = jax.vmap(self._gshard_auxiliary_loss)(gates_softmax)
      metrics = {
          "auxiliary_loss": _weighted_sum(
              (self.gshard_loss_weight, gshard_loss),
              (self.importance_loss_weight, importance_loss)),
          "gshard_loss": gshard_loss,
          "importance_loss": importance_loss,
      }
      return gates_softmax, metrics
    else:
      noise_std = (1.0 / num_experts) * self.noise_std
      logits_noise = noise_std * jax.random.normal(
          key=self.make_rng("gating"), shape=gates_logits.shape)
      gates_logits_noisy = gates_logits + logits_noise
      gates_softmax_noisy = jax.nn.softmax(gates_logits_noisy)
      load_loss = jax.vmap(
          functools.partial(
              self._load_auxiliary_loss,
              num_selected_experts=self.num_selected_experts,
              noise_std=noise_std))(gates_logits, gates_logits_noisy)
      gshard_loss = jax.vmap(self._gshard_auxiliary_loss)(gates_softmax_noisy)

      metrics = {
          "auxiliary_loss": _weighted_sum(
              (self.gshard_loss_weight, gshard_loss),
              (self.importance_loss_weight, importance_loss),
              (self.load_loss_weight, load_loss)),
          "gshard_loss": gshard_loss,
          "importance_loss": importance_loss,
          "load_loss": load_loss,
      }
      return gates_softmax_noisy, metrics

  @nn.nowrap
  def _create_dispatcher(self, gates_dispatch):
    # Creates a dispatcher implementing the TopExpertsPerItem routing algorithm,
    # that uses at most `num_selected_experts` per item. Notice that each
    # group is dispatched independently.
    dispatcher_kwargs = dict(**(self.dispatcher or {}))
    use_bfloat16 = dispatcher_kwargs.pop("bfloat16", False)
    get_top_experts_per_item_dispatcher_vmapped = jax.vmap(
        functools.partial(
            vmoe.moe.get_top_experts_per_item_dispatcher,
            num_selected_experts=self.num_selected_experts,
            **dispatcher_kwargs))
    dispatcher = get_top_experts_per_item_dispatcher_vmapped(gates_dispatch)
    if use_bfloat16:
      dispatcher = vmoe.moe.Bfloat16Dispatcher(dispatcher)
    return dispatcher

  @classmethod
  def _gshard_auxiliary_loss(cls, gates: Array) -> Array:
    # See `l_{aux}` in Algorithm 1 in https://arxiv.org/pdf/2006.16668.pdf.
    _, num_experts = gates.shape
    # Line (3) in Algorithm 1.
    mean_gates_per_expert = gates.mean(axis=0)
    # Lines (11, 13) in Algorithm 1.
    mean_top1_per_expert = jax.nn.one_hot(
        jnp.argmax(gates, axis=1), num_experts, dtype=jnp.int32).mean(axis=0)
    # Note: Only gradients through mean_gates_per_expert affect the gating,
    # since hard counts from top_k+one_hot are not differentiable.
    auxiliary_loss = jnp.mean(mean_top1_per_expert * mean_gates_per_expert)
    # Note: Not mentioned in the paper, but it's done in their source code.
    # https://github.com/tensorflow/lingvo/blob/84b85514d7ad3652bc9720cb45acfab08604519b/lingvo/core/gshard_layers.py#L2223
    auxiliary_loss *= num_experts**2
    return auxiliary_loss

  @classmethod
  def _importance_auxiliary_loss(cls, gates: Array) -> Array:
    axis = tuple(range(gates.ndim - 1))  # All except last.
    importance_per_expert = jnp.sum(gates, axis=axis)
    std_importance_per_expert = jnp.std(importance_per_expert)
    mean_importance_per_expert = jnp.mean(importance_per_expert)
    # Compute coefficient of variation (i.e. std/mean) squared.
    return (std_importance_per_expert / mean_importance_per_expert)**2

  @classmethod
  def _load_auxiliary_loss(cls, logits: Array, logits_noisy: Array,
                           noise_std: Array,
                           num_selected_experts: int) -> Array:
    # For each example, compute the weight required for an expert to be selected
    # among the top-k.
    # NOTE: DO NOT TRY TO SIMPLIFY THIS. This convoluted way of obtaining the
    # threshold_per_item avoids adding all-gather ops during backpropagation.
    num_experts = logits_noisy.shape[-1]
    threshold_per_item_index = jax.lax.top_k(
        logits_noisy, num_selected_experts)[-1][..., -1]
    threshold_per_item = jnp.sum(
        jax.nn.one_hot(threshold_per_item_index, num_experts) * logits_noisy,
        axis=-1)
    # For each example and expert, find how far they were from the threshold and
    # normalize this value by the noise_std to use the standard Gaussian CDF.
    noise_required_to_win = threshold_per_item[..., None] - logits
    noise_required_to_win /= noise_std
    # p is the probability of being above the threshold for each (item, expert)
    # if the random noise (with its std) was re-sampled again.
    p = 1. - jax.scipy.stats.norm.cdf(noise_required_to_win)
    # We compute the average such probability for each expert over examples.
    p_mean = jnp.mean(p, axis=0)
    # Compute p_mean's coefficient of variation squared.
    return (jnp.std(p_mean) / jnp.mean(p_mean))**2


class NoisyTopItemsPerExpertRouter(nn.Module):
  """Noisy TopItemsPerExpert router.

  Instead of picking the Top-K experts with highest score for each item, and
  then ignore choices that exceed the capacity (C) of any given expert, here we
  pick the Top-C items with highest score for each expert.

  This makes the load across experts automatically balanced, however the number
  of experts assigned to each item is not bounded and can vary. Some items may
  not be routed to any expert. In practice, though, this works very well.

  This was coined "Experts Choice Routing" in https://arxiv.org/abs/2202.09368.
  """
  num_experts: int
  noise_std: float = 1.0
  dispatcher: Optional[KwArgs] = None
  deterministic: bool = False
  dtype: Optional[DType] = None

  @nn.compact
  def __call__(self, inputs: Array) -> Tuple[BaseDispatcher, Metrics]:
    gates_softmax = self._compute_gates_softmax(inputs, self.num_experts)
    dispatcher, metrics = self._create_dispatcher_and_metrics(gates_softmax)
    metrics["auxiliary_loss"] = 0.
    return dispatcher, metrics  # pytype: disable=bad-return-type

  @nn.nowrap
  def _compute_gates_softmax(self, inputs: Array, num_experts: int) -> Array:
    if inputs.ndim != 3:
      raise ValueError(f"inputs.ndim must be 3, but it is {inputs.ndim}")
    dtype = self.dtype or inputs.dtype
    # Compute the gating logits for each pair of (item, expert).
    gates_logits = nn.Dense(features=num_experts, use_bias=False,
                            dtype=dtype, name="dense")(inputs)
    if self.deterministic or self.noise_std == 0.0:
      gates_softmax = jax.nn.softmax(gates_logits)
      return gates_softmax
    else:
      noise_std = (1.0 / num_experts) * self.noise_std
      logits_noise = noise_std * jax.random.normal(
          key=self.make_rng("gating"), shape=gates_logits.shape)
      gates_logits_noisy = gates_logits + logits_noise
      gates_softmax_noisy = jax.nn.softmax(gates_logits_noisy)
      return gates_softmax_noisy

  @nn.nowrap
  def _create_dispatcher_and_metrics(self, gates_dispatch):
    # Creates a dispatcher implementing the TopItemsPerExpert routing algorithm.
    # Notice that each group is dispatched independently.
    dispatcher_kwargs = dict(**(self.dispatcher or {}))
    use_bfloat16 = dispatcher_kwargs.pop("bfloat16", False)
    get_top_items_per_expert_dispatcher_vmapped = jax.vmap(
        functools.partial(
            vmoe.moe.get_top_items_per_expert_dispatcher, **dispatcher_kwargs))
    dispatcher, metrics = get_top_items_per_expert_dispatcher_vmapped(
        gates_dispatch)
    if use_bfloat16:
      dispatcher = vmoe.moe.Bfloat16Dispatcher(dispatcher)
    return dispatcher, metrics


def _weighted_sum(*args):
  """Returns a weighted sum of [(weight, element), ...] for weights > 0."""
  # Note: some losses might be ill-defined in some scenarios (e.g. they may
  # have inf/NaN gradients), in those cases we don't apply them on the total
  # auxiliary loss, by setting their weights to zero.
  return sum(x * w for w, x in args if w > 0)
