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

"""Module with routing layers using an ensemble logic.

The ensemble logic is described in
  Urquhart Allingham et al., 2021 https://arxiv.org/abs/2110.03360.

Throughout the file (and following vmoe.moe.py), we use the notation:
  G = number of groups (see vmoe.moe.py)
  S = group size
  E = total number of experts
  K = number of selected experts
  M = number of ensemble members, also referred to as ensemble size.
  (we assume that E is a multiple of M).
  H = hidden size (a.k.a. number of dimensions of each token).
"""
import functools
from typing import Mapping, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import vmoe.nn.routing as routing

Array = jnp.ndarray
BaseDispatcher = routing.BaseDispatcher
Metrics = Mapping[str, Array]


def reshape_from_diag_blocks(diagonal_blocks: Array) -> Array:
  # Shape: from (G, M, S/M, E/M) to (G, S, E).
  return jax.vmap(lambda g: jax.scipy.linalg.block_diag(*g))(diagonal_blocks)


class NoisyTopExpertsPerItemEnsembleRouter(
    routing.NoisyTopExpertsPerItemRouter):
  """Noisy TopExpertsPerItem router used in https://arxiv.org/abs/2110.03360.

  The logic closely follows that of routing.NoisyTopExpertsPerItemRouter. The
  main difference lies in two features:
   (1) The batch of inputs is assumed to be tiled by a factor M, and
   (2) the set of E experts is partitioned into M subsets of E/M experts. The
   m-th part of the tiled inputs can only be routed in the m-th subset of the
   expert partition (i.e., the resulting gating matrix is block diagonal).
  """
  ensemble_size: int = 1

  @nn.compact
  def __call__(self, inputs: Array) -> Tuple[BaseDispatcher, Metrics]:
    gates_softmax, metrics = self._compute_gates_softmax_and_metrics(inputs)
    dispatcher = self._create_dispatcher(gates_softmax)
    return dispatcher, metrics

  def _compute_gates_softmax_and_metrics(
      self, inputs: Array) -> Tuple[Array, Metrics]:
    if inputs.ndim != 3:
      raise ValueError(f'inputs.ndim must be 3, but it is {inputs.ndim}')
    if self.ensemble_size <= 0:
      raise ValueError(f'Ensemble size must be >= 1; got {self.ensemble_size}.')
    if self.num_experts % self.ensemble_size != 0:
      raise ValueError(f'num_experts must be multiple of ensemble_size, but '
                       f'got {self.num_experts} and {self.ensemble_size}.')
    num_groups, group_size, hidden_size = inputs.shape
    if group_size % self.ensemble_size != 0:
      raise ValueError(f'group_size must be multiple of ensemble_size, but '
                       f'got {group_size} and {self.ensemble_size}.')
    inputs = inputs.reshape(num_groups, group_size // self.ensemble_size,
                            self.ensemble_size, hidden_size)
    # Note: The input has the S/M axis before M, but the output has it reversed.
    # This is for convenience with the reshape_from_diag_blocks function, but a
    # reshape + transpose is needed afterwards.
    # TODO(jpuigcerver,rjenatton): Find a reshape_from_diag_blocks alternative.
    # (G, S / M, M, H) -> (G, M, S / M, E / M).
    gates_softmax, metrics = self._super_compute_gates_softmax_and_metrics(
        inputs)
    # (G, M, S / M, E / M) -> (G, S, E).
    gates_softmax = reshape_from_diag_blocks(gates_softmax)
    # (G, S, E) -> (G, M, S / M, E) -> (G, S / M, M, E) -> (G, S, E).
    gates_softmax = gates_softmax.reshape(
        num_groups, self.ensemble_size, group_size // self.ensemble_size,
        self.num_experts)
    gates_softmax = gates_softmax.transpose(0, 2, 1, 3)
    gates_softmax = gates_softmax.reshape(
        num_groups, group_size, self.num_experts)
    return gates_softmax, metrics

  # Wrap the super's _compute_gates_softmax_and_metrics with vmap over both
  # inputs and parameters.
  @functools.partial(
      nn.vmap,
      variable_axes={'params': 1},
      split_rngs={'params': True, 'gating': True},
      in_axes=2,
      out_axes=1)
  def _super_compute_gates_softmax_and_metrics(
      self, inputs: Array) -> Tuple[Array, Metrics]:
    return super()._compute_gates_softmax_and_metrics(
        inputs=inputs, num_experts=self.num_experts // self.ensemble_size)
