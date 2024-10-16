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

"""Module with optimal transport (OT) routing layers."""
import functools
import flax.linen as nn
import jax
import jax.numpy as jnp
import vmoe.moe
from vmoe.nn import routing


class BaseOTNoisyTopExpertsPerItemRouter(routing.NoisyTopExpertsPerItemRouter):
  """The Base class for tokens-choose-experts routers.

  It provides the dispatcher and auxiliary losses for tokens-choose-experts
  routers.
  """
  maxiter: int = 20
  use_softmax_combine_weights: bool = True

  @nn.nowrap
  def _create_dispatcher(self, gates_dispatch, gates_combine=None):
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

    # If gates_combine is not None, use it as the combine weights; otherwise
    # use the gates_dispatch as the combine weights.
    if gates_combine is not None:
      dispatcher = _replace_combine_weights(dispatcher, gates_combine)
    if use_bfloat16:
      dispatcher = vmoe.moe.Bfloat16Dispatcher(dispatcher)
    return dispatcher


class BaseOTNoisyTopItemsPerExpertRouter(routing.NoisyTopItemsPerExpertRouter):

  """The Base class for experts-choose-tokens routers.

  It provides the dispatcher for tokens-choose-experts routers.
  """
  maxiter: int = 20
  use_softmax_combine_weights: bool = True

  @nn.nowrap
  def _create_dispatcher_and_metrics(self, gates_dispatch, gates_combine=None):
    # Creates a dispatcher implementing the TopItemsPerExpert routing algorithm.
    # Notice that each group is dispatched independently.
    dispatcher_kwargs = dict(**(self.dispatcher or {}))
    use_bfloat16 = dispatcher_kwargs.pop("bfloat16", False)
    get_top_items_per_expert_dispatcher_vmapped = jax.vmap(
        functools.partial(
            vmoe.moe.get_top_items_per_expert_dispatcher, **dispatcher_kwargs))
    dispatcher, metrics = get_top_items_per_expert_dispatcher_vmapped(
        gates_dispatch)
    if gates_combine is not None:
      dispatcher = _replace_combine_weights(dispatcher, gates_combine)
    if use_bfloat16:
      dispatcher = vmoe.moe.Bfloat16Dispatcher(dispatcher)
    return dispatcher, metrics


def _replace_combine_weights(dispatcher, replacing_gates):
  """Replace the combine_weights of a dispatcher.

  Args:
    dispatcher: A dispatcher
    replacing_gates: The gates that we use to replace the combine_weights.
  Returns:
    A dispatcher with combine_weights replaced
  """

  if isinstance(dispatcher, vmoe.moe.EinsumDispatcher):
    dispatch_weights = dispatcher.dispatch_weights
    if dispatch_weights is None:
      dispatch_weights = dispatcher.combine_weights > 0.
    # dispatch_weights is a binary array with shape (G, S, E, C)
    # replacing_gates is (G, S, E).
    combine_weights = jnp.einsum(
        "GSEC,GSE->GSEC", dispatch_weights, replacing_gates)
    dispatcher = dispatcher.replace(
        combine_weights=combine_weights, dispatch_weights=dispatch_weights)
  else:
    # For the time being, only EinsumDispatcher is supported.
    raise TypeError(f"Unsupported dispatcher: {type(dispatcher)}")
  return dispatcher


def _weighted_sum(*args):
  """Returns a weighted sum of [(weight, element), ...] for weights > 0."""
  # Note: some losses might be ill-defined in some scenarios (e.g. they may
  # have inf/NaN gradients), in those cases we don't apply them on the total
  # auxiliary loss, by setting their weights to zero.
  return sum(x * w for w, x in args if w > 0)
