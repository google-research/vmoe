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

"""Implements Vision Transformer with Sparse MoE layers and ensemble routing."""
from typing import Optional, Type

import flax.linen as nn
import jax.numpy as jnp
import vmoe.moe
import vmoe.nn.ensemble_routing as ensemble_routing
import vmoe.nn.routing
import vmoe.nn.vit_moe
import vmoe.utils

NoisyTopExpertsPerItemEnsembleRouter = ensemble_routing.NoisyTopExpertsPerItemEnsembleRouter
reshape_to = ensemble_routing.reshape_to_group_size_representation
reshape_from = ensemble_routing.reshape_from_group_size_representation


def _parse(ensemble_size: Optional[int]) -> int:
  if ensemble_size is None or ensemble_size <= 0:
    raise ValueError(f'The ensemble size mut be >= 1; got {ensemble_size}.')
  return ensemble_size


class MlpMoeWithNoisyTopExpertsPerItemEnsembleRouter(
    vmoe.nn.vit_moe.MlpMoeBlock):
  """Extension of MlpMoeBlock to the ensemble case.

  Attributes:
    ensemble_size: Number of ensemble members used by the ensemble routing
      strategy developed in https://arxiv.org/pdf/2110.03360.pdf.
  """
  ensemble_size: Optional[int] = None

  @nn.nowrap
  def create_router(self) -> nn.Module:
    router_kwargs = dict(num_experts=self.num_experts, **(self.router or {}))
    # By default, the router will be deterministic during inference. But we
    # allow to override it.
    router_kwargs['deterministic'] = router_kwargs.get('deterministic',
                                                       self.deterministic)
    router_kwargs['ensemble_size'] = _parse(self.ensemble_size)
    return NoisyTopExpertsPerItemEnsembleRouter(
        dtype=self.dtype, name='Router', **router_kwargs)

  @nn.compact
  def __call__(self, inputs):

    ensemble_size = _parse(self.ensemble_size)
    num_tokens_per_image = inputs.shape[1]
    # h = hidden dimension.
    # (num_seqs, num_tokens_per_image, h) --> (num_groups, group_size, h).
    inputs = reshape_to(inputs, self.group_size, ensemble_size)
    outputs, metrics = super().__call__(inputs)
    # o = output dimension.
    # (num_groups, group_size, o) --> (num_seqs, num_tokens_per_image, o).
    outputs = reshape_from(outputs, num_tokens_per_image, ensemble_size)
    return outputs, metrics


class EncoderMoeEnsemble(vmoe.nn.vit_moe.EncoderMoe):
  """Extension of EncoderMoe to the case of ensemble.

  Before the first MLP MoE layer, the batch of image tokens is repeated by a
  factor `ensemble_size` to form the different ensemble-member predictions.
  """

  @nn.compact
  def __call__(self, inputs):
    assert inputs.ndim == 3, f'Expected ndim = 3, but got shape {inputs.shape}'
    x = vmoe.nn.vit_moe.AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
        name='posembed_input')(inputs)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(x)

    dense_mlp_params = dict(mlp_dim=self.mlp_dim,
                            dropout_rate=self.dropout_rate)
    moe_mlp_params = {**dense_mlp_params, **(self.moe or {})}
    ensemble_size = _parse(moe_mlp_params.get('ensemble_size'))
    moe_mlp_layers = moe_mlp_params.pop('layers', ())

    dense_mlp_cls = vmoe.utils.partialclass(
        vmoe.nn.vit_moe.MlpBlock, **dense_mlp_params, name='Mlp')
    moe_mlp_cls = vmoe.utils.partialclass(
        MlpMoeWithNoisyTopExpertsPerItemEnsembleRouter,
        **moe_mlp_params,
        name='Moe')
    encoder_block_cls = vmoe.utils.partialclass(
        vmoe.nn.vit_moe.EncoderBlock,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        deterministic=self.deterministic,
        dtype=self.dtype)

    metrics = {}
    is_first_moe_mlp_layer = True
    for block in range(self.num_layers):
      if block in moe_mlp_layers:
        if is_first_moe_mlp_layer:
          x = jnp.repeat(x, ensemble_size, axis=0)
          is_first_moe_mlp_layer = False
        x, metrics[f'encoderblock_{block}'] = encoder_block_cls(
            name=f'encoderblock_{block}', mlp_block=moe_mlp_cls)(x)
      else:
        x = encoder_block_cls(
            name=f'encoderblock_{block}', mlp_block=dense_mlp_cls)(x)
    encoded = nn.LayerNorm(name='encoder_norm')(x)
    # Sum auxiliary losses from all blocks.
    metrics['auxiliary_loss'] = sum(
        m['auxiliary_loss'] for m in metrics.values())
    return encoded, metrics


class VisionTransformerMoeEnsemble(vmoe.nn.vit_moe.VisionTransformerMoe):
  """Vision Transformer with Sparse MoE layers and ensemble routing.

  It corresponds to the implementation of https://arxiv.org/pdf/2110.03360.pdf.
  """
  encoder_cls: Type[nn.Module] = EncoderMoeEnsemble
