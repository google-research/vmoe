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

"""Module with gating layers."""
from typing import Any, ClassVar, Iterable, Mapping, Optional, Tuple, Type, Union

import flax.linen as nn
import jax.numpy as jnp
from vit_jax import models_vit
import vmoe.moe
from vmoe.nn import routing
import vmoe.utils


Array = jnp.ndarray
PRNGKey = jnp.ndarray
AddPositionEmbs = models_vit.AddPositionEmbs
DType = type(jnp.float32)
IdentityLayer = models_vit.IdentityLayer
KwArgs = Mapping[str, Any]
Metrics = Mapping[str, Array]
MultiHeadDotProductAttention = nn.MultiHeadDotProductAttention
Shape = Iterable[int]


# Slight modification of the VisionTransformer's MlpBlock API.
class MlpBlock(models_vit.MlpBlock):
  dtype: Optional[DType] = None
  dropout_rate: float = 0.0
  deterministic: bool = False

  @nn.compact
  def __call__(self, inputs):  # pytype: disable=signature-mismatch  # overriding-parameter-name-checks
    return super().__call__(inputs, deterministic=self.deterministic)


class MlpMoeBlock(nn.Module):
  """Sparse MoE layer of MLPs.

  Attributes:
    mlp_dim: Size of the bottleneck in the MLP.
    num_experts: Number of experts in the MoE.
    group_size: Group size to use. All tokens (from all sequences) are split
      into groups of this size and routed independently.
    dropout_rate: Dropout rate used in the MLP.
    deterministic: If True, runs this layer in deterministic mode.
      Notice that the router can override this, by passing `deterministic=False`
      to the router-specific arguments.
    router: Specific parameters for the router (e.g. num_selected_experts,
      noise_std, ...).
    dtype: DType used in this layer.
    split_rngs: If True, initializes the parameters of each expert with a
      different random seed. Otherwise, it will use the same PRNG for all.
    router_cls: Router class used by the MLP MoE layer.
  """
  mlp_dim: int
  num_experts: int
  group_size: int
  dropout_rate: float = 0.0
  deterministic: bool = False
  router: Optional[KwArgs] = None
  dtype: Optional[DType] = None
  split_rngs: Union[bool, Iterable[str]] = False

  @nn.nowrap
  def create_router(self) -> nn.Module:
    router_kwargs = dict(num_experts=self.num_experts, **(self.router or {}))
    # By default, the router will be deterministic during inference. But we
    # allow to override it.
    router_kwargs['deterministic'] = router_kwargs.get('deterministic',
                                                       self.deterministic)
    # Create instance of the router class.
    router_cls = router_kwargs.pop('name', 'NoisyTopExpertsPerItemRouter')
    router_cls, router_args, router_kwargs2 = vmoe.utils.parse_call(
        router_cls, default_module=routing)
    return router_cls(*router_args, dtype=self.dtype, name='Router',
                      **(router_kwargs2 | router_kwargs))

  @nn.nowrap
  def create_split_rngs(self) -> Mapping[str, bool]:
    if isinstance(self.split_rngs, bool):
      return {'params': self.split_rngs, 'dropout': self.split_rngs}
    else:
      split_rngs = set(self.split_rngs)
      return {
          'params': 'params' in split_rngs,
          'dropout': 'dropout' in split_rngs,
      }

  @nn.compact
  def __call__(self, inputs):
    assert inputs.ndim == 3, f'Expected ndim = 3, but got shape {inputs.shape}'
    # Reshape inputs from (num_seqs, seq_length, hidden_size) to
    # (num_groups, groups_size, hidden_size).
    inputs_shape = inputs.shape
    inputs = inputs.reshape(-1, self.group_size, inputs.shape[-1])
    dispatcher, metrics = self.create_router()(inputs)
    # Use the dispatcher to apply a MoE of MlpBlocks.
    mlp_moe_layer = vmoe.moe.sparse_moe_spmd(
        MlpBlock,
        has_aux=False,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs=self.create_split_rngs())(
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            deterministic=self.deterministic,
            name='Mlp')
    outputs = mlp_moe_layer(dispatcher, inputs)
    # Reshape outputs from (num_groups, group_size, output_dim) to
    # (num_seqs, seqs_length, output_dim).
    outputs = outputs.reshape(*inputs_shape[:-1], outputs.shape[-1])
    return outputs, metrics


class MapHead(nn.Module):
  """Multihead Attention Pooling."""
  mlp_dim: int
  num_heads: int
  qk_norm: bool = False

  @nn.compact
  def __call__(self, x):
    assert x.ndim == 3, f'Unexpected ndim = {x.ndim}, it must be 3'
    probe = self.param('probe', nn.initializers.xavier_uniform(),
                       (1, 1, x.shape[-1]), x.dtype)
    probe = jnp.tile(probe, [x.shape[0], 1, 1])
    x = MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=True,
        normalize_qk=self.qk_norm,
        name='MultiHeadDotProductAttention')(inputs_q=probe, inputs_kv=x)
    y = nn.LayerNorm(name='LayerNorm')(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, deterministic=True, name='Mlp')(y)
    return (x + y)[:, 0, :]


class EncoderBlock(nn.Module):
  """Encoder block with a Sparse MoE of MLPs."""
  mlp_block: Type[nn.Module]
  num_heads: int
  dtype: Optional[DType] = None
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  attention_qk_norm: bool = False
  deterministic: bool = False

  @nn.compact
  def __call__(self, inputs):
    # Attention Block.
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=self.deterministic,
        dropout_rate=self.attention_dropout_rate,
        normalize_qk=self.attention_qk_norm,
        num_heads=self.num_heads,
        name='SelfAttention')(inputs_q=x, inputs_kv=x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(x)
    x = x + inputs
    # MLP-MoE block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = self.mlp_block(dtype=self.dtype, deterministic=self.deterministic)(y)
    if isinstance(y, jnp.ndarray):
      return x + y
    else:
      y, metrics = y
      return x + y, metrics


class EncoderMoe(nn.Module):
  """Transformer encoder with optional blocks of Sparse MoE of MLPs.

  To add MoEs to a given block of the encoder, pass a sequence of integers
  (block IDs) named 'layers' to the `moe` parameters dictionary.

  For example, to replace the MLPs in the last two layers with MoEs, use:
  ```
    encoder = EncoderMoe(
      # ... rest of arguments
      num_layers: 8,
      moe={
        'layers': (6, 7),
        # ... other MoE options
      })
  ```

  Attributes:
    num_layers: Number of encoder blocks.
    mlp_dim: Size of the bottleneck in the MLP.
    num_heads: Number of attention heads.
    dropout_rate: Dropout rate to use after the attention layer and in the MLPs.
    attention_dropout_rate: Dropout rate to use in the attention layers.
    moe: Specific parameters for the blocks with MoE layers (e.g. num_experts,
      group_size, router-specific options, etc).
    deterministic: If True, run the encoder in deterministic mode. Notice that
      the routers in the MoE layers can override this.
    dtype: DType used in this layer.
  """
  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  attention_qk_norm: bool = False
  moe: Optional[KwArgs] = None
  deterministic: bool = False
  dtype: Optional[DType] = None
  position_emb: Optional[KwArgs] = None
  # Class variable:
  DEFAULT_SINCOS2D_TEMPERATURE: ClassVar[float] = 10_000.

  @nn.compact
  def __call__(self, inputs):
    assert inputs.ndim == 3, f'Expected ndim = 3, but got shape {inputs.shape}'
    x = self.add_position_emb(inputs)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(x)

    dense_mlp_params = dict(mlp_dim=self.mlp_dim,
                            dropout_rate=self.dropout_rate)
    moe_mlp_params = {**dense_mlp_params, **(self.moe or {})}
    moe_mlp_layers = moe_mlp_params.pop('layers', ())

    dense_mlp_cls = vmoe.utils.partialclass(
        MlpBlock, **dense_mlp_params, name='Mlp')
    moe_mlp_cls = vmoe.utils.partialclass(
        MlpMoeBlock, **moe_mlp_params, name='Moe')
    encoder_block_cls = vmoe.utils.partialclass(
        EncoderBlock,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        attention_qk_norm=self.attention_qk_norm,
        deterministic=self.deterministic,
        dtype=self.dtype)

    metrics = {}
    for block in range(self.num_layers):
      if block in moe_mlp_layers:
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

  @nn.nowrap
  def add_position_emb(self, inputs):
    # By default, for back-compatibility, we use learned positional embeddings.
    position_emb = self.position_emb or {}
    name = position_emb.get('name', 'learned')
    if name == 'none':
      return inputs
    if name == 'learned':
      return AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='posembed_input')(inputs)
    if name == 'sincos2d':
      _, n, c = inputs.shape
      h = position_emb['h']
      w = position_emb['w']
      # Sincos 2D position embedding like the one used in MoCo v3.
      # https://arxiv.org/abs/2104.02057
      if c % 4 != 0 or c < 8:
        raise ValueError(f'The hidden_size={c} must be multiple of 4 and >= 8.')
      temperature = position_emb.get('temperature',
                                     self.DEFAULT_SINCOS2D_TEMPERATURE)
      y, x = jnp.mgrid[:h, :w].astype(inputs.dtype)
      y, x = y.flatten(), x.flatten()
      omega = jnp.arange(c // 4, dtype=inputs.dtype) / (c // 4 - 1)
      omega = 1. / (temperature**omega)
      x = jnp.einsum('n,d->nd', x, omega)
      y = jnp.einsum('n,d->nd', y, omega)
      # Each of the four arrays sin/cos have shape equal to (n, c//4), we
      # concatenate the four here along the last axis to get (n, c). Then we
      # simply sum that array to the inputs array (with broadcasting along the
      # batch size axis).
      posembed = jnp.concatenate(
          [jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=-1)[None, ...]
      if n == h * w:
        return inputs + posembed
      elif n == h * w + 1:
        a = inputs[:, :1, :]
        b = inputs[:, 1:, :] + posembed
        return jnp.concatenate([a, b], axis=1)
      else:
        raise ValueError(
            f'Unsupported sequence length {n=} for the given {h=} and {w=}.')
    raise ValueError(f'Unsupported position embedding: {self.position_emb}')


class VisionTransformerMoe(nn.Module):
  """Vision Transformer with Sparse MoE layers.

  This is the model used in the paper https://arxiv.org/abs/2106.05974.
  """
  num_classes: Optional[int]
  patch_size: Tuple[int, int]
  hidden_size: int
  encoder: KwArgs
  classifier: str = 'token'
  representation_size: Optional[int] = None
  deterministic: bool = False
  head_bias_init: float = 0.0
  head_kernel_zero_init: bool = True
  encoder_cls: Type[nn.Module] = EncoderMoe

  @property
  def kernel_init(self) -> nn.initializers.Initializer:
    if self.head_kernel_zero_init:
      return nn.initializers.zeros
    else:
      return nn.linear.default_kernel_init

  @nn.compact
  def __call__(self, inputs):
    # Encode patches into tokens of hidden_size.
    x = nn.Conv(
        features=self.hidden_size, kernel_size=self.patch_size,
        strides=self.patch_size, padding='VALID', name='embedding')(inputs)
    # sincos2d positional embedding needs the grid size, but the Encoder expects
    # a flatten sequence, so we must feed that info as part of the kwargs.
    encoder_kwargs = dict(self.encoder)
    if encoder_kwargs.get('position_emb', {}).get('name') == 'sincos2d':
      encoder_kwargs['position_emb'] = dict(encoder_kwargs['position_emb'])
      encoder_kwargs['position_emb']['h'] = x.shape[1]
      encoder_kwargs['position_emb']['w'] = x.shape[2]
    # Reshape images into sequences of tokens.
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, x.shape[-1]))
      cls = jnp.tile(cls, [x.shape[0], 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
    # Encode tokens unsing the MoE encoder.
    x, metrics = self.encoder_cls(
        name='Encoder', deterministic=self.deterministic, **encoder_kwargs)(x)
    # Get a single vector representation of the full sequence.
    if self.classifier == 'token' or self.classifier == '0':
      x = x[:, 0]
    elif self.classifier == 'gap':
      x = x.mean(axis=tuple(range(1, x.ndim - 1)))
    elif self.classifier == 'map':
      x = MapHead(
          num_heads=self.encoder['num_heads'], mlp_dim=self.encoder['mlp_dim'],
          qk_norm=self.encoder.get('attention_qk_norm', False),
          name='MapHead')(x)
    else:
      raise ValueError(f'Unknown classifier: {self.classifier!r}')
    if self.representation_size is not None:
      x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = IdentityLayer(name='pre_logits')(x)
    if self.num_classes:
      # Linear head outputing the logits for classification.
      logits = nn.Dense(
          features=self.num_classes,
          name='head',
          kernel_init=self.kernel_init,
          bias_init=nn.initializers.constant(self.head_bias_init))(x)
      return logits, metrics
    else:
      return x, metrics
