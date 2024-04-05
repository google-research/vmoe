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

"""Two tower model used for contrastive learning."""
import functools
import sys
from typing import Any, Mapping, Literal, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from vmoe import utils
from vmoe.nn import vit_moe

Array = jax.Array

_default_image_module = vit_moe
_default_text_module = sys.modules[__name__]


class TextTransformer(nn.Module):
  """Text transformer similar to CLIP, allowing blocks with MoEs."""

  # Differences to CLIP text encoder (gpt-2) that I am aware of:
  # 1. https://imgur.com/HNi3jix (gpt-1)
  # 2. https://imgur.com/qKGZgBR (gpt-2)
  # 3. https://imgur.com/a/xrpYHF0 (clip)
  # - LayerNorm is on res-path (like pre-activation resnet)
  # - dropout 0.1 everywhere
  # - init as var=0.02, scaled by depth
  # - BOS and EOS tokens, take repr from EOS.
  # - self-attention is autoregressively masked.
  # - scaled in width only, with the image model.
  vocab_size: int
  num_classes: Optional[int]
  hidden_size: int
  encoder: Mapping[str, Any]
  pool_type: Literal['last', 'first', 'gap', 'gmp', 'map'] = 'last'
  deterministic: bool = False
  head_bias_init: float = 0.0
  head_kernel_zero_init: bool = False

  @property
  def kernel_init(self) -> nn.initializers.Initializer:
    if self.head_kernel_zero_init:
      return nn.initializers.zeros
    else:
      return nn.linear.default_kernel_init

  @nn.compact
  def __call__(self, text):
    # We can't use where/argwhere since the output shape is not fixed.
    # Here we use the fact that sequences are padded with EOS tokens, that the
    # EOS token has value 1, and that argmin returns the first index.
    # eos_indices = jnp.argmin(text, axis=1)

    embedding = nn.Embed(
        num_embeddings=self.vocab_size, features=self.hidden_size)
    x = embedding(text)

    # TODO(jpuigcerver): Move position embedding outside of the Encoder class.
    encoder_kwargs = dict(self.encoder)
    if encoder_kwargs.get('position_emb', {}).get('name') == 'sincos2d':
      raise ValueError(
          'sincos2d position embeddings are not supproted for text.')

    x, metrics = vit_moe.EncoderMoe(
        name='Encoder', deterministic=self.deterministic, **encoder_kwargs)(x)

    x = self.apply_pooling(x)

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

  @nn.nowrap
  def apply_pooling(self, x):
    match self.pool_type:
      case 'last': return x[:, -1, :]
      case 'first': return x[:, 0, :]
      case 'gap': return x.mean(axis=1)
      case 'gmp': return x.max(axis=1)
      case 'map':
        return vit_moe.MapHead(
            num_heads=self.encoder['num_heads'],
            mlp_dim=self.encoder['mlp_dim'],
            qk_norm=self.encoder.get('attention_qk_norm', False),
            name='MapHead')(x)
      case _:
        raise NotImplementedError(f'Cannot do pooling {self.pool_type!r}')


class TwoTower(nn.Module):
  """A two-tower encoder model."""
  image: Mapping[str, Any]
  text: Mapping[str, Any]
  scale_init: float = 1.0
  bias_init: float | None = None
  deterministic: bool = False

  @functools.cached_property
  def image_model_class(self):
    # Default model for the image encoder is a Vision Transformer with MoEs.
    model_cls = self.image.get('name', 'VisionTransformerMoe')
    model_cls, args, kwargs = utils.parse_call(model_cls, _default_image_module)
    kwargs.update({k: v for k, v in self.image.items() if k != 'name'})
    return functools.partial(
        model_cls, *args, **kwargs, deterministic=self.deterministic)

  @functools.cached_property
  def text_model_class(self):
    # Default model for the text encoder is a Text Transformer.
    model_cls = self.text.get('name', 'TextTransformer')
    model_cls, args, kwargs = utils.parse_call(model_cls, _default_text_module)
    kwargs.update({k: v for k, v in self.text.items() if k != 'name'})
    return functools.partial(
        model_cls, *args, **kwargs, deterministic=self.deterministic)

  @nn.compact
  def __call__(
      self,
      images: Array | None,
      texts: Array | None,
  ) -> Tuple[Array, Mapping[str, Any]]:
    if images is None and texts is None:
      raise ValueError('You must give at least one of images or texts arrays.')
    zimg, ztxt, metrics = None, None, {}

    if images is not None:
      zimg, metrics_img = self.image_model_class(name='img')(images)
      zimg_norm = jnp.linalg.norm(zimg, axis=-1, keepdims=True)
      zimg /= zimg_norm + 1e-8
      self.sow('intermediates', 'zimg', zimg)
      metrics['img'] = metrics_img

    if texts is not None:
      ztxt, metrics_txt = self.text_model_class(name='txt')(texts)
      ztxt_norm = jnp.linalg.norm(ztxt, axis=-1, keepdims=True)
      ztxt /= ztxt_norm + 1e-8
      self.sow('intermediates', 'ztxt', ztxt)
      metrics['txt'] = metrics_txt

    if images is None:
      # Return text embeddings and metrics.
      return ztxt, metrics
    elif texts is None:
      # Return image embeddings and metrics.
      return zimg, metrics
    else:
      # Compute logits as the dot product of the image and text embeddings.
      logits = jnp.einsum('...md,...nd->...mn', zimg, ztxt)

      # Note: Big Vision calls this "temperature", but it's actually
      # 1/temperature, if one uses the standard definition of temperature.
      scale_init = jnp.log(self.scale_init)
      s = self.param('s', nn.initializers.constant(scale_init),
                     (), jnp.float32).astype(logits.dtype)
      s = jnp.exp(s)
      logits *= s
      metrics['scale'] = s

      if self.bias_init is not None:
        b = self.param('b', nn.initializers.constant(self.bias_init),
                       (), jnp.float32).astype(logits.dtype)
        logits += b

      # Return the logits and the metrics.
      return logits, metrics
