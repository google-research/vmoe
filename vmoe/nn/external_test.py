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

"""Tests for external models."""
import abc

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import ml_collections
from vmoe.nn import external


class BaseExternalTest(abc.ABC):

  @abc.abstractmethod
  def create_model(self, deterministic: bool):
    pass

  def test_forward(self):
    model = self.create_model(deterministic=True)

    @jax.jit
    def fn(x):
      variables = model.init(jax.random.PRNGKey(0), x)
      return model.apply(variables, x)

    x = jax.random.normal(jax.random.PRNGKey(0), (8, 64, 64, 3))
    logits, metrics = fn(x)
    self.assertEqual(logits.shape, (8, 10))
    self.assertEmpty(metrics)

  def test_backward(self):
    model = self.create_model(deterministic=False)

    @jax.jit
    def fn(x):

      @jax.jit
      @jax.grad
      def compute_grad(params):
        logits, _ = model.apply({'params': params}, x)
        target = jax.random.normal(jax.random.PRNGKey(1), logits.shape)
        return jnp.sum(jnp.square(logits - target))

      variables = model.init(jax.random.PRNGKey(0), x)
      params = variables['params']
      for _ in range(2):
        grads = compute_grad(params)
        params = jax.tree_map(lambda p, g: p - 0.01 * g, params, grads)

      return grads

    x = jax.random.normal(jax.random.PRNGKey(0), (8, 64, 64, 3))
    grads = fn(x)
    grads_norm = jax.tree_map(lambda x: jnp.linalg.norm(x.flatten()), grads)
    zeros = jax.tree_map(jnp.zeros_like, grads_norm)
    print(grads_norm)
    chex.assert_trees_all_equal_comparator(lambda x, y: x > y,
                                           '{} is not greater than {}'.format,
                                           grads_norm, zeros)


class MlpMixerTest(absltest.TestCase, BaseExternalTest):

  def create_model(self, deterministic: bool):
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_dim = 64
    config.num_blocks = 1
    config.tokens_mlp_dim = 128
    config.channels_mlp_dim = 128
    config.num_classes = 10
    return external.MlpMixer(**config, deterministic=deterministic)


class VisionTransformerTest(absltest.TestCase, BaseExternalTest):

  def create_model(self, deterministic: bool):
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 64
    config.num_classes = 10
    config.transformer = ml_collections.ConfigDict({
        'num_layers': 1,
        'mlp_dim': 128,
        'num_heads': 2,
        'dropout_rate': 0.0,
        'attention_dropout_rate': 0.0,
    })
    return external.VisionTransformer(**config, deterministic=deterministic)


if __name__ == '__main__':
  absltest.main()
