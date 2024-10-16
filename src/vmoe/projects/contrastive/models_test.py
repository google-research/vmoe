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

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from vmoe.projects.contrastive import models


class TwoTowerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.image_config = {
        'num_classes': self.output_dim,
        'patch_size': (2, 2),
        'hidden_size': 64,
        'classifier': 'gap',
        'encoder': {
            'num_layers': 2,
            'mlp_dim': 256,
            'num_heads': 2,
        },
        'head_kernel_zero_init': True,
    }
    self.text_config = {
        'num_classes': self.output_dim,
        'hidden_size': 64,
        'encoder': {
            'num_layers': 2,
            'mlp_dim': 256,
            'num_heads': 2,
        },
        'vocab_size': 128,
    }

  @property
  def output_dim(self) -> int:
    return 32

  def test(self):
    """Tests initialization and forward pass."""
    batch_size, height, width, text_len = 4, 8, 8, 16
    model = models.TwoTower(
        image=self.image_config,
        text=self.text_config,
        scale_init=2.0,
        bias_init=1.0,
    )

    @jax.jit
    def init_fn():
      images = jnp.zeros((batch_size, height, width, 3), dtype=jnp.float32)
      texts = jnp.zeros((batch_size, text_len), dtype=jnp.int32)
      return model.init({'params': jax.random.PRNGKey(0)}, images, texts)

    variables = init_fn()
    self.assertIn('txt', variables['params'])
    self.assertIn('img', variables['params'])
    self.assertIn('s', variables['params'])
    self.assertIn('b', variables['params'])
    # Check shape and initial values for scale and bias params.
    chex.assert_trees_all_close(
        variables['params']['s'], jnp.log(jnp.asarray(2., dtype=jnp.float32)))
    chex.assert_trees_all_close(
        variables['params']['b'], jnp.asarray(1., dtype=jnp.float32))

    @jax.jit
    def forward(variables, images, text):
      return model.apply(variables, images, text)

    # Forward with both images and text embeddings, logits' shape must be
    # (batch_size, batch_size).
    images = jnp.zeros((batch_size, height, width, 3), dtype=jnp.float32)
    texts = jnp.zeros((batch_size, text_len), dtype=jnp.int32)
    logits, _ = forward(variables, images, texts)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        logits,
        jax.ShapeDtypeStruct((batch_size, batch_size), jnp.float32))

    # Forward only images: the output should be all 0s, since the image head
    # kernel is initialized with 0.
    zimg, _ = forward(variables, images, None)
    chex.assert_trees_all_close(
        zimg, jnp.zeros((batch_size, self.output_dim), jnp.float32))

    # Forward only texts: the output should be different than 0s, since the text
    # head kernel is NOT initialized with 0s.
    ztxt, _ = forward(variables, None, texts)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        ztxt,
        jax.ShapeDtypeStruct((batch_size, self.output_dim), jnp.float32))
    self.assertGreater(jnp.abs(ztxt).sum(), 0.)


if __name__ == '__main__':
  absltest.main()
