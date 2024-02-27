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
from absl.testing import parameterized
import jax
from jax.experimental import pjit
import jax.numpy as jnp
import numpy as np
from vmoe.initialization import rules as _rules


class ZoomTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('d',))

  def wrap_fn(self, fn, use_pjit):
    if use_pjit:
      return self.mesh(
          pjit.pjit(
              fn, in_shardings=(), out_shardings=jax.sharding.PartitionSpec()
          )
      )
    else:
      return jax.jit(fn)

  def test_parse(self):
    rules = _rules.Rules.parse([('a', 'b', 'zoom')])
    self.assertLen(rules.rules, 1)
    self.assertIsInstance(rules.rules[0], _rules.ZoomRule)
    source = jnp.ones((6, 4, 8), dtype=jnp.float32)
    target = jax.ShapeDtypeStruct((6, 8, 12), jnp.float32)
    self.assertIsInstance(rules.rules[0].get_transformation(source, target),
                          _rules.ZoomTransformation)

  @parameterized.named_parameters(('jit', False), ('pjit', True))
  def test_transform(self, use_pjit):
    def zoom_fn():
      target_shape_dtype = jax.ShapeDtypeStruct(
          shape=(6, 8, 12), dtype=jnp.float32)
      tx = _rules.ZoomTransformation(
          source=jnp.ones((6, 4, 8)),
          target=None,
          target_shape_dtype=target_shape_dtype)
      return tx()

    zoom_fn = self.wrap_fn(zoom_fn, use_pjit)
    np.testing.assert_allclose(zoom_fn(), jnp.ones((6, 8, 12)))


class VitZoomTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Mesh used with pjit.
    self.mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('d',))

  def wrap_fn(self, fn, use_pjit):
    if use_pjit:
      return self.mesh(
          pjit.pjit(
              fn, in_shardings=(), out_shardings=jax.sharding.PartitionSpec()
          )
      )
    else:
      return jax.jit(fn)

  def test_parse(self):
    rules = _rules.Rules.parse([('a', 'b', 'vit_zoom')])
    self.assertLen(rules.rules, 1)
    self.assertIsInstance(rules.rules[0], _rules.VitZoomRule)

  @parameterized.named_parameters(('jit', False), ('pjit', True))
  def test_transform_no_target_tok(self, use_pjit):
    def zoom_fn():
      target_shape_dtype = jax.ShapeDtypeStruct(
          shape=(1, 16, 14), dtype=jnp.float32)
      tx = _rules.VitZoomTransformation(
          source=jnp.ones((1, 64, 14)),
          target=None,
          target_shape_dtype=target_shape_dtype)
      return tx()

    zoom_fn = self.wrap_fn(zoom_fn, use_pjit)
    np.testing.assert_allclose(zoom_fn(), jnp.ones((1, 16, 14)))

  @parameterized.named_parameters(('jit', False), ('pjit', True))
  def test_transform_target_tok_source_tok(self, use_pjit):
    def zoom_fn():
      target_shape_dtype = jax.ShapeDtypeStruct(
          shape=(1, 16 + 1, 14), dtype=jnp.float32)
      tx = _rules.VitZoomTransformation(
          source=jnp.ones((1, 64 + 1, 14)),
          target=jnp.zeros((1, 16 + 1, 14)),
          target_shape_dtype=target_shape_dtype)
      return tx()

    zoom_fn = self.wrap_fn(zoom_fn, use_pjit)
    np.testing.assert_allclose(zoom_fn(), jnp.ones((1, 16 + 1, 14)))

  @parameterized.named_parameters(('jit', False), ('pjit', True))
  def test_transform_target_tok_array_no_source_tok(self, use_pjit):
    def zoom_fn():
      target_shape_dtype = jax.ShapeDtypeStruct(
          shape=(1, 16 + 1, 14), dtype=jnp.float32)
      tx = _rules.VitZoomTransformation(
          source=jnp.ones((1, 64, 14)),
          target=jnp.zeros((1, 16 + 1, 14)),
          target_shape_dtype=target_shape_dtype)
      return tx()

    zoom_fn = self.wrap_fn(zoom_fn, use_pjit)
    expected = jnp.concatenate(
        [jnp.zeros((1, 1, 14)), jnp.ones((1, 16, 14))], axis=1)
    np.testing.assert_allclose(zoom_fn(), expected)

  @parameterized.named_parameters(('jit', False), ('pjit', True))
  def test_transform_target_tok_shape_dtype_no_source_tok(self, use_pjit):
    def zoom_fn():
      target_shape_dtype = jax.ShapeDtypeStruct(
          shape=(1, 16 + 1, 14), dtype=jnp.float32)
      tx = _rules.VitZoomTransformation(
          source=jnp.ones((1, 64, 14)),
          target=None,
          target_shape_dtype=target_shape_dtype)
      return tx()
    # Test only from index=1: in axis=1, since index=0 is Gaussian noise.
    zoom_fn = self.wrap_fn(zoom_fn, use_pjit)
    np.testing.assert_allclose(zoom_fn()[:, 1:, :], jnp.ones((1, 16, 14)))


if __name__ == '__main__':
  absltest.main()
