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

"""Tests for mapping."""
from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from vmoe.initialization import mapping


class LayerA(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=8, name='Dense_0')(x)
    x = nn.Dense(features=16, name='Dense_1')(x)
    return x


class LayerB(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=8, name='layer_0')(x)
    x = nn.Dense(features=16, name='layer_1')(x)
    return x


class MappingTest(absltest.TestCase):

  def test_linen_variables(self):
    x = np.random.normal(size=(8, 3)).astype(np.float32)
    variables_a = LayerA().init(jax.random.PRNGKey(0), x)
    # Copy variables because map_state dict may delete the source/target data.
    variables_a_copy = jax.tree_util.tree_map(np.asarray, variables_a)
    variables_b = jax.eval_shape(LayerB().init, jax.random.PRNGKey(1), x)
    rules = [(r'params/Dense_(\d+)/(kernel|bias)', r'params/layer_\1/\2')]
    output = mapping.map_state_dict(variables_a, variables_b, rules)
    np.testing.assert_allclose(output['params']['layer_0']['kernel'],
                               variables_a_copy['params']['Dense_0']['kernel'])
    np.testing.assert_allclose(output['params']['layer_1']['kernel'],
                               variables_a_copy['params']['Dense_1']['kernel'])
    np.testing.assert_allclose(output['params']['layer_0']['bias'],
                               variables_a_copy['params']['Dense_0']['bias'])
    np.testing.assert_allclose(output['params']['layer_1']['bias'],
                               variables_a_copy['params']['Dense_1']['bias'])

  def test_skip_name(self):
    source = {'a': jnp.zeros((1,)), 'b': jnp.ones((2,))}
    target = {'c': jnp.ones((1,))}
    rules = [('a', 'c'), ('b', None)]
    output = mapping.map_state_dict(source, target, rules)
    self.assertSetEqual(set(output.keys()), {'c'})
    np.testing.assert_allclose(output['c'], np.zeros((1,)))

  def test_transform_expand_tile(self):
    source = {'a': jnp.ones((1,)), 'b': jnp.ones((2,))}
    target = {'c': jnp.zeros((5, 1)), 'd': jnp.zeros((4, 2, 3))}
    rules = [
        ('a', 'c', 'expand_tile', 0, 5),
        ('b', 'd', 'expand_tile', (0, 2), (4, 3)),
    ]
    output = mapping.map_state_dict(source, target, rules)
    np.testing.assert_allclose(output['c'], np.ones((5, 1)))
    np.testing.assert_allclose(output['d'], np.ones((4, 2, 3)))

  def test_transform_reshape(self):
    source = {'a': jnp.zeros((4, 3, 2)), 'b': np.zeros((3, 2))}
    target = {'c': jnp.ones((4, 6)), 'd': np.ones((6,))}
    rules = [
        ('a', 'c', 'reshape', (4, 6)),
        ('b', 'd', 'reshape'),  # This is allowed, but not recommended.
    ]
    output = mapping.map_state_dict(source, target, rules)
    np.testing.assert_allclose(output['c'], np.zeros((4, 6)))
    np.testing.assert_allclose(output['d'], np.zeros((6,)))

  def test_transform_squeeze(self):
    source = {'a': jnp.zeros((4, 1, 2))}
    target = {'b': jnp.ones((4, 2))}
    rules = [('a', 'b', 'squeeze', 1)]
    output = mapping.map_state_dict(source, target, rules)
    np.testing.assert_allclose(output['b'], np.zeros((4, 2)))

  def test_transform_stack(self):
    source = {
        'a_0': 1 * jnp.ones((1,)),
        'a_1': 2 * jnp.ones((1,)),
        'a_11': 3 * jnp.ones((1,)),
        'a_2': 4 * jnp.ones((1,)),
    }
    target = {'b': jnp.zeros((1, 4))}
    rules = [(r'a_\d+', 'b', 'stack', 1)]
    output = mapping.map_state_dict(source, target, rules)
    np.testing.assert_allclose(output['b'], [[1, 2, 4, 3]])

  def test_raises_shape_mismatch(self):
    source = {'a': jnp.zeros((1,))}
    target = {'b': jnp.ones((2,))}
    rules = [('a', 'b')]
    with self.assertRaisesRegex(
        ValueError, 'Target key .* has mismatched shapes.'):
      mapping.map_state_dict(source, target, rules)

  def test_no_raises_shape_mismatch(self):
    # Shapes do not match, but shapes are compatible for a broadcast.
    source = {'a': jnp.zeros((1,))}
    target = {'b': jnp.ones((1, 1, 1))}
    rules = [('a', 'b')]
    output = mapping.map_state_dict(
        source, target, rules, raise_if_shape_mismatched=False)
    np.testing.assert_allclose(output['b'], np.zeros((1, 1, 1)))

  def test_raises_source_unmatched(self):
    source = {'a': jnp.zeros((1,))}
    target = {'a': jnp.ones((1,))}
    rules = []
    with self.assertRaisesRegex(KeyError, 'Source key .* does not match'):
      mapping.map_state_dict(source, target, rules)

  def test_no_raises_source_unmatched(self):
    source = {'a': jnp.zeros((1,))}
    target = {'a': jnp.ones((1,))}
    rules = []
    output = mapping.map_state_dict(
        source, target, rules, raise_if_source_unmatched=False)
    np.testing.assert_allclose(output['a'], np.zeros((1,)))

  def test_raises_target_unmatched(self):
    source = {'a': jnp.zeros((1,))}
    target = {'b': jnp.ones((1,)), 'c': jnp.zeros((2,))}
    rules = [('a', 'b')]
    with self.assertRaisesRegex(ValueError, 'Target key .* is unmatched'):
      mapping.map_state_dict(source, target, rules)

  def test_no_raises_target_unmatched(self):
    source = {'a': jnp.zeros((1,))}
    target = {'b': jnp.ones((1,)), 'c': jnp.zeros((2,))}
    rules = [('a', 'b')]
    output = mapping.map_state_dict(
        source, target, rules, raise_if_target_unmatched=False)
    self.assertSetEqual(set(output.keys()), {'b', 'c'})
    np.testing.assert_allclose(output['b'], np.zeros((1,)))
    np.testing.assert_allclose(output['c'], np.zeros((2,)))

  def test_raises_target_not_exists(self):
    source = {'a': jnp.zeros((1,))}
    target = {'b': jnp.ones((2,))}
    rules = [('a', 'c')]
    with self.assertRaisesRegex(KeyError,
                                r'Source key .* was matched with target key '
                                r'.*, but it does not exist\.'):
      mapping.map_state_dict(source, target, rules)

  def test_raises_unknown_transform(self):
    source = {'a': jnp.zeros((1,))}
    target = {'b': jnp.ones((1,))}
    rules = [('a', 'b', 'unknown')]
    with self.assertRaisesRegex(ValueError, r'Rule .* cannot be parsed\.'):
      mapping.map_state_dict(source, target, rules)


class NaturalSortTest(parameterized.TestCase):

  @parameterized.parameters(
      (['c/3', 'a/1', 'b/2'], ['a/1', 'b/2', 'c/3']),
      (['a0', 'a11', 'a3'], ['a0', 'a3', 'a11']),
  )
  def test(self, x, expected):
    y = mapping._natural_sort(x)
    self.assertEqual(y, expected)


if __name__ == '__main__':
  absltest.main()
