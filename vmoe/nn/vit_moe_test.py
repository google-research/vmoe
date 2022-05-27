# Copyright 2022 Google LLC.
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

"""Tests for vit_moe."""
import copy

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from vmoe.nn import vit_moe

# Default configuration for the V-MoE.
DEFAULT_TEST_CONFIG = {
    'num_classes': 4,
    'patch_size': (2, 2),
    'hidden_size': 8,
    'encoder': {
        'num_layers': 2,
        'mlp_dim': 16,
        'num_heads': 2,
        'moe': {
            'layers': (1,),
            'num_experts': 4,
            'group_size': 4,
            'router': {
                'num_selected_experts': 1,
                'noise_std': 1e-3,
                'importance_loss_weight': 0.02,
                'load_loss_weight': 0.02,
                'dispatcher': {
                    'name': 'einsum',
                    'capacity': 2,
                    'batch_priority': False,
                    'bfloat16': False,
                }
            },
        },
        'dropout_rate': 0.0,
        'attention_dropout_rate': 0.0,
    },
    'classifier': 'gap',
    'representation_size': None,
}

EXPECTED_DEFAULT_ATTENTION_SHAPES = {
    'key': {'bias': (2, 4), 'kernel': (8, 2, 4)},
    'out': {'bias': (8,), 'kernel': (2, 4, 8)},
    'query': {'bias': (2, 4), 'kernel': (8, 2, 4)},
    'value': {'bias': (2, 4), 'kernel': (8, 2, 4)},
}
EXPECTED_DEFAULT_MLP_SHAPES = {
    'Dense_0': {'bias': (16,), 'kernel': (8, 16)},
    'Dense_1': {'bias': (8,), 'kernel': (16, 8)},
}
EXPECTED_DEFAULT_MOE_SHAPES = {
    'Dense_0': {'bias': (4, 16), 'kernel': (4, 8, 16)},
    'Dense_1': {'bias': (4, 8), 'kernel': (4, 16, 8)},
}
EXPECTED_DEFAULT_LAYER_NORM_SHAPES = {'bias': (8,), 'scale': (8,)}


class VitMoeTest(parameterized.TestCase):

  def test_initialize_shapes(self):
    """Tests that the shapes of the parameters are the expected ones."""
    def init(rngs, x):
      model = vit_moe.VisionTransformerMoe(**DEFAULT_TEST_CONFIG)
      return model.init(rngs, x)

    rngs = dict(params=jax.random.PRNGKey(0), gating=jax.random.PRNGKey(1))
    x = jax.random.normal(jax.random.PRNGKey(0), (16, 4, 4, 3))
    shapes = jax.tree_map(lambda x: x.shape, jax.eval_shape(init, rngs, x))
    shapes = shapes.unfreeze()
    expected_shapes = {
        'params': {
            'Encoder': {
                'encoder_norm': EXPECTED_DEFAULT_LAYER_NORM_SHAPES,
                'posembed_input': {'pos_embedding': (1, 4, 8)},
                'encoderblock_0': {
                    'LayerNorm_0': EXPECTED_DEFAULT_LAYER_NORM_SHAPES,
                    'LayerNorm_1': EXPECTED_DEFAULT_LAYER_NORM_SHAPES,
                    'SelfAttention': EXPECTED_DEFAULT_ATTENTION_SHAPES,
                    'Mlp': EXPECTED_DEFAULT_MLP_SHAPES,
                },
                'encoderblock_1': {
                    'LayerNorm_0': EXPECTED_DEFAULT_LAYER_NORM_SHAPES,
                    'LayerNorm_1': EXPECTED_DEFAULT_LAYER_NORM_SHAPES,
                    'SelfAttention': EXPECTED_DEFAULT_ATTENTION_SHAPES,
                    'Moe': {
                        'Mlp': EXPECTED_DEFAULT_MOE_SHAPES,
                        'Router': {'dense': {'kernel': (8, 4)}},
                    },
                },
            },
            'embedding': {'bias': (8,), 'kernel': (2, 2, 3, 8)},
            'head': {'bias': (4,), 'kernel': (8, 4)},
        }
    }
    self.assertDictEqual(shapes, expected_shapes)

  def test_forward(self):
    """Tests that the model runs in forward mode. Correctness is not tested."""
    model = vit_moe.VisionTransformerMoe(**DEFAULT_TEST_CONFIG)
    rngs = dict(params=jax.random.PRNGKey(0), gating=jax.random.PRNGKey(1))
    x = jax.random.normal(jax.random.PRNGKey(0), (16, 4, 4, 3))
    output, _ = model.init_with_output(rngs, x)
    self.assertIsInstance(output, tuple)
    output, metrics = output
    self.assertIn('auxiliary_loss', metrics)
    self.assertTupleEqual(output.shape, (16, 4))

  def test_forward_moe_dropout(self):
    config = copy.deepcopy(DEFAULT_TEST_CONFIG)
    config['encoder']['moe']['dropout_rate'] = 0.2
    config['encoder']['moe']['split_rngs'] = ('dropout',)
    model = vit_moe.VisionTransformerMoe(**config, deterministic=False)
    rngs = dict(params=jax.random.PRNGKey(0),
                gating=jax.random.PRNGKey(1),
                dropout=jax.random.PRNGKey(2))
    x = jax.random.normal(jax.random.PRNGKey(0), (16, 4, 4, 3))
    variables = model.init(rngs, x)
    variables = variables.unfreeze()
    variables['params']['head']['kernel'] = jax.random.normal(
        jax.random.PRNGKey(0), variables['params']['head']['kernel'].shape)
    output1, _ = model.apply(
        variables, x, rngs=dict(gating=jax.random.PRNGKey(1),
                                dropout=jax.random.PRNGKey(2)))
    output2, _ = model.apply(
        variables, x, rngs=dict(gating=jax.random.PRNGKey(1),
                                dropout=jax.random.PRNGKey(3)))
    different_fn = lambda x, y: jax.numpy.abs(x - y).sum() > 0.01
    error_msg_fn = lambda x, y: f'{x} is too close to {y}'
    chex.assert_trees_all_equal_comparator(
        different_fn, error_msg_fn, output1, output2)

  @parameterized.named_parameters(
      ('map', 'map', 4, {'MapHead': {
          'probe': (1, 1, 8),
          'MultiHeadDotProductAttention': EXPECTED_DEFAULT_ATTENTION_SHAPES,
          'LayerNorm': EXPECTED_DEFAULT_LAYER_NORM_SHAPES,
          'Mlp': EXPECTED_DEFAULT_MLP_SHAPES,
      }}),
      ('gap', 'gap', 4, {}),
      ('token', 'token', 5, {}),
  )
  def test_classifier(self, classifier, seq_length, params_subset):
    config = copy.deepcopy(DEFAULT_TEST_CONFIG)
    config['classifier'] = classifier
    model = vit_moe.VisionTransformerMoe(**config)
    rngs = dict(params=jax.random.PRNGKey(0), gating=jax.random.PRNGKey(1))
    x = jax.ShapeDtypeStruct((16, 4, 4, 3), jax.numpy.float32)
    shapes = jax.tree_map(lambda x: x.shape,
                          jax.eval_shape(model.init, rngs, x))
    shapes = shapes.unfreeze()
    self.assertDictEqual(shapes['params']['Encoder']['posembed_input'],
                         {'pos_embedding': (1, seq_length, 8)})
    self.assertDictContainsSubset(params_subset, shapes['params'])


if __name__ == '__main__':
  absltest.main()
