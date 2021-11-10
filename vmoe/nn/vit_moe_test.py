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

"""Tests for vit_moe."""
from absl.testing import absltest
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


class VitMoeTest(absltest.TestCase):

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


if __name__ == '__main__':
  absltest.main()
