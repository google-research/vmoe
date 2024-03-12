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

"""Tests for vit_moe_ensemble."""
import copy

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
from vmoe.nn import vit_moe_ensemble
from vmoe.nn import vit_moe_test as t

VisionTransformerMoeEnsemble = vit_moe_ensemble.VisionTransformerMoeEnsemble


class VitMoeEnsembleTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('M=1, E=4, H=8', 1, (8, 1, 4)),
      ('M=2, E=4, H=8', 2, (8, 2, 4 // 2)))
  def test_initialize_shapes(self, ensemble_size, expected_router_shape):
    """Tests that the shapes of the parameters are the expected ones."""
    ensemble_config = copy.deepcopy(t.DEFAULT_TEST_CONFIG)
    ensemble_config['encoder']['moe']['ensemble_size'] = ensemble_size
    # The group size scales with respect to the ensemble size.
    ensemble_config['encoder']['moe']['group_size'] *= ensemble_size

    def init(rngs, x):
      model = VisionTransformerMoeEnsemble(**ensemble_config)
      return model.init(rngs, x)

    rngs = dict(params=jax.random.PRNGKey(0), gating=jax.random.PRNGKey(1))
    x = jax.random.normal(jax.random.PRNGKey(0), (16, 4, 4, 3))
    shapes = jax.tree_util.tree_map(
        lambda x: x.shape, jax.eval_shape(init, rngs, x))
    shapes = flax.core.unfreeze(shapes)
    expected_shapes = {
        'params': {
            'Encoder': {
                'encoder_norm': t.EXPECTED_DEFAULT_LAYER_NORM_SHAPES,
                'posembed_input': {'pos_embedding': (1, 4, 8)},
                'encoderblock_0': {
                    'LayerNorm_0': t.EXPECTED_DEFAULT_LAYER_NORM_SHAPES,
                    'LayerNorm_1': t.EXPECTED_DEFAULT_LAYER_NORM_SHAPES,
                    'SelfAttention': t.EXPECTED_DEFAULT_ATTENTION_SHAPES,
                    'Mlp': t.EXPECTED_DEFAULT_MLP_SHAPES,
                },
                'encoderblock_1': {
                    'LayerNorm_0': t.EXPECTED_DEFAULT_LAYER_NORM_SHAPES,
                    'LayerNorm_1': t.EXPECTED_DEFAULT_LAYER_NORM_SHAPES,
                    'SelfAttention': t.EXPECTED_DEFAULT_ATTENTION_SHAPES,
                    'Moe': {
                        'Mlp': t.EXPECTED_DEFAULT_MOE_SHAPES,
                        'Router': {'dense': {'kernel': expected_router_shape}},
                    },
                },
            },
            'embedding': {'bias': (8,), 'kernel': (2, 2, 3, 8)},
            'head': {'bias': (4,), 'kernel': (8, 4)},
        }
    }
    self.assertDictEqual(shapes, expected_shapes)

  @parameterized.named_parameters(
      ('B=16, M=2, L=(1,)', 16, 2, (1,), (16 * 2, 4)),
      ('B=16, M=2, L=(5, 7)', 16, 2, (5, 7), (16 * 2, 4)),
      ('B=16, M=2, L=()', 16, 2, tuple(), (16, 4)),
      ('B=16, M=1, L=(1,)', 16, 1, (1,), (16 * 1, 4)))
  def test_forward(self, batch_size, ensemble_size, moe_layers,
                   expected_output_shape):
    """Tests that the model runs in forward mode. Correctness is not tested."""
    ensemble_config = copy.deepcopy(t.DEFAULT_TEST_CONFIG)
    if moe_layers:
      ensemble_config['encoder']['num_layers'] = max(moe_layers) + 1
    ensemble_config['encoder']['moe']['layers'] = moe_layers
    ensemble_config['encoder']['moe']['ensemble_size'] = ensemble_size
    # The group size scales with respect to the ensemble size.
    ensemble_config['encoder']['moe']['group_size'] *= ensemble_size
    model = VisionTransformerMoeEnsemble(**ensemble_config)
    rngs = dict(params=jax.random.PRNGKey(0), gating=jax.random.PRNGKey(1))
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 4, 4, 3))
    output, _ = model.init_with_output(rngs, x)
    self.assertIsInstance(output, tuple)
    output, metrics = output
    self.assertIn('auxiliary_loss', metrics)
    self.assertTupleEqual(output.shape, expected_output_shape)


if __name__ == '__main__':
  absltest.main()
