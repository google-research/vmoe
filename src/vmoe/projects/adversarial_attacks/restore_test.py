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

"""Tests for restore.py."""
import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import numpy as np
from vmoe.projects.adversarial_attacks import restore


Bfloat16Dispatcher = restore.moe.Bfloat16Dispatcher
EinsumDispatcher = restore.moe.EinsumDispatcher
ExpertIndicesDispatcher = restore.moe.ExpertIndicesDispatcher


class ComputeLossPredictCwFnTest(absltest.TestCase):

  def test(self):

    def apply_fn(x, rngs, capture_intermediates):
      del x, rngs, capture_intermediates
      logits = np.asarray([[.1, .3, .2], [.4, .5, .6]])
      metrics, intermediates = {}, {}
      return (logits, metrics), intermediates

    def loss_fn(logits, labels, metrics):
      del logits, labels, metrics
      return np.asarray([.7, .8])

    x = np.zeros((2, 10), dtype=np.float32)
    y = np.asarray([[0, 0, 1], [0, 0, 1]])
    loss, pred, correct, combine_weights = jax.jit(
        functools.partial(
            restore.compute_loss_predict_cw_fn,
            apply_fn=apply_fn,
            loss_fn=loss_fn))(x, y, {})
    np.testing.assert_allclose(loss, [.7, .8])
    np.testing.assert_array_equal(pred, [1, 2])
    np.testing.assert_array_equal(correct, [0, 1])
    self.assertEmpty(combine_weights)


class CreateMeshTest(absltest.TestCase):

  def test(self):
    mesh = restore.create_mesh(1)
    self.assertEqual(mesh.axis_names, ('expert', 'replica'))
    self.assertEqual(mesh.devices.shape, (1, jax.device_count()))

  def test_override_num_expert_partitions(self):
    mesh = restore.create_mesh(10**12)
    self.assertEqual(mesh.axis_names, ('expert', 'replica'))
    self.assertEqual(mesh.devices.shape, (1, jax.device_count()))


class GetCombineWeightsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # Test with no intermediates (i.e. no MoE layers).
      ('_no_intermediates', {}, {}),
      # Test with intermediates from MoEs using EinsumDispatcher.
      # The shape of the combine_weights in the dispatcher is:
      # (G=1, S=2, E=3, C=2). The shape of the output is (G * S=1, E=3).
      ('_einsum', {
          'intermediates': {
              'foo': [[
                  EinsumDispatcher(
                      combine_weights=np.asarray([[
                          [[.2, .0], [.0, .1], [.8, .0]],
                          [[.0, .9], [.5, .0], [.0, .2]],
                      ]])),
              ]],
          }
      }, {
          'foo': np.asarray([[.2, .1, .8], [.9, .5, .2]])
      }),
      # Test with intermediates from MoEs using ExpertIndicesDispatcher.
      # The shape of the original combine_weights is (G=1, S=2, K=3, 2).
      # The shape of the output is (G * S = 2, E=3).
      ('_indices', {
          'intermediates': {
              'foo': [[
                  ExpertIndicesDispatcher(
                      indices=np.asarray([[
                          [(0, 0), (1, 1), (2, 0)],
                          [(0, 1), (1, 0), (2, 1)],
                      ]]),
                      combine_weights=np.asarray([[
                          [.2, .1, .8],
                          [.9, .5, .2],
                      ]]),
                      num_experts=3,
                      capacity=2),
              ]],
          }
      }, {
          'foo': np.asarray([[.2, .1, .8], [.9, .5, .2]])
      }),
      # Test with intermediates from MoEs wrapping their dispatcher with a
      # Bfloat16Dispatcher. The original combine_weights are trivial (all 0s),
      # so here we are only testing that the Bfloat16Dispatcher is handled
      # correctly, and that the output shapes are correct.
      ('_bfloat16', {
          'intermediates': {
              'foo': {
                  'bar': [[
                      Bfloat16Dispatcher(
                          dispatcher=EinsumDispatcher(
                              combine_weights=np.zeros((5, 4, 3, 2)))),
                  ]],
              }
          }
      }, {
          'foo/bar': np.zeros((20, 3)),
      }),
  )
  def test(self, intermediates, expected_output):
    chex.assert_trees_all_close(
        jax.jit(restore.get_combine_weights)(intermediates),
        expected_output)


class GetLossFnTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_sigmoid',
       'sigmoid_xent',
       np.log(np.asarray([.1, .3, .6])) - np.log(1 - np.asarray([.1, .3, .6])),
       [-np.log(.9) - np.log(.3) - np.log(.4)]),
      ('_softmax',
       'softmax_xent', np.log(np.asarray([[.1, .3, .6]])), [-np.log(.3)]),
  )
  def test(self, name, logits, expected_output):
    fn = jax.jit(restore.get_loss_fn(name))
    output = fn(logits, np.asarray([[.0, 1., .0]]))
    np.testing.assert_allclose(output, expected_output, rtol=1e-5)

  def test_unknown_name(self):
    with self.assertRaises(ValueError):
      restore.get_loss_fn('foo')


class GetLossWithAuxiliaryFnTest(parameterized.TestCase):

  @parameterized.parameters((True, [-np.log(.3) + .25]), (False, [-np.log(.3)]))
  def test(self, attack_auxiliary_loss, expected_output):
    logits = np.log(np.asarray([[.1, .3, .6]]))
    labels = np.asarray([[.0, 1., .0]])
    intermediate = {'auxiliary_loss': np.asarray([.2, .3])}
    loss_with_aux_fn = restore.get_loss_including_auxiliary_fn(
        name='softmax_xent', attack_auxiliary_loss=attack_auxiliary_loss)
    loss = loss_with_aux_fn(logits, labels, intermediate)
    np.testing.assert_allclose(loss, expected_output, rtol=1e-5)


class RestoreFromConfigTest(absltest.TestCase):

  @mock.patch.object(restore.checkpoints, 'restore_checkpoint_partitioned',
                     autospec=True)
  def test(self, unused_mock_restore_checkpoint):
    config = restore.ml_collections.ConfigDict({
        'model': {
            'name': 'VisionTransformer',
            'num_classes': 10,
            'patches': {'size': (2, 2)},
            'hidden_size': 8,
            'transformer': {
                'num_layers': 1,
                'num_heads': 1,
                'mlp_dim': 8,
                'dropout_rate': 0.,
                'attention_dropout_rate': 0.,
            },
        },
        'loss': {'name': 'sigmoid_xent'},
        'params_axis_resources': [],
    })
    mesh = restore.create_mesh(1)
    flax_module, _, _, _, router_keys, rng_keys = restore.restore_from_config(
        config, '/foo/bar', (1, 16, 16, 3), mesh)
    self.assertIsInstance(flax_module, restore.vmoe.nn.models.VisionTransformer)
    self.assertEmpty(router_keys)
    self.assertEqual(rng_keys, ())


if __name__ == '__main__':
  absltest.main()
