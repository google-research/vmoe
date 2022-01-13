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

"""Tests for ensemble."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from vmoe.evaluate import ensemble


class EnsembleTest(parameterized.TestCase):

  @parameterized.parameters(1, 3, 5)
  def test_ensemble_softmax_xent(self, ensemble_size):
    batch_size, num_classes = 16, 3
    logits_3d_shape = (ensemble_size, batch_size, num_classes)
    logits_2d_shape = (ensemble_size * batch_size, num_classes)

    key = jax.random.PRNGKey(666)
    logits_3d = jax.random.normal(key, logits_3d_shape)
    # The ensemble softmax CE assumes the logits have a structure corresponding
    # to jnp.repeat(..., ensemble_size).
    logits_2d_with_repeat = np.zeros(logits_2d_shape)
    for e in range(ensemble_size):
      logits_2d_with_repeat[e::ensemble_size] = logits_3d[e]
    logits_2d_with_repeat = jnp.asarray(logits_2d_with_repeat)

    labels = jnp.asarray([i % num_classes for i in range(batch_size)])
    one_hot_labels = jax.nn.one_hot(labels, num_classes)

    xent = ensemble.ensemble_softmax_xent(logits_2d_with_repeat,
                                          one_hot_labels, ensemble_size)

    ensemble_softmax = jnp.mean(jax.nn.softmax(logits_3d), 0)
    expected_xent = jnp.asarray([
        -jnp.log(ensemble_softmax[i, labels[i]]) for i in range(batch_size)
    ])

    self.assertSequenceAlmostEqual(list(expected_xent), list(xent), places=5)

  @parameterized.parameters(1, 3, 5)
  def test_label_pred_ensemble_softmax(self, ensemble_size):
    batch_size, num_classes = 16, 8
    logits_3d_shape = (ensemble_size, batch_size, num_classes)
    logits_2d_shape = (ensemble_size * batch_size, num_classes)

    key = jax.random.PRNGKey(666)
    logits_3d = jax.random.normal(key, logits_3d_shape)
    # The ensemble softmax CE assumes the logits have a structure corresponding
    # to jnp.repeat(..., ensemble_size).
    logits_2d_with_repeat = np.zeros(logits_2d_shape)
    for e in range(ensemble_size):
      logits_2d_with_repeat[e::ensemble_size] = logits_3d[e]
    logits_2d_with_repeat = jnp.asarray(logits_2d_with_repeat)

    labels = ensemble.label_pred_ensemble_softmax(logits_2d_with_repeat,
                                                  ensemble_size)

    ensemble_softmax = jnp.mean(jax.nn.softmax(logits_3d), 0)
    expected_labels = jnp.argmax(ensemble_softmax, 1)

    self.assertSequenceEqual(list(labels), list(expected_labels))

if __name__ == '__main__':
  absltest.main()
