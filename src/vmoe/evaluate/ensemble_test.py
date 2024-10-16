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

"""Tests for ensemble."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from vmoe.evaluate import ensemble


class EnsembleTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('ensemble_size_1', 1, [1, 0, 1, 0]),
      ('ensemble_size_2', 2, [1, 1]),
      ('ensemble_size_4', 4, [1]),
  )
  def test_label_pred_ensemble_softmax(self, ensemble_size, expected_labels):
    p = jnp.asarray([[.1, .9], [.8, .2], [.3, .7], [.6, .3]])
    logits = jnp.log(p)
    labels = ensemble.label_pred_ensemble_softmax(logits, ensemble_size)
    np.testing.assert_allclose(labels, expected_labels)

  @parameterized.named_parameters(
      ('ensemble_size_1', 1, [0., 1., 0., 1.],
       [-np.log(.1), -np.log(.8), -np.log(.3), -np.log(.6)]),
      ('ensemble_size_2', 2, [0., 1.],
       [-np.log(.1), -np.log(.2), -np.log(.7), -np.log(.6)]),
      ('ensemble_size_4', 4, [1.],
       [-np.log(.9), -np.log(.8), -np.log(.7), -np.log(.6)]),
  )
  def test_ensemble_softmax_xent_train(self, ensemble_size, labels,
                                       expected_loss):
    p = jnp.asarray([[.1, .9], [.2, .8], [.3, .7], [.4, .6]])
    logits = jnp.log(p)
    labels = jax.nn.one_hot(labels, num_classes=2)
    loss = ensemble.ensemble_softmax_xent_train(logits, labels, ensemble_size)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)

  @parameterized.named_parameters(
      ('ensemble_size_1', 1, [0., 1., 0., 1.],
       [-np.log(.1), -np.log(.8), -np.log(.3), -np.log(.6)]),
      ('ensemble_size_2', 2, [0., 1.], [-np.log(.15), -np.log(.65)]),
      ('ensemble_size_4', 4, [1.], [-np.log((.9 + .8 + .7 + .6) / 4)]),
  )
  def test_ensemble_softmax_xent_eval(self, ensemble_size, labels,
                                      expected_loss):
    p = jnp.asarray([[.1, .9], [.2, .8], [.3, .7], [.4, .6]])
    logits = jnp.log(p)
    labels = jax.nn.one_hot(labels, num_classes=2)
    loss = ensemble.ensemble_softmax_xent_eval(logits, labels, ensemble_size)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)

  @parameterized.named_parameters(
      ('ensemble_size_1', 1, [[0.], [1.], [0.], [1.]],
       [-np.log1p(-.1), -np.log(.2), -np.log1p(-.3), -np.log(.4)]),
      ('ensemble_size_2', 2, [[0.], [1.]],
       [-np.log1p(-.1), -np.log1p(-.2), -np.log(.3), -np.log(.4)]),
      ('ensemble_size_4', 4, [[1.]],
       [-np.log(.1), -np.log(.2), -np.log(.3), -np.log(.4)]),
  )
  def test_ensemble_sigmoid_xent_train(self, ensemble_size, labels,
                                       expected_loss):
    p = jnp.asarray([[.1], [.2], [.3], [.4]])
    logits = jnp.log(p) - jnp.log1p(-p)
    labels = jnp.asarray(labels)
    loss = ensemble.ensemble_sigmoid_xent_train(logits, labels, ensemble_size)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)

  @parameterized.named_parameters(
      ('ensemble_size_1', 1, [[0.], [1.], [0.], [1.]],
       [-np.log1p(-.1), -np.log(.2), -np.log1p(-.3), -np.log(.4)]),
      ('ensemble_size_2', 2, [[0.], [1.]], [-np.log1p(-.15), -np.log(.35)]),
      ('ensemble_size_4', 4, [[1.]], [-np.log(.25)]),
  )
  def test_ensemble_sigmoid_xent_eval(self, ensemble_size, labels,
                                      expected_loss):
    p = jnp.asarray([[.1], [.2], [.3], [.4]])
    logits = jnp.log(p) - jnp.log1p(-p)
    labels = jnp.asarray(labels)
    loss = ensemble.ensemble_sigmoid_xent_eval(logits, labels, ensemble_size)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
