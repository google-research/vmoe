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
import numpy as np
import optax
from vmoe.train import train_state


class TrainStateTreeAxisResourcesTest(absltest.TestCase):

  def test_apply_gradients_and_compute_global_norms(self):
    state = train_state.TrainState.create(
        apply_fn=lambda x: x,
        params={'w': np.asarray((1.,), dtype=np.float32)},
        tx=optax.sgd(0.5),
        rngs={})
    grads = {'w': np.asarray((1.,), dtype=np.float32)}
    new_state, global_norms = state.apply_gradients_and_compute_global_norms(
        grads, rngs={})
    chex.assert_trees_all_close(new_state.params,
                                {'w': np.asarray((.5,), dtype=np.float32)})
    chex.assert_trees_all_close(global_norms, {
        'grads': np.asarray((1.,), dtype=np.float32),
        'params': np.asarray((.5,), dtype=np.float32),
        'updates': np.asarray((.5,), dtype=np.float32),
    })


if __name__ == '__main__':
  absltest.main()
