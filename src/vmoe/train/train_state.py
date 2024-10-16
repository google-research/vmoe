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

"""TrainState and other related classes."""
from typing import Any, Callable, Dict, Mapping, Tuple

import flax.training.train_state
import jax
import optax

PRNGKey = jax.Array


class TrainState(flax.training.train_state.TrainState):
  """Extension of FLAX's TrainState."""
  rngs: Dict[str, PRNGKey]

  def apply_gradients_and_compute_global_norms(self, grads, **kwargs):
    updates, new_opt_state = self.tx.update(
        grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)
    state = self.replace(
        step=self.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        **kwargs,
    )
    global_norms = {
        'grads': optax.global_norm(grads),
        'updates': optax.global_norm(updates),
        'params': optax.global_norm(state.params),
    }
    return state, global_norms


# TrainStateAxisResources is a PyTree with the same structure as TrainState but
# whose leaves are PartitionSpec objects.
TrainStateAxisResources = TrainState
TrainStepFn = Callable[..., 'TrainStepResult']
TrainStepResult = Tuple[TrainState, Mapping[str, Any]]
