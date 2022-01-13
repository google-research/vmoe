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

"""TrainState and other related classes."""
from typing import Dict, Union

import flax.training.train_state
import jax

PRNGKey = Union[jax.numpy.ndarray, jax.random.KeyArray]


class TrainState(flax.training.train_state.TrainState):
  rngs: Dict[str, PRNGKey]


# TrainStateAxisResources is a PyTree with the same structure as TrainState but
# whose leaves are PartitionSpec objects.
TrainStateAxisResources = TrainState
