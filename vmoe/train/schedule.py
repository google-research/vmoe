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

"""Learning rate schedules."""
import optax


def create_learning_rate_schedule(*,
                                  schedule: str,
                                  total_steps: int,
                                  **kwargs) -> optax.Schedule:
  """Creates a optax learning rate schedule."""
  if schedule == 'constant':
    return optax.constant_schedule(**kwargs)
  if schedule == 'warmup_exponential_decay':
    return optax.warmup_exponential_decay_schedule(init_value=0.0, **kwargs)
  # The following schedules support decay_steps, set its default value.
  kwargs['decay_steps'] = kwargs.pop('decay_steps', total_steps)
  if schedule == 'warmup_cosine_decay':
    return optax.warmup_cosine_decay_schedule(init_value=0.0, **kwargs)
  if schedule == 'warmup_linear_decay':
    return warmup_polynomial_decay_schedule(power=1.0, **kwargs)
  if schedule == 'warmup_polynomial_decay':
    return warmup_polynomial_decay_schedule(**kwargs)
  # Unknown learning rate schedule.
  raise ValueError(f'Unknown learning rate schedule: {schedule!r}')


def warmup_polynomial_decay_schedule(peak_value: float, end_value: float,
                                     power: float, warmup_steps: int,
                                     decay_steps: int) -> optax.Schedule:
  """Linear warmup followed by polynomial decay."""
  return optax.join_schedules([
      optax.linear_schedule(
          init_value=0.0, end_value=peak_value, transition_steps=warmup_steps),
      optax.polynomial_schedule(
          init_value=peak_value,
          end_value=end_value,
          power=power,
          transition_steps=decay_steps - warmup_steps)
  ], [warmup_steps])
