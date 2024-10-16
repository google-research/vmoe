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

"""PeriodicActions used during training."""
import dataclasses
from typing import Any, Dict, Optional

from clu import periodic_actions
import flax.traverse_util
import jax

try:
  import memory_profiler  # pylint: disable=g-import-not-at-top
except ImportError:
  memory_profiler = None

PeriodicAction = periodic_actions.PeriodicAction


@dataclasses.dataclass
class SingleProcessPeriodicAction:
  """Runs a given PeriodicAction on a single process."""
  periodic_action: PeriodicAction
  process_index: int = 0

  def __call__(self, *args, **kwargs) -> bool:
    if jax.process_index() == self.process_index:
      return self.periodic_action(*args, **kwargs)
    return False


class ReportProgress(periodic_actions.ReportProgress):
  """Reports training progress, including metrics."""

  def __init__(self, *, process_index: Optional[int] = 0, **kwargs):
    self._process_index = process_index
    super().__init__(**kwargs)

  def __call__(self, step: int, t: Optional[float] = None, **kwargs) -> bool:
    if super().__call__(step, t):
      self._apply_extra(step, t, **kwargs)
      return True
    return False

  def _should_trigger(self, step: int, t: float) -> bool:
    if (self._process_index is not None and
        jax.process_index() != self._process_index):
      return False
    return super()._should_trigger(step, t)

  def _apply_extra(
      self, step: int, t: float, scalar_metrics: Optional[Dict[str, Any]] = None
  ):
    if self._writer is not None:
      scalar_metrics = scalar_metrics or {}
      scalar_metrics = {
          '/'.join(k): float(v)
          for k, v in flax.traverse_util.flatten_dict(scalar_metrics).items()
      }
      if memory_profiler:
        memory_usage_mb = memory_profiler.memory_usage(max_usage=True)
        scalar_metrics['host_memory_mb'] = memory_usage_mb
      for key, value in self._time_per_part.items():
        scalar_metrics[f'uptime_{key}'] = value
      self._writer.write_scalars(step, scalar_metrics)
