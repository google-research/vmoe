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

"""Utils for multihost synchronization."""
import functools
import zlib

import jax
import jax.experimental.maps
import jax.experimental.pjit
import numpy as np


@functools.partial(
    jax.experimental.pjit.pjit,
    in_axis_resources=jax.experimental.pjit.PartitionSpec('d'),
    out_axis_resources=None,
)
def _sync_devices_sum(x):
  return jax.numpy.sum(x)


def sync_devices(name: str, main_process: int = 0):
  """Creates a barrier across all hosts/devices."""
  # All devices will be initialized with the value 0, except the first device
  # of the `main_process`, which will be initiaized with the CRC32 of the
  # `name`.
  h = np.int32(zlib.crc32(name.encode()))
  x = np.zeros(jax.local_device_count(), dtype=np.int32)
  if jax.process_index() == main_process:
    x[0] = h
  # The values in all devices are summed. Thus, the result in all processes
  # should be 'h'.
  with jax.experimental.maps.mesh(
      devices=np.asarray(jax.devices()),
      axis_names=('d',)):
    x = _sync_devices_sum(x)
  if x != h:
    raise ValueError(
        f'sync_devices failed for {name!r}. Found value: {x}, expected: {h}')
