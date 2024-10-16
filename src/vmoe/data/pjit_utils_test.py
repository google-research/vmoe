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

"""Tests for pjit_utils."""
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental import pjit
import numpy as np
from vmoe.data import pjit_utils


def assert_trees_all_equivalent_sharding(*trees):

  def cmp_fn(a, b):
    device_map_a = a.sharding.devices_indices_map(a.shape)
    device_map_b = b.sharding.devices_indices_map(b.shape)
    return device_map_a == device_map_b

  def err_msg_fn(a, b):
    return f'{a.sharding=!r} is not equivalent to {b.sharding=!r}'

  chex.assert_trees_all_equal_comparator(cmp_fn, err_msg_fn, *trees)


class PrefetchToDevice(parameterized.TestCase):

  def test(self):
    devices = np.asarray(jax.local_devices())
    if devices.size > 1:
      np.random.shuffle(devices)
      devices = devices[:devices.size - devices.size % 2]
      devices = devices.reshape((-1, 2))
    else:
      devices = devices.reshape((1, 1))
    # Generate random test data.
    x = np.random.normal(size=(256, 32))
    axis_resources = jax.sharding.PartitionSpec(('a', 'b'))
    with jax.sharding.Mesh(devices, ('a', 'b')):
      # Transfer data to device using explicit call to pjit, wrapping an
      # identity function. This is the expected ShardedDeviceArray.
      expected = pjit.pjit(
          fun=lambda x: x,
          in_shardings=(axis_resources,),
          out_shardings=axis_resources,
      )(x)
      # Prefetch data using `prefetch_to_device` and check that the result is
      # the same as explicitly calling pjit with an identity function.
      y = list(pjit_utils.prefetch_to_device(iter([x]), size=1))
      self.assertLen(y, 1)
      chex.assert_trees_all_close(expected, y[0])
      assert_trees_all_equivalent_sharding(expected, y[0])

  @parameterized.named_parameters(('zero', 0), ('negative', -1))
  def test_size(self, size):
    """Tests that the original objects are iterated if size is <= 0."""
    with jax.sharding.Mesh(np.asarray(jax.devices()), ('d',)):
      objects = [np.ones(16), 1 * np.ones(16), 3 * np.ones(16)]
      new_objects = list(pjit_utils.prefetch_to_device(iter(objects), size))
      chex.assert_trees_all_equal(objects, new_objects)


if __name__ == '__main__':
  absltest.main()
