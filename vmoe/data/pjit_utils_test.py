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

"""Tests for pjit_utils."""
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental import maps
from jax.experimental import pjit
import numpy as np
from vmoe.data import pjit_utils


class PrefetchToDevice(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_array_first_mesh_both', (('a', 'b'),)),
      ('_array_first_mesh_first_array_second_mesh_second', ('a', 'b')),
  )
  def test(self, axis_resources):
    devices = np.asarray(jax.local_devices())
    if devices.size > 1:
      np.random.shuffle(devices)
      devices = devices[:devices.size - devices.size % 2]
      devices = devices.reshape((-1, 2))
    else:
      devices = devices.reshape((1, 1))
    axis_names = ('a', 'b')
    # Generate random test data.
    x = np.random.normal(size=(256, 32))
    axis_resources = pjit.PartitionSpec(*axis_resources)
    with maps.mesh(devices, axis_names):
      # Transfer data to device using explicit call to pjit, wrapping an
      # identity function. This is the expected ShardedDeviceArray.
      expected = pjit.pjit(
          fun=lambda x: x,
          in_axis_resources=(axis_resources,),
          out_axis_resources=axis_resources)(x)
      # Prefetch data using `prefetch_to_device` and check that the result is
      # the same as explicitly calling pjit with an identity function.
      y = list(pjit_utils.prefetch_to_device(iter([x]), axis_resources, size=1))
      self.assertLen(y, 1)
      # Check that the two ShardedDeviceArrays are the same (not only the values
      # but how they are sharded).
      chex.assert_trees_all_close(expected.device_buffers, y[0].device_buffers)
      self.assertEqual([b.device() for b in expected.device_buffers],
                       [b.device() for b in y[0].device_buffers])
      self.assertEqual(expected.sharding_spec, y[0].sharding_spec)
      self.assertEqual(expected.indices, y[0].indices)

  @parameterized.named_parameters(('zero', 0), ('negative', -1))
  def test_size(self, size):
    """Tests that the original objects are iterated if size is <= 0."""
    objects = [object(), object(), object()]
    new_objects = list(pjit_utils.prefetch_to_device(
        iter(objects), axis_resources=None, size=size))
    self.assertEqual(objects, new_objects)


if __name__ == '__main__':
  absltest.main()
