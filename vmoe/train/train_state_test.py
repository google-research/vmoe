# Copyright 2022 Google LLC.
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

"""Tests for train_state."""

from absl.testing import absltest
from vmoe import partitioning
from vmoe.train import train_state


PartitionSpec = partitioning.PartitionSpec


class TrainStateTreeAxisResourcesTest(absltest.TestCase):

  def test_train_state(self):
    params = {'a': 1, 'b': 2, 'c': 3}
    rngs = {'dropout': None}
    state = train_state.TrainState.create(
        apply_fn=lambda x: x,
        params=params,
        tx=lambda x: x,
        rngs=rngs)
    output = partitioning.tree_axis_resources_from_regexes(
        tree=state, axis_resources_regexes=[
            ('.*/a$', ('expert',)),
            ('.*/c$', (('expert', 'width'),)),
        ])
    self.assertIsInstance(output, state.TrainState)
    self.assertEqual(output.params['a'], PartitionSpec('expert'))
    self.assertEqual(output.params['b'], PartitionSpec())
    self.assertEqual(output.params['c'], PartitionSpec(('expert', 'width')))


if __name__ == '__main__':
  absltest.main()
