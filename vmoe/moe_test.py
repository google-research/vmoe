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

"""Tests for moe."""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import flax.core
import flax.linen as nn
import jax
import jax.experimental.maps as maps
import jax.experimental.pjit as pjit
import jax.numpy as jnp
import numpy as np
from vmoe import moe


class DispatcherTest(parameterized.TestCase):
  """Tests for the dispatcher classes.

  This only tests that the output of the dispatch() and combine() methods are
  correct. It does not test that the used communication primitives are optimal.
  """

  def _run_dispatch(self, dispatcher):  # pylint: disable=g-unreachable-test-method
    # (G=2, S=3, D=4)
    data = jnp.asarray([
        [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
        [[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]],
    ], jnp.float32)
    output = dispatcher.dispatch(data)
    # (E=4, GC=4, D=4).
    expected = jnp.asarray([
        [[2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]],
        [[1, 1, 1, 1], [3, 3, 3, 3], [6, 6, 6, 6], [4, 4, 4, 4]],
    ], dtype=jnp.float32)
    np.testing.assert_array_almost_equal(output, expected)

  def _run_combine(self, dispatcher):  # pylint: disable=g-unreachable-test-method
    # (E=4, GC=4, D=4).
    data = jnp.asarray([
        [[2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]],
        [[1, 1, 1, 1], [3, 3, 3, 3], [6, 6, 6, 6], [4, 4, 4, 4]],
    ], dtype=jnp.float32)
    output = dispatcher.combine(data)
    expected = jnp.asarray([
        [[1 * .7, 1 * .7, 1 * .7, 1 * .7],
         [2 * .7, 2 * .7, 2 * .7, 2 * .7],
         [3 * 1., 3 * 1., 3 * 1., 3 * 1.]],
        [[4 * 1., 4 * 1., 4 * 1., 4 * 1.],
         [5 * .7, 5 * .7, 5 * .7, 5 * .7],
         [6 * .3, 6 * .3, 6 * .3, 6 * .3]],
    ], dtype=jnp.float32)
    np.testing.assert_array_almost_equal(output, expected)

  @parameterized.named_parameters(
      ('none', None, None),
      ('str', 'a', moe.PartitionSpec('a')),
      ('tuple_1', ('a', 'b'), moe.PartitionSpec('a', 'b')),
      ('tuple_2', (('a', 'b'), 'c'), moe.PartitionSpec(('a', 'b'), 'c')),
  )
  def test_convert_partition_spec(self, partition_spec, expected):
    self.assertEqual(moe._convert_partition_spec(partition_spec), expected)

  @parameterized.named_parameters(
      ('dispatch', '_run_dispatch'),
      ('combine', '_run_combine'),
  )
  def test_einsum(self, run_method):
    # (G=2, S=3, E=2, C=2).
    combine_weights = jnp.asarray([
        [[[.0, .0], [.7, .0]],
         [[.7, .0], [.0, .0]],
         [[.0, .3], [.0, .7]]],
        [[[.7, .0], [.0, .3]],
         [[.0, .7], [.0, .0]],
         [[.0, .0], [.3, .0]]],
    ], dtype=jnp.float32)
    dispatcher = moe.EinsumDispatcher(
        combine_weights=combine_weights,
        einsum_precision=jax.lax.Precision.HIGHEST)
    getattr(self, run_method)(dispatcher)

  @parameterized.named_parameters(
      ('dispatch', '_run_dispatch'),
      ('combine', '_run_combine'),
  )
  def test_indices(self, run_method):
    # (G=2, S=3, K=2, 2). Note: 9's are meaningless since E = 2, C = 2.
    indices = jnp.asarray([
        [[[1, 0], [9, 9]], [[0, 0], [9, 9]], [[1, 1], [0, 1]]],
        [[[0, 0], [1, 1]], [[0, 1], [9, 9]], [[9, 9], [1, 0]]],
    ], dtype=jnp.int32)
    # (G=2, S=3, K=2). Each item uses weight 0.7 for the top-1 and 0.3 for the
    # top-2 expert.
    combine_weights = jnp.ones((2, 3, 2), dtype=jnp.float32)
    combine_weights = combine_weights * jnp.asarray([[[.7, .3]]])
    dispatcher = moe.ExpertIndicesDispatcher(
        indices=indices,
        combine_weights=combine_weights,
        num_experts=2,
        capacity=2,
        einsum_precision=jax.lax.Precision.HIGHEST)
    getattr(self, run_method)(dispatcher)

  def test_bfloat16_dispatcher(self):
    # We mock a base dispatcher that simply check that the type of the passed
    # argument is a bfloat16 array.
    def assert_bfloat16(x):
      self.assertEqual(x.dtype, jnp.bfloat16)
      return x
    base_dispatcher = mock.create_autospec(moe.BaseDispatcher, instance=True)
    base_dispatcher.dispatch.side_effect = assert_bfloat16
    base_dispatcher.combine.side_effect = assert_bfloat16
    # Create a bfloat16 dispatcher wrapping the mocked one.
    bfloat16_dispatcher = moe.Bfloat16Dispatcher(dispatcher=base_dispatcher)
    # Check that types for dispatch and combine are correctly converted back
    # to the original type.
    x = jnp.zeros((5, 4, 3), dtype=jnp.float32)
    self.assertEqual(bfloat16_dispatcher.dispatch(x).dtype, jnp.float32)
    self.assertEqual(bfloat16_dispatcher.combine(x).dtype, jnp.float32)

  # The following tests represent the same scenario. There are three items in
  # a group, and three experts, each with capacity = 2.
  #
  # For the first two items, their top-1 choice (expert_idx = 0) is always
  # selected but the buffer_idx depends on whether Vanilla / BPR are used.
  # Vanilla / BPR also determines which of the top-2 choices can be made.
  #
  # For the third item, the top-1 choice (expert_idx = 2) is always selected and
  # the item takes the buffer idx = 0. The top-2 choice (expert_idx = 0) can
  # never be selected.
  @parameterized.named_parameters(
      ('vanilla', False, [[[.4, .0], [.0, .0], [.0, .2]],
                          [[.0, .5], [.0, .0], [.0, .0]],
                          [[.0, .0], [.0, .0], [.6, .0]]]),
      ('bpr', True, [[[.0, .4], [.0, .0], [.0, .0]],
                     [[.5, .0], [.0, .0], [.0, .4]],
                     [[.0, .0], [.0, .0], [.6, .0]]]),
  )
  def test_top_experts_per_item_einsum(self, batch_priority, expected):
    gates = jnp.asarray([[.4, .0, .2], [.5, .1, .4], [.3, .2, .6]])
    dispatcher = moe._get_top_experts_per_item_einsum_dispatcher(
        gates,
        num_selected_experts=2,
        capacity=2,
        batch_priority=batch_priority)
    np.testing.assert_array_almost_equal(dispatcher.combine_weights, expected)

  @parameterized.named_parameters(
      ('vanilla', False,
       [[(0, 0), (2, 1)], [(0, 1), (2, 2)], [(2, 0), (0, 2)]]),
      ('bpr', True,
       [[(0, 1), (2, 2)], [(0, 0), (2, 1)], [(2, 0), (0, 2)]]),
  )
  def test_top_experts_per_item_expert_indices(self, batch_priority,
                                               expected_indices,
                                               ):
    gates = jnp.asarray([[.4, .0, .2], [.5, .1, .4], [.3, .2, .6]])
    dispatcher = moe._get_top_experts_per_item_expert_indices_dispatcher(
        gates,
        num_selected_experts=2,
        capacity=2,
        batch_priority=batch_priority)
    self.assertEqual(dispatcher.num_experts, 3)
    self.assertEqual(dispatcher.capacity, 2)
    np.testing.assert_array_equal(dispatcher.indices, expected_indices)
    expected_combine_weights = jnp.asarray([[.4, .2], [.5, .4], [.6, .3]])
    np.testing.assert_array_almost_equal(dispatcher.combine_weights,
                                         expected_combine_weights)

  @parameterized.named_parameters(
      ('einsum', 'einsum', moe.EinsumDispatcher),
      ('indices', 'indices', moe.ExpertIndicesDispatcher),
  )
  def test_get_top_experts_per_item_dispatcher(self, name, expected_class):
    gates = jnp.zeros((4, 32))
    dispatcher = moe.get_top_experts_per_item_dispatcher(
        gates, name, num_selected_experts=2, capacity=2, batch_priority=False)
    self.assertIsInstance(dispatcher, expected_class)

  def test_get_to_experts_per_item_dispatcher_unknown(self):
    gates = jnp.zeros((4, 32))
    with self.assertRaisesRegex(ValueError, 'Unknown dispatcher type'):
      moe.get_top_experts_per_item_dispatcher(
          gates, name='foo', num_selected_experts=2, capacity=2,
          batch_priority=False)


class DummyExpert(nn.Module):
  """Expert with a single scalar parameter that is added to the input."""

  @nn.compact
  def __call__(self, x):
    p = self.param('p', nn.initializers.normal(), ())
    return x + p


class SparseMoeSpmdLayerTest(parameterized.TestCase):

  @classmethod
  def _generate_data(cls, batch_size, group_size, partition_spec):
    err_msg = f'group_size must divide batch_size ({group_size} vs. {batch_size})'
    assert batch_size % group_size == 0, err_msg
    # Creates batch_size items, each with a single dimension = 1.
    x = 10 * jnp.arange(1, 1 + batch_size, dtype=jnp.float32)
    num_groups = batch_size // group_size
    x = x.reshape(num_groups, group_size, 1)
    x = moe.with_sharding_constraint(x, partition_spec)
    return x

  @classmethod
  def _generate_dispatcher_einsum(cls, batch_size, group_size, num_experts,
                                  capacity, partition_spec):
    num_groups = batch_size // group_size
    # Each item will be dispatched to the (i % num_experts)-th and the
    # ((i + 1) % num_experts)-th expert.
    w = jax.nn.one_hot(
        x=jnp.arange(2 * batch_size, dtype=jnp.int32) % num_experts,
        num_classes=num_experts,
        dtype=jnp.int32)
    w = w.reshape(num_groups, group_size, 2, num_experts)
    w = w.transpose(0, 2, 1, 3).reshape(-1, group_size * 2, num_experts)
    w = moe.with_sharding_constraint(w, partition_spec)
    w = jax.nn.one_hot(
        x=jnp.cumsum(w, axis=1) * w - 1,
        num_classes=capacity,
        dtype=jnp.float32)
    w = w.reshape(num_groups, 2, group_size, num_experts, capacity)
    # First expert with weight 0.7, second expert with weight 0.3
    w = w * jnp.asarray([0.7, 0.3])[None, :, None, None, None]
    # Reduce shape to (G, S, E, C).
    w = jnp.max(w, axis=1)
    dispatcher = moe.EinsumDispatcher(
        combine_weights=w,
        partition_spec=partition_spec,
        einsum_precision=jax.lax.Precision.HIGHEST)
    return dispatcher

  @classmethod
  def _generate_dispatcher_indices(cls, batch_size, group_size, num_experts,
                                   capacity, partition_spec):
    num_groups = batch_size // group_size
    # Compute expert indices for each example (K=2).
    expert_idx = jnp.arange(2 * batch_size, dtype=jnp.int32) % num_experts
    expert_idx = expert_idx.reshape(num_groups, group_size, 2)
    expert_idx = moe.with_sharding_constraint(expert_idx, partition_spec)
    # Compute buffer indices for each example (K=2).
    expert_idx = expert_idx.transpose(0, 2, 1).reshape(-1, 2 * group_size)
    expert_oh = jax.nn.one_hot(expert_idx, num_experts, dtype=jnp.int32)
    buffer_idx = jnp.max(jnp.cumsum(expert_oh, axis=1) * expert_oh - 1, axis=2)
    expert_idx = expert_idx.reshape(-1, 2, group_size).transpose(0, 2, 1)
    buffer_idx = buffer_idx.reshape(-1, 2, group_size).transpose(0, 2, 1)
    buffer_idx = moe.with_sharding_constraint(buffer_idx, partition_spec)
    # Compute combine weights for each example (K=2).
    combine_weights = jnp.ones((num_groups, group_size, 2), dtype=jnp.float32)
    combine_weights = combine_weights * jnp.asarray([[[.7, .3]]])
    combine_weights = moe.with_sharding_constraint(combine_weights,
                                                   partition_spec)
    dispatcher = moe.ExpertIndicesDispatcher(
        indices=jnp.stack([expert_idx, buffer_idx], axis=-1),
        combine_weights=combine_weights,
        num_experts=num_experts,
        capacity=capacity,
        partition_spec=partition_spec,
        einsum_precision=jax.lax.Precision.HIGHEST)
    return dispatcher

  @parameterized.named_parameters(
      ('einsum_false', '_generate_dispatcher_einsum', False),
      ('einsum_true', '_generate_dispatcher_einsum', True),
      ('indices_false', '_generate_dispatcher_indices', False),
      ('indices_true', '_generate_dispatcher_indices', True),
  )
  def test_initialize_split_rngs(self, generate_dispatcher_fn, split_rngs):
    num_devices = 4
    if jax.device_count() < num_devices:
      self.skipTest(f"Test can't run because it needs {num_devices} devices, "
                    f"but only {jax.device_count()} were found.")
    variables = self._run_initialize(
        batch_size=32,
        group_size=8,
        capacity=2,
        num_experts=4,
        num_devices=num_devices,
        split_rngs=split_rngs,
        generate_dispatcher_fn=generate_dispatcher_fn)
    p = variables['params']['p']
    # There are four experts, each expert has only 1 scalar parameter, thus the
    # shape of 'p' must be (4,).
    self.assertTupleEqual(p.shape, (4,))
    # If split_rngs=True, each expert is initialized using its own rng.
    # If split_rngs=False, all experts are initialized using the same rng.
    self.assertLen(set([hash(np.asarray(row).tobytes()) for row in p]),
                   4 if split_rngs else 1)

  @parameterized.named_parameters(
      ('einsum_devices_equal_to_experts',
       '_generate_dispatcher_einsum', 32, 4, 2, 4, 4),
      ('einsum_devices_greater_than_experts',
       '_generate_dispatcher_einsum', 32, 4, 2, 4, 8),
      ('einsum_devices_less_than_experts',
       '_generate_dispatcher_einsum', 32, 4, 2, 4, 2),
      ('indices_devices_equal_to_experts',
       '_generate_dispatcher_indices', 32, 4, 2, 4, 4),
      ('indices_devices_greater_than_experts',
       '_generate_dispatcher_indices', 32, 4, 2, 4, 8),
      ('indices_devices_less_than_experts',
       '_generate_dispatcher_indices', 32, 4, 2, 4, 2),
  )
  def test_apply(self, generate_dispatcher_fn, batch_size, group_size, capacity,
                 num_experts, num_devices):
    if jax.device_count() < num_devices:
      self.skipTest(f"Test can't run because it needs {num_devices} devices, "
                    f"but only {jax.device_count()} were found.")
    output = self._run_apply(batch_size, group_size, capacity, num_experts,
                             num_devices, generate_dispatcher_fn)
    # Construct array with the expected outputs and check that it matches with
    # the actual output. In all test cases, all items must be dispatched to
    # their target experts.
    b = (1 + np.arange(2 * batch_size) % num_experts).reshape(batch_size, 2)
    b = (b * np.asarray([[0.7, 0.3]])).sum(axis=1)
    expected_output = 10 * (1 + np.arange(batch_size)) + b
    expected_output = expected_output.reshape(-1, group_size, 1)
    np.testing.assert_array_almost_equal(output, expected_output, decimal=5)

  def _run_initialize(self, batch_size, group_size, capacity, num_experts,
                      num_devices, split_rngs, generate_dispatcher_fn):
    # The first axis of the data (e.g. batch_size) is split across two axes in
    # the device mesh.
    data_partition_spec = pjit.PartitionSpec(('expert', 'replica'))

    def init(rng):
      moe_layer = moe.sparse_moe_spmd(DummyExpert, split_rngs=split_rngs)()
      x = self._generate_data(batch_size, group_size, data_partition_spec)
      dispatcher = getattr(self, generate_dispatcher_fn)(
          batch_size, group_size, num_experts, capacity, data_partition_spec)
      return moe_layer.init(rng, dispatcher, x)

    # All parameters are partitioned across the first axis of the TPU mesh.
    # Thus, each group of devices in one "row" will share the same values for
    # all parameters.
    param_partition_spec = jax.tree_map(
        lambda _: pjit.PartitionSpec(('expert',)),
        jax.eval_shape(init, jax.random.PRNGKey(0)))
    init_pjit = pjit.pjit(
        init, in_axis_resources=None, out_axis_resources=param_partition_spec)

    # Create a devices mesh with the specified number of devices and run the
    # init function on that mesh.
    devices = np.asarray(jax.devices()[:num_devices])
    devices = devices.reshape(num_experts, num_devices // num_experts)
    with maps.mesh(devices, ('expert', 'replica')):
      return init_pjit(jax.random.PRNGKey(0))

  def _run_apply(self, batch_size, group_size, capacity, num_experts,
                 num_devices, generate_dispatcher_fn):
    # The first axis of the data (e.g. batch_size) is split across two axes in
    # the device mesh.
    data_partition_spec = pjit.PartitionSpec(('expert', 'replica'))

    def apply(variables):
      moe_layer = moe.sparse_moe_spmd(DummyExpert)()
      x = self._generate_data(batch_size, group_size, data_partition_spec)
      dispatcher = getattr(self, generate_dispatcher_fn)(
          batch_size, group_size, num_experts, capacity, data_partition_spec)
      return moe_layer.apply(variables, dispatcher, x)

    # All parameters are partitioned across the first axis of the TPU mesh.
    # Thus, each group of devices in one "row" will share the same values for
    # all parameters.
    param_partition_spec = flax.core.freeze(
        {'params': {'p': pjit.PartitionSpec(('expert',))}})
    variables = flax.core.freeze({'params': {'p': 1 + np.arange(num_experts)}})
    apply_pjit = pjit.pjit(
        apply,
        in_axis_resources=(param_partition_spec,),
        out_axis_resources=data_partition_spec)

    # Create a devices mesh with the specified number of devices and run the
    # apply function on that mesh.
    devices = np.asarray(jax.devices()[:num_devices])
    if num_devices >= num_experts:
      devices = devices.reshape(num_experts, num_devices // num_experts)
    else:
      devices = devices.reshape(num_devices, 1)
    with maps.mesh(devices, ('expert', 'replica')):
      return apply_pjit(variables)


if __name__ == '__main__':
  absltest.main()
