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

"""Tests for trainer."""
import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import clu.data
import flax.core
import flax.linen as nn
import jax
from jax.experimental import pjit
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import orbax.checkpoint
import tensorflow as tf
from vmoe.train import trainer


PartitionSpec = jax.sharding.PartitionSpec
Mesh = jax.sharding.Mesh
MetricWriter = trainer.metric_writers.MetricWriter
VALID_KEY = trainer.input_pipeline.VALID_KEY


class CreateFlaxModelTest(absltest.TestCase):

  @mock.patch.object(trainer.utils, 'parse_call')
  def test_success(self, mock_parse_class):
    foo_cls = mock.MagicMock()
    mock_parse_class.return_value = (foo_cls, (), {})
    config = ml_collections.ConfigDict({'name': 'foo', 'arg': 'bar'})
    _ = trainer.create_flax_model(config=config, deterministic=False)
    foo_cls.assert_called_once_with(deterministic=False, arg='bar')

  def test_missing_name(self):
    with self.assertRaisesRegex(KeyError,
                                'The model config must have a "name" field.'):
      trainer.create_flax_model(config={}, deterministic=False)


class CreateProfileHookTest(absltest.TestCase):

  def test_single_process(self):
    workdir = self.create_tempdir().full_path
    hook = trainer.create_profile_hook(workdir=workdir)
    self.assertIsInstance(hook, trainer.SingleProcessPeriodicAction)

  def test_all_processes(self):
    workdir = self.create_tempdir().full_path
    hook = trainer.create_profile_hook(workdir=workdir, all_processes=True)
    self.assertIsInstance(hook, trainer.periodic_actions.ProfileAllHosts)


class CreateOrReuseTrainStateTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_replicated', PartitionSpec()),
      ('_partitioned', PartitionSpec('a',)),
  )
  def test(self, partition_spec):
    """Tests the create_or_reuse_train_state function."""
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('a',))
    # This value will be reused when calling create_or_reuse_train_state.
    with mesh:
      bar = pjit.pjit(
          fun=lambda: jnp.arange(8, dtype=jnp.float32),
          in_shardings=(),
          out_shardings=partition_spec,
      )()
    # Get the ShapeDtypeStruct of all arrays in the TrainState.
    optimizer_tx = trainer.optimizer.optax.identity()
    def initialize_fn():
      params = {
          'foo': 1. * jnp.ones(32, dtype=jnp.float32),
          'bar': 2. * jnp.ones(16, dtype=jnp.float32),
      }
      return trainer.TrainState.create(
          apply_fn=lambda x: x, params=params, tx=optimizer_tx, rngs={})
    train_state = jax.eval_shape(initialize_fn)
    # By default, all arrays in the TrainState are replicated.
    sharding = jax.sharding.NamedSharding(mesh, PartitionSpec())
    train_state = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding),
        train_state)
    # Replace the value of the 'bar' variable with the one that we manually
    # created above.
    train_state = train_state.replace(params={
        'foo': jax.ShapeDtypeStruct(
            shape=train_state.params['foo'].shape,
            dtype=train_state.params['foo'].dtype,
            sharding=jax.sharding.NamedSharding(mesh, partition_spec)),
        'bar': bar})
    # Call create_or_reuse_train_state, which should create a new value for
    # 'foo' and keep the previous value for 'bar'.
    train_state = trainer.create_or_reuse_train_state(
        train_state=train_state, initialize_fn=initialize_fn, mesh=mesh)
    chex.assert_trees_all_equal(
        train_state.params,
        {'foo': np.ones((32,), dtype=np.float32), 'bar': bar})


class GetLossTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_softmax_xent', 'softmax_xent', {}, (8,)),
      ('_sigmoid_xent', 'sigmoid_xent', {}, (8,)),
  )
  def test(self, name, kwargs, expected_shape):
    train_loss_fn, eval_loss_fn, label_pred_fn = trainer.get_loss_fn(
        name, **kwargs)
    logits = jax.random.normal(jax.random.PRNGKey(0), (8, 16))
    labels = jax.random.uniform(jax.random.PRNGKey(0), (8, 16))

    train_loss = train_loss_fn(logits, labels)
    eval_loss = eval_loss_fn(logits, labels)

    self.assertEqual(train_loss.shape, expected_shape)
    self.assertEqual(eval_loss.shape, expected_shape)
    # The train and eval losses are the same in the standard case (no ensemble).
    self.assertSequenceEqual(list(train_loss), list(eval_loss))
    self.assertSequenceEqual(list(label_pred_fn(logits)),
                             list(jnp.argmax(logits, 1)))

  @parameterized.named_parameters(
      ('_ensemble1_softmax_xent', 1, (8,), (8,)),
      ('_ensemble4_softmax_xent', 4, (8,), (2,)),
  )
  def test_ensemble_softmax_xent(self, ensemble_size, train_expected_shape,
                                 eval_expected_shape):
    name = 'ensemble_softmax_xent'
    kwargs = {'ensemble_size': ensemble_size}
    train_loss_fn, eval_loss_fn, label_pred_fn = trainer.get_loss_fn(
        name, **kwargs)
    key = jax.random.PRNGKey(0)
    logits = jax.random.normal(key, (8, 16))
    # The repeating of the training labels happens in the train loss.
    train_labels = jax.random.uniform(key, (8 // ensemble_size, 16))
    eval_labels = jax.random.uniform(key, (8 // ensemble_size, 16))

    train_loss = train_loss_fn(logits, train_labels)
    eval_loss = eval_loss_fn(logits, eval_labels)

    self.assertEqual(train_loss.shape, train_expected_shape)
    self.assertEqual(eval_loss.shape, eval_expected_shape)
    self.assertEqual(label_pred_fn(logits).shape, eval_expected_shape)
    # When ensemble_size is 1, we are back in the standard case where the train
    # and eval losses are identical.
    if ensemble_size == 1:
      self.assertSequenceEqual(list(train_loss), list(eval_loss))


class GetTrainStepsAndEpochsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_train_epochs', {'train_epochs': 1}, (10, 1.0)),
      ('_train_steps', {'train_steps': 11}, (11, 1.1)),
  )
  def test(self, kwargs, expected):
    self.assertEqual(
        trainer.get_train_steps_and_epochs(
            **kwargs, train_batch_size=10, train_examples=100), expected)

  @parameterized.named_parameters(
      ('none', {}, 'You must specify'),
      ('both', {'train_epochs': 10, 'train_steps': 11},
       'You must specify .*, but not both:'),
  )
  def test_raises(self, kwargs, raises_regex):
    with self.assertRaisesRegex(ValueError, raises_regex):
      trainer.get_train_steps_and_epochs(
          **kwargs, train_batch_size=10, train_examples=100)


class InitializeTrainStateFromCheckpointTest(absltest.TestCase):
  """Tests that the appropriate initialization functions are called."""

  @mock.patch.object(
      trainer.initialization, 'initialize_from_vmoe', autospec=True)
  def test_initialize_from_vmoe(
      self, mock_initialize_from_vmoe):
    train_state = mock.create_autospec(trainer.TrainState, instance=True)
    mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)
    _ = trainer.initialize_train_state_from_checkpoint(
        train_state=train_state, name='initialize_from_vmoe', mesh=mesh,
        prefix='/foo', rules=[])
    mock_initialize_from_vmoe.assert_called_once_with(
        target=train_state, mesh=mesh, thread_pool=None, prefix='/foo',
        rules=[])

  @mock.patch.object(
      trainer.initialization, 'initialize_from_vit', autospec=True)
  def test_initialize_from_vit(
      self, mock_initialize_from_vit):
    train_state = mock.create_autospec(trainer.TrainState, instance=True)
    mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)
    _ = trainer.initialize_train_state_from_checkpoint(
        train_state=train_state, name='initialize_from_vit', mesh=mesh,
        filepath='/foo', rules=[])
    mock_initialize_from_vit.assert_called_once_with(
        target=train_state, mesh=mesh, filepath='/foo', rules=[])

  @mock.patch.object(
      trainer.initialization, 'initialize_from_orbax', autospec=True)
  def test_initialize_from_orbax(self, mock_initialize_from_orbax):
    train_state = mock.create_autospec(trainer.TrainState, instance=True)
    mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)
    _ = trainer.initialize_train_state_from_checkpoint(
        train_state=train_state, name='initialize_from_orbax', mesh=mesh,
        directory='/foo', rules=[])
    mock_initialize_from_orbax.assert_called_once_with(
        target=train_state, mesh=mesh, directory='/foo', rules=[])

  def test_unknown_method_raises(self):
    train_state = mock.create_autospec(trainer.TrainState, instance=True)
    mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)
    with self.assertRaisesRegex(ValueError, 'Unknown initialization method'):
      trainer.initialize_train_state_from_checkpoint(
          train_state=train_state, name='foo', mesh=mesh)


class MakeCreateTrainStateFnTest(absltest.TestCase):

  @mock.patch.object(trainer.optimizer, 'create_optimizer')
  def test(self, mock_create_optimizer):
    mock_create_optimizer.return_value = trainer.optimizer.optax.adam(
        learning_rate=0.1)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('d',))
    sharding = jax.sharding.NamedSharding(mesh, PartitionSpec(('d',)))
    shape_dtype = jax.ShapeDtypeStruct(
        shape=(8, 32, 32, 3), dtype=jnp.float32, sharding=sharding)
    train_state_init_fn = trainer.make_create_train_state_fn(
        model=trainer.nn.Conv(features=16, kernel_size=(3, 3)),
        optimizer_config={},
        input_shape_dtypes=(shape_dtype,),
        train_steps=10,
        seed=0,
        extra_rng_keys=('foo',))
    train_state = train_state_init_fn()
    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.unfreeze(train_state.params),
        {'kernel': jax.ShapeDtypeStruct((3, 3, 3, 16), jnp.float32),
         'bias': jax.ShapeDtypeStruct((16,), jnp.float32)})
    self.assertSetEqual(set(train_state.rngs.keys()), {'foo'})


class MakeTrainCostFnTest(parameterized.TestCase):

  @parameterized.parameters(
      ((None, None), {}),
      ((3., None), {'flops': 3. * 2 * 42}),
      ((None, 3.), {'device_seconds': 3. * 2 * 42}),
      ((5., 3.), {'flops': 5. * 2 * 42, 'device_seconds': 3. * 2 * 42}),
  )
  @mock.patch.object(jax, 'device_count', return_value=2)
  def test(self, flops_and_seconds_per_device, expected, _):
    with mock.patch.object(trainer.utils, 'get_flops_and_seconds_per_device',
                           return_value=flops_and_seconds_per_device):
      fn = trainer.make_train_cost_fn(compiled_fn=mock.Mock())
      self.assertEqual(fn(42), expected)


class MixupTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_batch', [[.6, .4]], (1, 2), [1.4, 2.4, 3.4, 4.4, 5.4, 4.]),
      ('_example',
       [[.6, .4], [.7, .3], [.5, .5], [1., 0.], [.9, .1], [.8, .2]], (6, 2),
       [1.4, 2.3, 3.5, 4., 5.1, 5.0]),
  )
  @mock.patch.object(jax.random, 'dirichlet')
  def test(self, alphas, shape, expected, mock_dirichlet):
    x = jnp.asarray([1., 2., 3., 4., 5., 6.])
    mock_dirichlet.return_value = jnp.asarray(alphas)
    output = trainer.mixup(
        jax.random.PRNGKey(0),
        x,
        concentration=10.0,  # Unused, dirichlet is mocked.
        shape=shape)
    np.testing.assert_allclose(output, np.asarray(expected))

  def test_inconsistent_shapes_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        r'Mixup with inconsistent shapes..* the first 1 dims must be equal to '
        r'\(3,\)'):
      trainer.mixup(
          jax.random.PRNGKey(0),
          tree=(np.asarray([1, 2, 3]), np.asarray([1, 2, 3, 4])),
          concentration=1)

  def test_not_broadcastable_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        r'Mixup with inconsistent shapes..* the first 1 dims must be '
        r'broadcastable to \(2,\)'):
      trainer.mixup(
          jax.random.PRNGKey(0),
          tree=np.asarray([1, 2, 3]),
          concentration=1,
          shape=(2, 2))

  def test_wrong_concentration_raises(self):
    with self.assertRaisesRegex(ValueError, "Mixup 'concentration'"):
      trainer.mixup(jax.random.PRNGKey(0), tree=np.asarray([1, 2, 3]),
                    concentration=0)

  def test_wrong_roll_axis_raises(self):
    with self.assertRaisesRegex(ValueError, "Mixup 'roll_axis'"):
      trainer.mixup(jax.random.PRNGKey(0), tree=np.asarray([1, 2, 3]),
                    concentration=0, shape=(1, 2), roll_axis=1)

  def test_wrong_shape_length_raises(self):
    with self.assertRaisesRegex(ValueError, "Mixup 'shape' has length 1"):
      trainer.mixup(
          jax.random.PRNGKey(0),
          tree=np.asarray([1, 2, 3]),
          concentration=1,
          shape=(2,))


class RestoreOrCreateTrainStateTest(absltest.TestCase):
  """Tests for the restore_or_create_train_state function."""

  def setUp(self):
    super().setUp()
    apply_fn = lambda x: x
    optimizer_tx = trainer.optimizer.optax.identity()
    # Function that creates a TrainState from scratch.
    def initialize_fn():
      return trainer.TrainState.create(
          apply_fn=apply_fn,
          params={'a': 1 * jnp.ones((5,)), 'b': 2 * jnp.ones((10,))},
          tx=optimizer_tx,
          rngs={})
    self.initialize_fn = initialize_fn
    self.mesh = Mesh(np.asarray(jax.devices()), ('a',))

  def test_create_from_scratch(self):
    """Tests when training starts from scratch."""
    ckpt_manager = mock.create_autospec(orbax.checkpoint.CheckpointManager,
                                        instance=True)
    ckpt_manager.latest_step.return_value = None
    train_state, last_seen = trainer.restore_or_create_train_state(
        ckpt_manager=ckpt_manager,
        initialize_fn=self.initialize_fn,
        axis_resources_regexes=[],
        mesh=self.mesh,
        initialization_kwargs={})
    chex.assert_trees_all_close(flax.core.unfreeze(train_state.params), {
        'a': 1 * np.ones((5,), dtype=np.float32),
        'b': 2 * np.ones((10,), dtype=np.float32),
    })
    chex.assert_trees_all_equal(train_state.step, 0)
    self.assertIsNone(last_seen)

  def test_continue_training(self):
    """Tests when training continues from an existing checkpoint."""
    # Mock the call to restore_checkpoint_partitioned.
    def restore_checkpoint_side_effect(*args, **kwargs):
      del args, kwargs
      def f():
        train_state = self.initialize_fn()
        train_state = train_state.replace(step=3)
        train_state = train_state.replace(params={
            'a': 3 * jnp.ones((5,), dtype=jnp.float32),
            'b': 4 * jnp.ones((10,), dtype=jnp.float32),
        })
        return train_state
      with self.mesh:
        state = pjit.pjit(f, out_shardings=None)()
      return {
          'state': state,
          'dataset_iterator': {'last_seen_index': 16},
      }
    ckpt_manager = mock.create_autospec(orbax.checkpoint.CheckpointManager,
                                        instance=True)
    ckpt_manager.latest_step.return_value = 3
    ckpt_manager.restore.side_effect = restore_checkpoint_side_effect
    # Call restore_or_create_train_state and check that the outputs are the
    # expected ones.
    train_state, last_seen = trainer.restore_or_create_train_state(
        ckpt_manager=ckpt_manager,
        initialize_fn=self.initialize_fn,
        axis_resources_regexes=[],
        mesh=self.mesh,
        initialization_kwargs={})
    chex.assert_trees_all_close(train_state.params, {
        'a': 3 * np.ones((5,), dtype=np.float32),
        'b': 4 * np.ones((10,), dtype=np.float32),
    })
    chex.assert_trees_all_equal(train_state.step, 3)
    self.assertEqual(last_seen, 16)

  @mock.patch.object(trainer, 'initialize_train_state_from_checkpoint')
  def test_initialize_from_checkpoint(self,
                                      mock_initialize_train_state_from_ckpt):
    """Tests when the model is initialized from a checkpoint."""
    # Mock the call to initialize_train_state_from_ckpt.
    def initialize_train_state_from_ckpt_side_effect(*args, **kwargs):
      del args, kwargs
      # Create value for parameter 'b', simulating that this is initialized from
      # an existing checkpoint.
      with self.mesh:
        b = pjit.pjit(
            lambda: 5 * jnp.ones((10,), dtype=jnp.float32),
            out_shardings=PartitionSpec(),
        )()
      # Create "empty" train state, containing only the expected shape and
      # sharding of the arrays.
      train_state = jax.eval_shape(self.initialize_fn)
      sharding = jax.sharding.NamedSharding(self.mesh, PartitionSpec())
      train_state = jax.tree_util.tree_map(
          lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding),
          train_state)
      train_state = train_state.replace(params={
          'a': train_state.params['a'], 'b': b,
      })
      return train_state
    mock_initialize_train_state_from_ckpt.side_effect = (
        initialize_train_state_from_ckpt_side_effect)
    # Call restore_or_create_train_state and check that the outputs are the
    # expected ones.
    ckpt_manager = mock.create_autospec(orbax.checkpoint.CheckpointManager,
                                        instance=True)
    ckpt_manager.latest_step.return_value = None
    train_state, last_seen = trainer.restore_or_create_train_state(
        ckpt_manager=ckpt_manager,
        initialize_fn=self.initialize_fn,
        axis_resources_regexes=[],
        mesh=self.mesh,
        initialization_kwargs={'foo': 'bar'})
    chex.assert_trees_all_close(flax.core.unfreeze(train_state.params), {
        'a': 1 * np.ones((5,), dtype=np.float32),
        'b': 5 * np.ones((10,), dtype=np.float32),
    })
    chex.assert_trees_all_equal(train_state.step, 0)
    self.assertIsNone(last_seen)


class TrainAndEvaluateTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.skipTest('foooo')

  @classmethod
  def create_dataset_train(cls):
    dataset = tf.data.Dataset.from_tensors({
        'image': tf.zeros((16, 32, 32, 3), dtype=tf.float32),
        'labels': tf.zeros((16, 10,), dtype=tf.float32),
    })
    dataset = dataset.repeat()
    return clu.data.TfDatasetIterator(dataset, checkpoint=False)

  @classmethod
  def create_dataset_eval(cls):
    dataset = tf.data.Dataset.from_tensors({
        'image': tf.zeros((16, 32, 32, 3), dtype=tf.float32),
        'labels': tf.zeros((16, 10,), dtype=tf.float32),
        VALID_KEY: tf.ones((16,), dtype=tf.bool),
    })
    return clu.data.TfDatasetIterator(dataset, checkpoint=False)

  @classmethod
  def create_dataset_fewshot(cls):
    dataset = tf.data.Dataset.from_tensors({
        'image': tf.zeros((16, 32, 32, 3), dtype=tf.float32),
        'label': tf.zeros((16,), dtype=tf.float32),
        VALID_KEY: tf.ones((16,), dtype=tf.bool),
    })
    return clu.data.TfDatasetIterator(dataset, checkpoint=False)

  @classmethod
  def create_flax_model(cls):

    class Model(nn.Module):

      @nn.compact
      def __call__(self, x):
        v = self.param('v', jax.nn.initializers.xavier_uniform(), (8, 3, 32))
        w = self.param('w', jax.nn.initializers.xavier_uniform(), (32, 10))
        y = jnp.mean(jnp.einsum('NHWC,ECD->ENHWD', x, v), axis=0)
        y = jnp.mean(jnp.einsum('NHWD,DO->NHWO', y, w), axis=(1, 2))
        return y, {'auxiliary_loss': jnp.square(y)}

    return Model()

  @parameterized.named_parameters(
      ('_standard', {}),
      ('_mixup', {'concentration': 0.2}),
  )
  @mock.patch.object(trainer, 'create_flax_model')
  @mock.patch.object(trainer.input_pipeline, 'get_data_num_examples',
                     return_value=32)
  @mock.patch.object(trainer.input_pipeline, 'get_datasets')
  def test(self, mixup_config, mock_get_datasets,
           unused_mock_get_data_num_examples, mock_create_flax_model):
    config = ml_collections.ConfigDict()
    config.dataset = ml_collections.ConfigDict()
    config.dataset.train = ml_collections.ConfigDict()
    config.dataset.train.batch_size = 16
    config.dataset.eval = ml_collections.ConfigDict()
    config.dataset.eval.batch_size = 16
    config.train_epochs = 2
    config.num_expert_partitions = jax.local_device_count()
    config.params_axis_resources = [('.*/v$', 'expert')]
    config.model = ml_collections.ConfigDict()
    config.model_eval_overrides = ml_collections.ConfigDict()
    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.name = 'sgd'
    config.optimizer.learning_rate = 1e-3
    config.optimizer.momentum = 0.9
    config.loss = ml_collections.ConfigDict()
    config.loss.name = 'softmax_xent'
    config.mixup = ml_collections.ConfigDict(mixup_config)
    config.extra_rng_keys = ('mixup',)
    config.summarize_arrays = [
        ('params_grads/v', 'norm'),
        ('opt_state/.*/hyperparams/learning_rate',),
    ]
    workdir = self.create_tempdir().full_path
    mock_get_datasets.return_value = {
        'train': self.create_dataset_train(),
        'eval': self.create_dataset_eval(),
    }
    mock_create_flax_model.return_value = self.create_flax_model()
    writer = mock.create_autospec(MetricWriter, instance=True)
    mesh = jax.sharding.Mesh(np.asarray(jax.local_devices()).reshape((-1, 1)),
                             ('expert', 'replica'))
    with mesh:
      trainer.train_and_evaluate(config, workdir, mesh, writer)

  @mock.patch.object(trainer, 'create_flax_model')
  @mock.patch.object(trainer.input_pipeline, 'get_data_num_examples',
                     return_value=32)
  @mock.patch.object(trainer.input_pipeline, 'get_datasets')
  @mock.patch.object(trainer.fewshot, '_get_datasets')
  def test_fewshot(self, mock_fewshot_get_datasets, mock_get_datasets,
                   unused_mock_get_data_num_examples, mock_create_flax_model):
    config = ml_collections.ConfigDict()
    config.dataset = ml_collections.ConfigDict()
    config.dataset.train = ml_collections.ConfigDict()
    config.dataset.train.batch_size = 16
    config.train_epochs = 2
    config.num_expert_partitions = jax.local_device_count()
    config.params_axis_resources = [('.*/v$', 'expert')]
    config.model = ml_collections.ConfigDict()
    config.model_eval_overrides = ml_collections.ConfigDict()
    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.name = 'sgd'
    config.optimizer.learning_rate = 1e-3
    config.optimizer.momentum = 0.9
    config.loss = ml_collections.ConfigDict()
    config.loss.name = 'softmax_xent'
    config.fewshot = ml_collections.ConfigDict()
    config.fewshot.datasets = {'foo': ('tfds_name', 'train', 'test')}
    config.fewshot.shots = [5]
    config.fewshot.l2_regs = [0.01]
    workdir = self.create_tempdir().full_path
    mock_get_datasets.return_value = {
        'train': self.create_dataset_train(),
    }
    mock_fewshot_get_datasets.return_value = {
        'foo':
            (self.create_dataset_fewshot(), self.create_dataset_fewshot(), 10)
    }
    mock_create_flax_model.return_value = self.create_flax_model()
    writer = mock.create_autospec(MetricWriter, instance=True)
    mesh = jax.sharding.Mesh(np.asarray(jax.local_devices()).reshape((-1, 1)),
                             ('expert', 'replica'))
    with mesh:
      trainer.train_and_evaluate(config, workdir, mesh, writer)

  @mock.patch.object(trainer.input_pipeline, 'get_datasets')
  def test_missing_train_dataset(self, mock_get_datasets):
    config = ml_collections.ConfigDict()
    config.num_expert_partitions = jax.local_device_count()
    config.dataset = ml_collections.ConfigDict()
    workdir = self.create_tempdir().full_path
    mock_get_datasets.return_value = {}
    writer = mock.create_autospec(MetricWriter, instance=True)
    mesh = jax.sharding.Mesh(np.asarray(jax.local_devices()).reshape((-1, 1)),
                             ('expert', 'replica'))
    with self.assertRaisesRegex(KeyError, 'You must have a "train" variant'):
      with mesh:
        trainer.train_and_evaluate(config, workdir, mesh, writer)


class TrainStepTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape((-1, 1)),
        ('expert', 'replica'))

  @parameterized.parameters((0,), (1,), (2,))
  def test(self, microsteps):
    """Tests that using microsteps does not affect the final gradients."""

    def apply_fn(variables, x, **_):
      y = jnp.einsum('nd,dk->nk', x, variables['params'])
      return y, {}

    @functools.partial(pjit.pjit, out_shardings=(None, None, None, None))
    def run_step():
      state = trainer.TrainState.create(
          apply_fn=apply_fn,
          params=jnp.zeros((1, 1), dtype=jnp.float32),
          tx=optax.sgd(1.), rngs={}
      )

      x = jnp.linspace(-2., +2., num=4 * jax.device_count()).reshape((-1, 1))
      y = 2. * x
      loss_fn = lambda xx, yy: (0.5 * jnp.square(xx - yy)).mean()
      state, metrics = trainer.train_step(
          state, x, y, loss_fn=loss_fn, microsteps=microsteps)
      return x, y, state, metrics

    x, y, state, metrics = self.mesh(run_step)()
    expected_grads = -(2 * x * x).mean(axis=0)[None, :]
    expected_params = -expected_grads
    chex.assert_trees_all_close(state.params, expected_params)
    expected_metrics = {
        'main_loss': 0.5 * jnp.square(-y).mean(),
        'total_loss': 0.5 * jnp.square(-y).mean(),
        'global_norm/grads': jnp.linalg.norm(expected_grads.flatten()),
        'global_norm/params': jnp.linalg.norm(expected_params.flatten()),
    }
    chex.assert_trees_all_close(
        {k: v for k, v in metrics.items() if k in expected_metrics},
        expected_metrics)


class WrapTrainStepWithAdversarialAttackTest(parameterized.TestCase):

  @classmethod
  def apply_fn(cls, variables, inputs, rngs):
    del rngs
    outputs = inputs * variables['params']['w']
    metrics = {'auxiliary_loss': jnp.square(outputs).sum()}
    return outputs, metrics

  @classmethod
  def loss_fn(cls, a, b):
    return jnp.square(a - b).sum()

  @classmethod
  def train_step_fn(cls, state, inputs, targets):
    @jax.grad
    def compute_grads(params):
      outputs, _ = state.apply_fn({'params': params}, inputs, rngs=state.rngs)
      return cls.loss_fn(outputs, targets)
    grads = compute_grads(state.params)
    return state.apply_gradients(grads=grads), {}

  # We are solving a linear regression problem with a single weight (initial
  # weight w = 1): (x * w - y)^2. x = [-1, 0, +1], y = [-2, 0, +2].
  @parameterized.named_parameters(
      # Gradient of the loss w.r.t the inputs (at w = 1): [+2, 0, -2].
      # The adversarial attack (without auxiliary loss) changes the input to:
      # [-1, 0, +1] + 0.1 * [+1, 0, -1] = [-.9, 0, +.9].
      # Gradient of the loss w.r.t the weight with the resulting inputs:
      # 2 * (- 0.9 * 1.1) = -3.96.
      # SGD updates the parameter to: 1 - 0.1 * (-3.96) = 1.396
      ('_no_attack_aux', False, 1.396),
      # Gradient of the loss w.r.t the inputs (at w = 1): [+2, 0, -2].
      # Gradient of the auxiliary loss w.r.t the inputs (at w = 1): [-2, 0, +2].
      # The adversarial attack (which also attacks the aux loss) changes the
      # input to:
      # [-1, 0, +1] + 0.1 * sign([+2, 0, -2] + [+2, 0, -2]) = [-1, 0, +1].
      # Gradient of the loss w.r.t. the weight with the resulting inputs:
      # 2 * (- 1 * 2) = -4.
      # SGD updates the parameter to: 1 - 0.1 * (-4.) = 1.4
      ('_attack_aux', True, 1.4)
  )
  def test(self, attack_auxiliary_loss, expected_weight):
    wrapped_train_step_fn = trainer.wrap_train_step_with_adversarial_attack(
        train_step_fn=self.train_step_fn, loss_fn=self.loss_fn,
        max_epsilon=0.1, num_updates=10,
        attack_auxiliary_loss=attack_auxiliary_loss)
    wrapped_train_step_fn = jax.jit(wrapped_train_step_fn)

    state = trainer.TrainState.create(
        apply_fn=self.apply_fn,
        params={'w': np.ones((1,), dtype=np.float32)},
        tx=optax.sgd(0.1),
        rngs={})
    inputs = np.asarray([-1., 0., 1.], dtype=np.float32)
    targets = np.asarray([-2, 0., 2.], dtype=np.float32)
    new_state, _ = wrapped_train_step_fn(state, inputs, targets)
    self.assertAlmostEqual(new_state.params['w'], expected_weight)


class WrapTrainStepWithMixupTest(parameterized.TestCase):

  @classmethod
  def train_step_fn(cls, state, images, unused_labels, extra):
    rngs = {'foo': jnp.ones_like(state.rngs['foo'])}
    if isinstance(state.rngs, trainer.flax.core.FrozenDict):
      rngs = trainer.flax.core.freeze(rngs)
    state = state.replace(rngs=rngs)
    state = state.replace(step=state.step + 1)
    return state, {'extra': extra, 'images': images}

  @parameterized.parameters((None,), (0.,))
  def test_no_concentration(self, concentration):
    fn = trainer.wrap_train_step_with_mixup(self.train_step_fn,
                                            concentration=concentration)
    self.assertEqual(fn, self.train_step_fn)

  @parameterized.parameters(('batch',), ('device',), ('example',))
  def test_granularity(self, granularity):
    # Wrap it with mixup.
    wrapped_train_state_fn = jax.jit(
        trainer.wrap_train_step_with_mixup(
            self.train_step_fn, concentration=1.0, granularity=granularity))
    # PRNGKeys used by the train step.
    rngs = {'mixup': jax.random.PRNGKey(1), 'foo': jax.random.PRNGKey(3)}
    # A trivial TrainState.
    train_state = trainer.TrainState(
        step=0, apply_fn=lambda _: _, params={}, tx=lambda _: _, opt_state=(),
        rngs=rngs)
    x = (jnp.arange(16 * 32, dtype=jnp.float32).reshape(16, 32) + 1) * 100
    y = jnp.arange(16, dtype=jnp.float32)
    train_state, metrics = wrapped_train_state_fn(train_state, x, y, -1.0)
    self.assertEqual(train_state.step, 1)
    self.assertEqual(metrics['extra'], -1.0)
    # Check that the images passed to the wrapped train_state were modified.
    np.testing.assert_array_less(np.zeros_like(x),
                                 np.abs(metrics['images'] - x))
    # Convert PRNGKeys to int64 to subtract them.
    rngs = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.int64), rngs)
    new_rngs = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.int64),
                                      train_state.rngs)
    # Check that both PRNGKeys have been updated.
    np.testing.assert_array_less(
        np.zeros_like(rngs['mixup']), np.abs(rngs['mixup'] - new_rngs['mixup']))
    np.testing.assert_array_less(
        np.zeros_like(rngs['foo']), np.abs(rngs['foo'] - new_rngs['foo']))


if __name__ == '__main__':
  absltest.main()
