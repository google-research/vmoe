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

"""Tests for trainer."""
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import clu.data
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf
from vmoe.train import trainer


PartitionSpec = trainer.PartitionSpec
Mesh = trainer.Mesh
VALID_KEY = trainer.input_pipeline.VALID_KEY


class CreateFlaxModelTest(absltest.TestCase):

  @mock.patch.object(trainer, '_getattr')
  def test_success(self, mock_getattr):
    foo_cls = mock.MagicMock()
    mock_getattr.return_value = foo_cls
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


class CreateTrainStateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform not in ('gpu', 'tpu'):
      self.skipTest('CreateTrainStateTest can only be tested on GPU or TPUs.')

  @mock.patch.object(trainer.optimizer, 'create_optimizer')
  def test(self, mock_create_optimizer):
    mock_create_optimizer.return_value = trainer.optimizer.optax.adam(
        learning_rate=0.1)
    devices = np.asarray(jax.devices())[:2]
    devices = np.expand_dims(devices, axis=-1)
    axis_names = ('a', 'b')
    mesh = Mesh(devices=devices, axis_names=axis_names)
    rngs = {'params': jax.random.PRNGKey(0), 'foo': jax.random.PRNGKey(1)}
    train_state_init_fn = trainer.create_train_state_initialize_from_scratch_fn(
        model=trainer.nn.Conv(features=16, kernel_size=(3, 3)),
        optimizer_config=ml_collections.ConfigDict(),
        input_shape=(16, 224, 224, 3),
        input_axis_resources=PartitionSpec(('a', 'b')),
        train_steps=10)
    train_state_axis_resources = trainer.partitioning.tree_axis_resources_from_regexes(
        tree=jax.eval_shape(train_state_init_fn, rngs),
        axis_resources_regexes=[
            ('kernel', (None, None, None, 'a')),
            ('bias', 'b'),
        ])
    train_state = trainer.create_train_state(
        initialize_fn=train_state_init_fn,
        axis_resources=train_state_axis_resources,
        rngs=rngs,
        mesh=mesh)
    self.assertEqual(
        jax.tree_structure(train_state),
        jax.tree_structure(train_state_axis_resources))
    self.assertEqual(int(train_state.step), 0)
    self.assertEqual(set(train_state.rngs.keys()), {'foo'})


class CreateOrReuseTrainStateTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform not in ('gpu', 'tpu'):
      self.skipTest(
          'CreateOrReuseTrainStateTest can only be tested on GPU or TPUs.')

  @parameterized.named_parameters(
      ('_replicated', PartitionSpec()),
      ('_partitioned', PartitionSpec('a',)),
  )
  def test(self, partition_spec):
    # Parameter 'foo' is created, 'bar' is reused.
    devices = np.asarray(jax.local_devices())
    mesh = Mesh(devices=devices, axis_names=('a',))
    rngs = {'params': jax.random.PRNGKey(0)}
    apply_fn = lambda x: x
    optimizer_tx = trainer.optimizer.optax.identity()

    train_state = trainer.TrainState(
        step=3 * np.ones((), dtype=np.int32),
        apply_fn=apply_fn,  # Unused.
        params={
            'foo': np.array([], dtype=np.float32),
            'bar': np.zeros((devices.size * 8, 2), dtype=np.float32),
        },
        tx=optimizer_tx,  # Unused.
        opt_state=optimizer_tx.init(None),  # Unused.
        rngs={})  # Unused.
    train_state_axis_resources = train_state.replace(
        step=PartitionSpec(),
        params={'foo': partition_spec, 'bar': partition_spec})

    def initialize_fn(_):
      params = {
          'foo': 1 * jnp.ones((devices.size * 4, 3), dtype=np.float32),
          'bar': 2 * jnp.ones((devices.size * 8, 2), dtype=np.float32),
      }
      return trainer.TrainState.create(
          apply_fn=apply_fn, params=params, tx=optimizer_tx, rngs={})

    train_state = trainer.create_or_reuse_train_state(
        initialize_fn=initialize_fn, axis_resources=train_state_axis_resources,
        rngs=rngs, reuse_train_state=train_state, mesh=mesh)
    chex.assert_trees_all_equal(train_state.step, 3)
    chex.assert_trees_all_equal(
        train_state.params,
        {
            'foo': np.ones((devices.size * 4, 3), dtype=np.float32),
            'bar': np.zeros((devices.size * 8, 2), dtype=np.float32),
        })


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


class InitializeTrainStateFromCheckpointTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.train_state = trainer.TrainState(
        step=jnp.zeros((), dtype=np.int32),
        apply_fn=lambda x: x,  # Unused.
        params={
            'a': jnp.zeros((4, 3), dtype=np.float32),
        },
        tx=lambda x: x,  # Unused.
        opt_state={},  # Unused.
        rngs={})  # Unused.
    self.train_state_axis_resources = jax.tree_map(
        lambda _: trainer.PartitionSpec(), self.train_state)
    self.mock_initialize_from_vit = self.enter_context(
        mock.patch.object(trainer.initialization, 'initialize_from_vit'))
    self.mock_initialize_from_vit.return_value = {
        'a': jnp.ones((4, 3), dtype=np.float32),
    }
    self.mock_initialize_from_vmoe_release = self.enter_context(
        mock.patch.object(trainer.initialization,
                          'initialize_from_vmoe_release'))
    self.mock_initialize_from_vmoe_release.return_value = {
        'a': jnp.ones((4, 3), dtype=np.float32),
    }

  @parameterized.parameters(
      ('initialize_from_vmoe_release', 0),
      ('initialize_from_vit', 0),
  )
  def test(self, name, step_after_initialization):
    train_state = trainer.initialize_train_state_from_checkpoint(
        train_state=self.train_state,
        axis_resources=self.train_state_axis_resources,
        name=name)
    # We only initialize the parameters of the TrainState, step should be 0.
    chex.assert_trees_all_close(
        train_state.params, {'a': np.ones((4, 3), dtype=np.float32)})
    np.testing.assert_allclose(train_state.step, step_after_initialization)

  def test_unknown_method_raises(self):
    with self.assertRaisesRegex(ValueError, 'Unknown initialization method'):
      trainer.initialize_train_state_from_checkpoint(
          train_state=self.train_state,
          axis_resources=self.train_state_axis_resources,
          name='foo')


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

  def setUp(self):
    super().setUp()
    apply_fn = lambda x: x
    optimizer_tx = trainer.optimizer.optax.identity()
    def initialize_fn(rngs):
      return trainer.TrainState.create(
          apply_fn=apply_fn,
          params={'a': 1 * jnp.ones((5,)), 'b': 2 * jnp.ones((10,))},
          tx=optimizer_tx,
          rngs=rngs)
    self.rngs = {'params': jax.random.PRNGKey(0)}
    self.initialize_fn = initialize_fn
    self.axis_resources = jax.tree_map(
        lambda _: PartitionSpec(), jax.eval_shape(initialize_fn, self.rngs))
    self.mesh = Mesh(np.asarray(jax.devices()), ('a',))

  def test_create_from_scratch(self):
    prefix = os.path.join(self.create_tempdir().full_path, 'ckpt_1')
    train_state = trainer.restore_or_create_train_state(
        prefix=prefix, initialize_fn=self.initialize_fn,
        axis_resources=self.axis_resources, rngs=self.rngs, mesh=self.mesh,
        initialization_kwargs={})
    chex.assert_trees_all_close(train_state.params, {
        'a': 1 * np.ones((5,), dtype=np.float32),
        'b': 2 * np.ones((10,), dtype=np.float32),
    })
    chex.assert_trees_all_equal(train_state.step, 0)
    chex.assert_trees_all_equal(train_state.rngs,
                                {'params': jax.random.PRNGKey(0)})

  def test_continue_training(self):
    prefix = os.path.join(self.create_tempdir().full_path, 'ckpt_1')
    with mock.patch.object(
        trainer.checkpoints_base,
        'find_latest_complete_checkpoint_for_prefix',
        return_value=prefix):
      with mock.patch.object(trainer.checkpoints_partitioned,
                             'restore_checkpoint') as mock_restore_checkpoint:
        mock_restore_checkpoint.return_value = self.axis_resources.replace(
            step=3 * np.ones((), dtype=np.int32),
            params={
                'a': 3 * np.ones((5,), dtype=np.float32),
                'b': 4 * np.ones((10,), dtype=np.float32),
            },
            rngs={'params': jax.random.PRNGKey(5)})
        train_state = trainer.restore_or_create_train_state(
            prefix=prefix, initialize_fn=self.initialize_fn,
            axis_resources=self.axis_resources, rngs=self.rngs, mesh=self.mesh,
            initialization_kwargs={})
    chex.assert_trees_all_close(train_state.params, {
        'a': 3 * np.ones((5,), dtype=np.float32),
        'b': 4 * np.ones((10,), dtype=np.float32),
    })
    chex.assert_trees_all_equal(train_state.step, 3)
    chex.assert_trees_all_equal(train_state.rngs,
                                {'params': jax.random.PRNGKey(5)})

  def test_initialize_from_checkpoint(self):
    prefix = os.path.join(self.create_tempdir().full_path, 'ckpt_1')

    def mock_initialize_train_state_from_checkpoint(train_state, **_):
      return train_state.replace(
          step=jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32),
          params={
              'a': train_state.params['a'],
              'b': 5 * np.ones_like(train_state.params['b']),
          },
          rngs={'params': jax.random.PRNGKey(9)})

    with mock.patch.object(
        trainer, 'initialize_train_state_from_checkpoint',
        side_effect=mock_initialize_train_state_from_checkpoint):
      train_state = trainer.restore_or_create_train_state(
          prefix=prefix, initialize_fn=self.initialize_fn,
          axis_resources=self.axis_resources, rngs=self.rngs, mesh=self.mesh,
          initialization_kwargs={'foo': 'bar'})
    chex.assert_trees_all_close(train_state.params, {
        'a': 1 * np.ones((5,), dtype=np.float32),
        'b': 5 * np.ones((10,), dtype=np.float32),
    })
    chex.assert_trees_all_equal(train_state.step, 0)
    chex.assert_trees_all_equal(train_state.rngs,
                                {'params': jax.random.PRNGKey(9)})


class TrainAndEvaluateTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform not in ('gpu', 'tpu'):
      self.skipTest('CreateTrainStateTest can only be tested on GPU or TPUs.')

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
  @mock.patch.object(trainer.metric_writers, 'create_default_writer')
  @mock.patch.object(trainer, 'create_flax_model')
  @mock.patch.object(trainer.input_pipeline, 'get_data_num_examples',
                     return_value=32)
  @mock.patch.object(trainer.input_pipeline, 'get_datasets')
  def test(self, mixup_config, mock_get_datasets,
           unused_mock_get_data_num_examples, mock_create_flax_model,
           mock_create_default_writer):
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
    config.plot_norm_grad_patterns = ('v',)
    config.plot_norm_train_state_patterns = (
        'opt_state/.*/hyperparams/learning_rate',)
    workdir = self.create_tempdir().full_path
    mock_get_datasets.return_value = {
        'train': self.create_dataset_train(),
        'eval': self.create_dataset_eval(),
    }
    mock_create_flax_model.return_value = self.create_flax_model()
    mock_create_default_writer.return_value = mock.MagicMock()
    trainer.train_and_evaluate(config=config, workdir=workdir)


  @mock.patch.object(trainer.input_pipeline, 'get_datasets')
  def test_missing_train_dataset(self, mock_get_datasets):
    config = ml_collections.ConfigDict()
    config.num_expert_partitions = jax.local_device_count()
    config.dataset = ml_collections.ConfigDict()
    workdir = self.create_tempdir().full_path
    mock_get_datasets.return_value = {}
    with self.assertRaisesRegex(KeyError, 'You must have a "train" variant'):
      trainer.train_and_evaluate(config=config, workdir=workdir)


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
    rngs = jax.tree_map(lambda x: np.asarray(x, dtype=np.int64), rngs)
    new_rngs = jax.tree_map(lambda x: np.asarray(x, dtype=np.int64),
                            train_state.rngs)
    # Check that both PRNGKeys have been updated.
    np.testing.assert_array_less(
        np.zeros_like(rngs['mixup']), np.abs(rngs['mixup'] - new_rngs['mixup']))
    np.testing.assert_array_less(
        np.zeros_like(rngs['foo']), np.abs(rngs['foo'] - new_rngs['foo']))


if __name__ == '__main__':
  absltest.main()
