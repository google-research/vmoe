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

"""Tests for trainer."""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf
from vmoe.train import trainer


PartitionSpec = trainer.PartitionSpec
Mesh = trainer.Mesh


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
    train_state_axis_resources = trainer.tree_axis_resources_from_regexes(
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
    train_labels = jax.random.uniform(key, (8, 16))
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


class ParsePartitionSpecTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_none', None, PartitionSpec()),
      ('_string', 'a', PartitionSpec('a')),
      ('_tuple', ('a', ('b', 'c')), PartitionSpec('a', ('b', 'c'))),
      ('_partition_spec', PartitionSpec('a'), PartitionSpec('a')),
  )
  def test(self, spec, expected):
    self.assertEqual(trainer.parse_partition_spec(spec), expected)


class TrainAndEvaluateTest(absltest.TestCase):

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
    return dataset

  @classmethod
  def create_dataset_eval(cls):
    dataset = tf.data.Dataset.from_tensors({
        'image': tf.zeros((16, 32, 32, 3), dtype=tf.float32),
        'labels': tf.zeros((16, 10,), dtype=tf.float32),
        '_fake': tf.zeros((16,), dtype=tf.bool),
    })
    return dataset

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

  @mock.patch.object(trainer.metric_writers, 'create_default_writer')
  @mock.patch.object(trainer, 'create_flax_model')
  @mock.patch.object(trainer.input_pipeline, 'get_data_num_examples',
                     return_value=32)
  @mock.patch.object(trainer.input_pipeline, 'get_datasets')
  def test(self, mock_get_datasets, unused_mock_get_data_num_examples,
           mock_create_flax_model, mock_create_default_writer):
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
    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.name = 'sgd'
    config.optimizer.learning_rate = 1e-3
    config.optimizer.momentum = 0.9
    config.loss = ml_collections.ConfigDict()
    config.loss.name = 'softmax_xent'
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
    config.dataset = ml_collections.ConfigDict()
    workdir = self.create_tempdir().full_path
    mock_get_datasets.return_value = {}
    with self.assertRaisesRegex(KeyError, 'You must have a "train" variant'):
      trainer.train_and_evaluate(config=config, workdir=workdir)


class TreeAxisResourcesFromRegexesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_empty_regexes', {'a': 1, 'b': 2, 'c': 3}, [],
       {'a': PartitionSpec(), 'b': PartitionSpec(), 'c': PartitionSpec()}),
      ('_single_string', {'a': 1, 'b': 2, 'c': 3},
       [('b', 'x')],
       {'a': PartitionSpec(), 'b': PartitionSpec('x'), 'c': PartitionSpec()}),
      ('_first_match', {'a': 1, 'bb': 2, 'c': 3},
       [('b', ('x',)), ('bb', ('x', 'y'))],
       {'a': PartitionSpec(), 'bb': PartitionSpec('x'), 'c': PartitionSpec()}),
  )
  def test(self, tree, axis_resources_regexes, expected):
    output = trainer.tree_axis_resources_from_regexes(
        tree=tree, axis_resources_regexes=axis_resources_regexes)
    self.assertEqual(output, expected)

  def test_train_state(self):
    params = {'a': 1, 'b': 2, 'c': 3}
    rngs = {'dropout': None}
    train_state = trainer.TrainState.create(
        apply_fn=lambda x: x,
        params=params,
        tx=trainer.optimizer.optax.rmsprop(0.1),
        rngs=rngs)
    output = trainer.tree_axis_resources_from_regexes(
        tree=train_state, axis_resources_regexes=[
            ('.*/a$', ('expert',)),
            ('.*/c$', (('expert', 'width'),)),
        ])
    self.assertIsInstance(output, trainer.TrainState)
    self.assertEqual(output.params['a'], PartitionSpec('expert'))
    self.assertEqual(output.params['b'], PartitionSpec())
    self.assertEqual(output.params['c'], PartitionSpec(('expert', 'width')))


if __name__ == '__main__':
  absltest.main()
