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

"""Classes and functions used for training (from-scratch and fine-tuning)."""
import functools
import multiprocessing.pool
import os
import re
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
import flax.linen as nn
import flax.serialization
import flax.training.train_state
import flax.traverse_util
import jax
from jax.experimental import maps
from jax.experimental import pjit
import jax.numpy as jnp
import ml_collections
import numpy as np
from vmoe import multihost_utils
from vmoe import partitioning
from vmoe import utils
from vmoe.checkpoints import base as checkpoints_base
from vmoe.checkpoints import partitioned as checkpoints_partitioned
from vmoe.checkpoints import periodic_actions as checkpoints_periodic_actions
from vmoe.data import input_pipeline
from vmoe.data import pjit_utils
from vmoe.evaluate import ensemble
from vmoe.evaluate import evaluator
from vmoe.nn import models
from vmoe.train import initialization
from vmoe.train import optimizer
from vmoe.train import periodic_actions as train_periodic_actions
from vmoe.train import train_state as train_state_module


Array = jax.numpy.ndarray
AsyncResult = multiprocessing.pool.AsyncResult
AxisResourcesRegexes = Sequence[Tuple[str, 'UnparsedPartitionSpec']]
Dataset = input_pipeline.tf.data.Dataset
Mesh = partitioning.Mesh
PartitionSpec = partitioning.PartitionSpec
PeriodicCheckpointSaver = checkpoints_periodic_actions.PeriodicSaveCheckpoint
PRNGKey = Union[jax.numpy.ndarray, jax.random.KeyArray]
PyTree = Any
ReportProgress = train_periodic_actions.ReportProgress
SingleProcessPeriodicAction = train_periodic_actions.SingleProcessPeriodicAction
ThreadPool = multiprocessing.pool.ThreadPool
TrainState = train_state_module.TrainState
TrainStateAxisResources = train_state_module.TrainStateAxisResources
TrainStateInitFromScratchFn = Callable[[Mapping[str, PRNGKey]], TrainState]
UnparsedPartitionSpec = Union[str, Tuple[Union[str, Tuple[str, ...]], ...]]


_getattr = getattr  # Alias of _getattr so that we can mock it in tests.


def create_checkpoint_hook(*, workdir: str, progress_hook: ReportProgress,
                           train_state_axis_resources: TrainStateAxisResources,
                           train_steps: int,
                           **kwargs) -> PeriodicCheckpointSaver:
  on_steps = set(kwargs.pop('on_steps', []))
  # Always save checkpoint on the last step.
  on_steps.update([train_steps])
  return PeriodicCheckpointSaver(
      prefix=os.path.join(workdir, 'ckpt'),
      state_axis_resources=train_state_axis_resources,
      report_progress=progress_hook,
      report_progress_name='ckpt',
      on_steps=on_steps,
      **kwargs)


def create_evaluation_hook(*, writer: metric_writers.MetricWriter,
                           progress_hook: ReportProgress,
                           datasets: Mapping[str, Dataset],
                           apply_fn: Callable[..., Any],
                           loss_fn: Callable[[Array, Array], Array],
                           params_axis_resources: PyTree,
                           input_axis_resources: PartitionSpec, first_step: int,
                           train_steps: int, extra_rng_keys: Sequence[str],
                           **kwargs) -> evaluator.EvaluateMultipleDatasets:
  """Returns a hook to evaluate a model periodically."""
  on_steps = set(kwargs.pop('on_steps', []))
  # Always evaluate on the first and last step.
  on_steps.update([first_step, train_steps])
  return evaluator.EvaluateMultipleDatasets(
      apply_fn=apply_fn,
      loss_fn=loss_fn,
      params_axis_resources=params_axis_resources,
      input_axis_resources=input_axis_resources,
      datasets=datasets,
      metric_writer=writer,
      rng_keys=extra_rng_keys,
      report_progress=progress_hook,
      report_progress_name='eval',
      on_steps=on_steps,
      **kwargs)


def create_flax_model(*, config: Dict[str, Any],
                      deterministic: bool) -> nn.Module:
  if 'name' not in config:
    raise KeyError('The model config must have a "name" field.')
  if isinstance(config, ml_collections.ConfigDict):
    config = config.to_dict()
  model_cls = _getattr(models, config.pop('name'))
  return model_cls(deterministic=deterministic, **config)


def create_profile_hook(*, workdir: str, **kwargs):
  all_processes = kwargs.pop('all_processes', False)
  if all_processes:
    return periodic_actions.ProfileAllHosts(logdir=workdir, **kwargs)
  else:
    return SingleProcessPeriodicAction(
        periodic_action=periodic_actions.Profile(logdir=workdir, **kwargs))


def create_progress_hook(*, writer: metric_writers.MetricWriter,
                         first_step: int, train_steps: int,
                         **kwargs) -> ReportProgress:
  on_steps = set(kwargs.pop('on_steps', []))
  # Always report progress on the first and last step.
  on_steps.update([first_step, train_steps])
  return ReportProgress(
      num_train_steps=train_steps, writer=writer, on_steps=on_steps, **kwargs)


def create_train_state(
    *,
    initialize_fn: TrainStateInitFromScratchFn,
    axis_resources: TrainStateAxisResources,
    rngs: Mapping[str, PRNGKey],
    mesh: Optional[Mesh] = None,
) -> TrainState:
  """Creates a TrainState object initialized from scratch.

  Args:
    initialize_fn: Function used to create and initialize a train state from
      scratch.
    axis_resources: A PyTree with the same structure as a TrainState, but with
      PartitionSpec leaves.
    rngs: Dictionary of PRNGs to use during model initialization.
    mesh: Logical mesh used by pjit. If None, uses the currently active mesh.

  Returns:
    A TrainState.
  """
  mesh = mesh or maps.thread_resources.env.physical_mesh
  with maps.mesh(mesh.devices, mesh.axis_names):
    train_state = pjit.pjit(
        initialize_fn,
        in_axis_resources=(None,),
        out_axis_resources=axis_resources)(rngs)
  return train_state


def create_train_state_initialize_from_scratch_fn(
    *,
    model: nn.Module,
    optimizer_config: Dict[str, Any],
    input_shape: Tuple[int, ...],
    input_axis_resources: PartitionSpec,
    train_steps: int) -> TrainStateInitFromScratchFn:
  """Returns a function that creates and initializes a TrainState from scratch.

  Args:
    model: Linen module representing the model.
    optimizer_config: A ConfigDict with the optimizer configuration.
    input_shape: Shape of the inputs to the model.
    input_axis_resources: PartitionSpec for the inputs of the model.
    train_steps: Total number of training steps.

  Returns:
    A function that creates a new TrainState when called.
  """
  # Optimizer is an optax.GradientTransform.
  tx = optimizer.create_optimizer(**optimizer_config, total_steps=train_steps)

  def initialize(rngs: Mapping[str, PRNGKey]):
    inputs = jnp.zeros(input_shape, dtype=jnp.float32)
    inputs = partitioning.with_sharding_constraint(inputs, input_axis_resources)
    variables = model.init(rngs, inputs)
    rngs = dict(**rngs)
    rngs.pop('params')  # This PRNGKey is not used anymore.
    return TrainState.create(
        apply_fn=model.apply, tx=tx, rngs=rngs, **variables)

  return initialize


def restore_or_create_train_state(
    *,
    prefix: str,
    initialize_fn: TrainStateInitFromScratchFn,
    axis_resources: TrainState,
    rngs: Mapping[str, PRNGKey],
    mesh: Optional[Mesh] = None,
    thread_pool: Optional[ThreadPool] = None,
) -> TrainState:
  """Restores a TrainState from the latest complete checkpoint or creates one.

  Args:
    prefix: Prefix used to find the checkpoint (e.g. '/tmp/ckpt'). This assumes
      that checkpoints are partitioned. Thus, a complete checkpoint has files
      such as '/tmp/ckpt_1.index' and '/tmp/ckpt_1.data-?????-of-?????'.
    initialize_fn: Function used to create and initialize a train state from
      scratch.
    axis_resources: A PyTree with the same structure as a TrainState, but with
      PartitionSpec leaves.
    rngs: Dictionary of PRNGs to use during model initialization.
    mesh: Logical mesh used by pjit. If None, uses the currently active mesh.
    thread_pool: Thread pool used to restore checkpoints.

  Returns:
    A TrainState.
  """
  mesh = mesh or maps.thread_resources.env.physical_mesh
  prefix = checkpoints_base.find_latest_complete_checkpoint_for_prefix(
      prefix=prefix, suffixes=('.index', '.data'))
  if prefix:
    # Restore train_state from checkpoints to CPU memory.
    train_state = checkpoints_partitioned.restore_checkpoint(
        prefix=prefix,
        tree=jax.eval_shape(initialize_fn, rngs),
        axis_resources=axis_resources,
        mesh=mesh,
        thread_pool=thread_pool)
    # Copy TrainState to device memory, and return.
    with maps.mesh(mesh.devices, mesh.axis_names):
      return pjit.pjit(
          fun=lambda x: x,
          in_axis_resources=(axis_resources,),
          out_axis_resources=axis_resources)(train_state)
  # If no complete checkpoints exist with the given prefix, create a train
  # state from scratch on device.
  return create_train_state(
      initialize_fn=initialize_fn, axis_resources=axis_resources, rngs=rngs,
      mesh=mesh)


def get_loss_fn(name: str, **kwargs):
  """Returns the train/evaluation losses and the way to predict labels."""
  softmax_xent = optimizer.optax.softmax_cross_entropy
  sigmoid_xent = optimizer.optax.sigmoid_binary_cross_entropy
  default_label_pred_fn = lambda logits: jnp.argmax(logits, -1)

  if name == 'softmax_xent':
    train_loss_fn = functools.partial(softmax_xent, **kwargs)
    eval_loss_fn = train_loss_fn
    return train_loss_fn, eval_loss_fn, default_label_pred_fn
  elif name == 'sigmoid_xent':
    # The sum with axis -1 is to sum the binary cross entropy over all classes.
    def train_loss_fn(logits, labels):
      return jnp.sum(sigmoid_xent(logits, labels, **kwargs), axis=-1)
    eval_loss_fn = train_loss_fn
    return train_loss_fn, eval_loss_fn, default_label_pred_fn
  elif name == 'ensemble_softmax_xent':
    # In the case of the ensemble softmax cross-entropy, the training loss and
    # the eval loss functions differ (see the discussion in Appendix D.1 of
    # https://arxiv.org/pdf/2110.03360.pdf).
    if 'ensemble_size' not in kwargs:
      raise ValueError(('The ensemble softmax CE needs the key ensemble_size: '
                        f'received {kwargs!r}.'))
    ensemble_size = kwargs.pop('ensemble_size')
    train_loss_fn = functools.partial(softmax_xent, **kwargs)
    eval_loss_fn = functools.partial(ensemble.ensemble_softmax_xent,
                                     ensemble_size=ensemble_size)
    label_pred_fn = functools.partial(ensemble.label_pred_ensemble_softmax,
                                      ensemble_size=ensemble_size)
    return train_loss_fn, eval_loss_fn, label_pred_fn
  else:
    raise ValueError(f'Unknown loss: {name!r}')


def get_train_steps_and_epochs(
    *,
    train_batch_size: int,
    train_examples: int,
    train_steps: Optional[int] = None,
    train_epochs: Optional[Union[int, float]] = None,
) -> Tuple[int, float]:
  """Returns number of train steps and epochs."""
  if not train_steps and not train_epochs:
    raise ValueError('You must specify either `train_steps` or `train_epochs`')
  if train_steps is not None and train_epochs is not None:
    raise ValueError('You must specify either `train_steps` or `train_epochs`, '
                     f'but not both: train_steps = {train_steps!r}, '
                     f'train_epochs = {train_epochs!r}')
  if not train_steps:
    train_steps = int(np.ceil(train_epochs * train_examples / train_batch_size))
  train_epochs = train_steps * train_batch_size / train_examples
  return train_steps, train_epochs


def initialize_train_state_from_checkpoint(
    *,
    train_state: TrainState,
    axis_resources: TrainStateAxisResources,
    name: str,
    **kwargs) -> TrainState:
  """Initializes a TrainState from a pre-trained checkpoint.

  This is useful for fine-tuning a pre-trained model, where typically only
  (a subset of) the parameters of the TrainState are initialized, while the
  other from-scratch attributes of the TrainState are kept (e.g. optimizer
  state).

  Args:
    train_state: TrainState to initialize from the checkpoint. The given object
      must not be used after the call to this function. Use the returned one
      instead.
    axis_resources: TrainStateAxisResources indicating how the `train_state` is
      partitioned.
    name: Name of the method used to initialize the train state.
    **kwargs: Additional arguments for the respective initialization function.

  Returns:
    A TrainState object.
  """
  if name == 'initialize_from_vmoe_release':
    return train_state.replace(
        params=initialization.initialize_from_vmoe_release(
            params=train_state.params,
            axis_resources=axis_resources.params,
            thread_pool=ThreadPool(kwargs.pop('num_threads', None)),
            **kwargs))
  elif name == 'initialize_from_vit':
    return train_state.replace(
        params=initialization.initialize_from_vit(
            params=train_state.params,
            axis_resources=axis_resources.params,
            **kwargs))
  else:
    raise ValueError(f'Unknown initialization method: {name!r}')


def parse_partition_spec(spec) -> PartitionSpec:
  if isinstance(spec, PartitionSpec):
    return spec
  if not spec:
    return PartitionSpec()
  spec = (spec,) if isinstance(spec, str) else tuple(spec)
  return PartitionSpec(*spec)


def train_step(
    state: TrainState,
    images: Array,
    labels: Array,
    loss_fn: Callable[[Array, Array], Array],
) -> Tuple[TrainState, Mapping[str, Any]]:
  """Performs one update step of the given TrainState object ."""
  rngs, next_rngs = utils.tree_rngs_split(state.rngs)

  @functools.partial(jax.grad, has_aux=True)
  def compute_grads_and_metrics(params):
    logits, metrics = state.apply_fn({'params': params}, images, rngs=rngs)
    metrics = dict(**metrics)
    metrics['main_loss'] = jnp.mean(loss_fn(logits, labels))
    metrics = jax.tree_map(jnp.mean, metrics)
    total_loss = metrics['main_loss'] + metrics.get('auxiliary_loss', 0.0)
    metrics['total_loss'] = total_loss
    return total_loss, metrics

  grads, metrics = compute_grads_and_metrics(state.params)
  return state.apply_gradients(grads=grads, rngs=next_rngs), metrics


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Trains a model and evaluates it periodically."""
  datasets = input_pipeline.get_datasets(config.dataset)
  if 'train' not in datasets:
    raise KeyError(f'You must have a "train" variant of the dataset. '
                   f'Available variants are {sorted(datasets.keys())!r}')
  train_examples = input_pipeline.get_data_num_examples(config.dataset.train)
  train_batch_size = config.dataset.train.batch_size
  train_steps, train_epochs = get_train_steps_and_epochs(
      train_steps=config.get('train_steps'),
      train_epochs=config.get('train_epochs'),
      train_batch_size=train_batch_size,
      train_examples=train_examples)
  logging.info(
      'Training for %d steps (%g epochs) over %d examples, with a '
      'batch size of %d', train_steps, train_epochs, train_examples,
      train_batch_size)

  # Set logical device mesh globally.
  mesh = partitioning.get_auto_logical_mesh(config.num_expert_partitions,
                                            jax.devices())
  partitioning.log_logical_mesh(mesh)
  maps.thread_resources.env = maps.thread_resources.env.with_mesh(mesh)

  # Get the global shape of the image array.
  train_image_shape = datasets['train'].element_spec['image'].shape
  train_image_shape = (train_batch_size,) + train_image_shape[1:]
  # Get the PartitionSpec for the inputs. By default, the first axis (batch)
  # is split among all axes of the logical device mesh, meaning that data is
  # fully partitioned, as usual.
  input_axis_resources = config.get('input_axis_resources', (mesh.axis_names,))
  input_axis_resources = parse_partition_spec(input_axis_resources)

  train_state_initialize_fn = create_train_state_initialize_from_scratch_fn(
      model=create_flax_model(config=config.model, deterministic=False),
      optimizer_config=config.optimizer,
      input_shape=train_image_shape,
      input_axis_resources=input_axis_resources,
      train_steps=train_steps)
  train_state_rngs = evaluator.make_rngs(
      ('params',) + tuple(config.get('extra_rng_keys', [])),
      config.get('seed', 0))
  train_state_axis_resources = tree_axis_resources_from_regexes(
      tree=jax.eval_shape(train_state_initialize_fn, train_state_rngs),
      axis_resources_regexes=config.params_axis_resources)
  train_state = restore_or_create_train_state(
      prefix=os.path.join(workdir, 'ckpt'),
      initialize_fn=train_state_initialize_fn,
      axis_resources=train_state_axis_resources,
      rngs=train_state_rngs,
      thread_pool=ThreadPool())
  init_step = int(train_state.step)
  if init_step == 0 and config.get('initialization'):
    train_state = initialize_train_state_from_checkpoint(
        train_state=train_state,
        axis_resources=train_state_axis_resources,
        **config.initialization)

  train_loss_fn, eval_loss_fn, label_pred_fn = get_loss_fn(**config.loss)
  train_step_pjit = pjit.pjit(
      fun=functools.partial(train_step, loss_fn=train_loss_fn),
      in_axis_resources=(
          train_state_axis_resources,  # train_state
          input_axis_resources,        # images
          input_axis_resources         # labels
      ),
      out_axis_resources=(
          train_state_axis_resources,  # train_state
          None,                        # metrics
      ),
      donate_argnums=(0, 1, 2))

  # Setup metric writer & hooks.
  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() > 0)
  profile_hook = create_profile_hook(
      workdir=workdir, **config.get('profile', {}))
  progress_hook = create_progress_hook(
      writer=writer, first_step=init_step + 1, train_steps=train_steps,
      **config.get('report_progress', {}))
  checkpoint_hook = create_checkpoint_hook(
      workdir=workdir, progress_hook=progress_hook,
      train_state_axis_resources=train_state_axis_resources,
      train_steps=train_steps, **config.get('save_checkpoint', {}))
  evaluation_hook = create_evaluation_hook(
      writer=writer,
      progress_hook=progress_hook,
      datasets={name: ds for name, ds in datasets.items() if name != 'train'},
      apply_fn=create_flax_model(config=config.model, deterministic=True).apply,
      loss_fn=eval_loss_fn,
      label_pred_fn=label_pred_fn,
      params_axis_resources=train_state_axis_resources.params,
      input_axis_resources=input_axis_resources,
      first_step=init_step + 1,
      train_steps=train_steps,
      extra_rng_keys=config.get('extra_rng_keys', []),
      **config.get('evaluate', {}))
  with metric_writers.ensure_flushes(writer):
    # Explicitly compile train_step here and report the compilation time.
    t0 = time.time()
    train_step_pjit = train_step_pjit.lower(*utils.tree_shape_dtype_struct((
        train_state,
        datasets['train'].element_spec['image'],
        datasets['train'].element_spec['labels']))).compile()
    t1 = time.time()
    writer.write_scalars(init_step + 1, {'train/compile_secs': t1 - t0})
    # Create iterator over the train dataset.
    tr_iter = pjit_utils.prefetch_to_device(
        iterator=input_pipeline.make_dataset_iterator(datasets['train']),
        axis_resources={'image': input_axis_resources,
                        'labels': input_axis_resources},
        size=config.dataset.train.get('prefetch_device'))
    for step, batch in zip(range(init_step + 1, train_steps + 1), tr_iter):
      profile_hook(step)
      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        train_state, metrics = train_step_pjit(train_state, batch['image'],
                                               batch['labels'])
      progress_hook(step, scalar_metrics=metrics)
      checkpoint_hook(step, state=train_state)
      evaluation_hook(step, params=train_state.params)
  # Write a checkpoint containing only the params, with no TrainState, to use
  # for fine-tuning or releasing it.
  logging.info('Saving checkpoint ready for releasing and fine-tuning.')
  checkpoints_partitioned.save_checkpoint(
      prefix=os.path.join(workdir, 'release_ckpt'),
      tree=train_state.params,
      axis_resources=train_state_axis_resources.params,
      num_shards=config.get('save_checkpoint', {}).get('num_shards', 0),
      makedirs=False,
      thread_pool=config.get('save_checkpoint', {}).get('num_threads')).wait()
  multihost_utils.sync_devices('checkpoints:release')
  logging.info('Training completed.')


def tree_axis_resources_from_regexes(
    *,
    tree: PyTree,
    axis_resources_regexes: AxisResourcesRegexes,
) -> PyTree:
  """Creates a PyTree with PartitionSpec leaves.

  Examples:
    >>> tree = {
    >>>   'dense': {'kernel': np.zeros((5, 10))},
    >>>   'moe': {
    >>>     'kernel': np.zeros((32, 10, 10)),
    >>>     'router': np.zeros((10, 32)),
    >>>   },
    >>> }
    >>> axis_resources_regexes = [
    >>>   ('.*/moe/kernel', ('expert',))
    >>> ]
    >>> resources = tree_axis_resources_from_regexes(
    >>>   tree=tree, axis_resources_regexes=axis_resources_regexes)
    >>> print(resources)
    {
      'dense': {'kernel': PartitionSpec()},
      'moe': {
        'kernel': PartitionSpec(('expert',)),
        'router': PartitionSpec(),
      },
    }

  Args:
    tree: A serializable PyTree (e.g. FrozenDict, TrainState, ...).
    axis_resources_regexes: A sequence of tuples (regex, spec_tuple), where
      `regex` is a Python regular expression to match the keys from the tree,
      and `spec_tuple` is a tuple of strings, or tuple of tuples of strings.

  Returns:
    A PyTree with the same structure as `tree`, with PartitionSpec leaves.
  """
  axis_resources_regexes = tuple(
      (re.compile(regex), parse_partition_spec(spec))
      for regex, spec in axis_resources_regexes)

  def search_partition_spec(key: str) -> PartitionSpec:
    for regex, partition_spec in axis_resources_regexes:
      if regex.search(key):
        return partition_spec
    return PartitionSpec()

  # NOTE: We use flax.serialization.to_state_dict to convert an arbitrary PyTree
  # to a dict, so that we can flatten it's structure using
  # flax.traverse_util.flatten_dict. This is not bulletproof, but works for our
  # standard cases (`tree` is a dict, or a TrainState).
  empty_node = flax.traverse_util.empty_node
  flat_tree_dict = flax.traverse_util.flatten_dict(
      flax.serialization.to_state_dict(tree), keep_empty_nodes=True)
  axis_resources = {
      k: empty_node if v is empty_node else search_partition_spec('/'.join(k))
      for k, v in flat_tree_dict.items()
  }
  axis_resources = flax.traverse_util.unflatten_dict(axis_resources)
  axis_resources = flax.serialization.from_state_dict(tree, axis_resources)
  return axis_resources
