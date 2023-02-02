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

"""Classes and functions used for training (from-scratch and fine-tuning)."""
import functools
import multiprocessing.pool
import os
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
from jax._src import sharding
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
from vmoe.evaluate import fewshot
from vmoe.nn import models
from vmoe.projects.adversarial_attacks import attacks as adversarial_attacks
from vmoe.train import initialization
from vmoe.train import optimizer
from vmoe.train import periodic_actions as train_periodic_actions
from vmoe.train import train_state as train_state_module
from vmoe.train import tree_summarizer


Array = jax.numpy.ndarray
ArraySpecDict = input_pipeline.ArraySpecDict
AsyncResult = multiprocessing.pool.AsyncResult
DatasetIterator = input_pipeline.DatasetIterator
Mesh = partitioning.Mesh
NamedSharding = sharding.NamedSharding
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
TrainStepFn = train_state_module.TrainStepFn
TreeSummarizer = tree_summarizer.TreeSummarizer


_getattr = getattr  # Alias of _getattr so that we can mock it in tests.

# pylint: disable=protected-access
_PositionalSemantics = maps._PositionalSemantics
_prepare_axis_resources = pjit._prepare_axis_resources
# pylint: enable=protected-access


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


def create_evaluation_hook(
    *,
    base_model_config: ml_collections.ConfigDict,
    writer: metric_writers.MetricWriter,
    progress_hook: ReportProgress,
    datasets: Mapping[str, DatasetIterator],
    loss_fn: Callable[[Array, Array], Array],
    params_axis_resources: PyTree,
    input_axis_resources: PartitionSpec,
    first_step: int,
    train_steps: int,
    extra_rng_keys: Sequence[str],
    model_overrides: Optional[ml_collections.ConfigDict] = None,
    **kwargs,
) -> Tuple[evaluator.EvaluateMultipleDatasets, ml_collections.ConfigDict]:
  """Returns a hook to evaluate a model periodically."""
  model_config = override_base_config(base_model_config, model_overrides)
  apply_fn = create_flax_model(
      config=model_config.to_dict(), deterministic=True).apply
  on_steps = set(kwargs.pop('on_steps', []))
  # Always evaluate on the first and last step.
  on_steps.update([first_step, train_steps])
  periodic_action = evaluator.EvaluateMultipleDatasets(
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
  return periodic_action, model_config


def create_fewshot_hook(
    *,
    base_model_config: ml_collections.ConfigDict,
    writer: metric_writers.MetricWriter,
    progress_hook: ReportProgress,
    variables_axis_resources: PyTree,
    input_axis_resources: PartitionSpec,
    first_step: int,
    train_steps: int,
    extra_rng_keys: Sequence[str],
    model_overrides: Optional[ml_collections.ConfigDict] = None,
    **kwargs) -> Tuple[Callable[..., Any], ml_collections.ConfigDict]:
  """Returns a hook to run fewshot evaluation of a model periodically."""
  model_config = override_base_config(base_model_config, model_overrides)
  # Few-shot eval requires additional mandatory parameters. If none of those is
  # given, we assume that no few-shot eval should be done.
  if not kwargs:
    return (lambda *args, **kw: None, model_config)
  apply_fn = create_flax_model(
      config=model_config.to_dict(), deterministic=True).apply
  on_steps = set(kwargs.pop('on_steps', []))
  # Always evaluate on the first and last step.
  on_steps.update([first_step, train_steps])
  periodic_action = fewshot.FewShotPeriodicAction(
      metric_writer=writer,
      apply_fn=apply_fn,
      variables_axis_resources=variables_axis_resources,
      input_axis_resources=input_axis_resources,
      rng_keys=extra_rng_keys,
      report_progress=progress_hook,
      report_progress_name='fewshot',
      on_steps=on_steps,
      **kwargs)
  return periodic_action, model_config


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


def create_tree_summarizer(config) -> Optional[TreeSummarizer]:
  if not config:
    return None
  if isinstance(config, (list, tuple)):
    return TreeSummarizer(rules=config)
  else:
    return TreeSummarizer(**config)


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
      PartitionSpec leaves, indicating how the output TrainState is partitioned.
    rngs: Dictionary of PRNGs to use during model initialization.
    mesh: Logical mesh used by pjit. If None, uses the currently active mesh.

  Returns:
    A TrainState.
  """
  mesh = mesh or maps.thread_resources.env.physical_mesh
  with maps.Mesh(mesh.devices, mesh.axis_names):
    return pjit.pjit(
        initialize_fn,
        in_axis_resources=(None,),
        out_axis_resources=axis_resources)(rngs)


def create_or_reuse_train_state(
    *,
    initialize_fn: TrainStateInitFromScratchFn,
    axis_resources: TrainStateAxisResources,
    rngs: Mapping[str, PRNGKey],
    reuse_train_state: TrainState,
    mesh: Optional[Mesh] = None,
) -> TrainState:
  """Creates a TrainState object initialized from scratch, re-using some parts.

  Args:
    initialize_fn: Function used to create and initialize a train state from
      scratch.
    axis_resources: A PyTree with the same structure as a TrainState, but with
      PartitionSpec leaves, indicating how the output TrainState is partitioned.
    rngs: Dictionary of PRNGs to use during model initialization.
    reuse_train_state: TrainState containing the arrays to re-use. All arrays
      with a size other than 0 are re-used.
    mesh: Logical mesh used by pjit. If None, uses the currently active mesh.

  Returns:
    A TrainState.
  """
  mesh = mesh or maps.thread_resources.env.physical_mesh
  out_axis_resources = axis_resources
  # Replace ShapeDtypeStruct objects with Numpy arrays of size 0.
  # pylint: disable=g-long-lambda
  reuse_train_state = jax.tree_map(
      lambda x: np.array([], dtype=x.dtype)
      if isinstance(x, jax.ShapeDtypeStruct) else np.asarray(x),
      reuse_train_state)
  # pylint: enable=g-long-lambda
  # Replace PartitionSpec of arrays with size = 0. These are not partitioned.
  in_axis_resources = jax.tree_map(
      lambda x, r: pjit.PartitionSpec() if x.size == 0 else r,
      reuse_train_state, axis_resources)
  # Wrap the given initialize_fn with a new one that selects the arrays in the
  # reuse_train_state or newly created arrays based on the size of the first.
  # We leverage the fact that XLA will remove unused variables/ops when this
  # new initialize function is compiled.
  def initialize(reuse_train_state: TrainState, rngs: Mapping[str, PRNGKey]):
    train_state = initialize_fn(rngs)
    return jax.tree_map(
        lambda a, b: b if a.size == 0 else a, reuse_train_state, train_state)

  with maps.Mesh(mesh.devices, mesh.axis_names):
    return pjit.pjit(
        initialize,
        in_axis_resources=(in_axis_resources, None),
        out_axis_resources=out_axis_resources,
        donate_argnums=(0,))(reuse_train_state, rngs)


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
    initialization_kwargs: Optional[Mapping[str, Any]] = None,
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
    initialization_kwargs: Optional dictionary containing the kwargs used to
      initialize the TrainState from an existing checkpoint.

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
    with maps.Mesh(mesh.devices, mesh.axis_names):
      return pjit.pjit(
          fun=lambda x: x,
          in_axis_resources=(axis_resources,),
          out_axis_resources=axis_resources,
          donate_argnums=(0,))(train_state)
  if initialization_kwargs:
    # Compute global shapes of the TrainState and convert them to local shapes.
    train_state = jax.eval_shape(initialize_fn, rngs)
    train_state = tree_global_to_local_shape(train_state, axis_resources, mesh)
    # Initialize TrainState from checkpoint. Arrays that are not initialized,
    # are ShapeDtypeStruct objects.
    train_state = initialize_train_state_from_checkpoint(
        train_state=train_state,
        axis_resources=axis_resources,
        **initialization_kwargs)
    return create_or_reuse_train_state(
        initialize_fn=initialize_fn, axis_resources=axis_resources, rngs=rngs,
        mesh=mesh, reuse_train_state=train_state)
  # Otherwise, create a new train state from scratch on device.
  return create_train_state(
      initialize_fn=initialize_fn, axis_resources=axis_resources, rngs=rngs,
      mesh=mesh)


def get_loss_fn(name: str, **kwargs):
  """Returns the train/evaluation losses and the way to predict labels."""
  def default_sigmoid_xent(logits, labels, **kw):
    return jnp.sum(
        optimizer.optax.sigmoid_binary_cross_entropy(logits, labels, **kw),
        axis=-1)

  default_label_pred_fn = lambda logits: jnp.argmax(logits, -1)
  loss_fns = {
      'softmax_xent': (
          optimizer.optax.softmax_cross_entropy,
          optimizer.optax.softmax_cross_entropy,
          default_label_pred_fn),
      'sigmoid_xent':
          (default_sigmoid_xent, default_sigmoid_xent, default_label_pred_fn),
      # In the case of the ensemble cross-entropy, the training loss and the
      # eval loss functions differ (see the discussion in Appendix D.1 of
      # https://arxiv.org/pdf/2110.03360.pdf).
      'ensemble_softmax_xent': (
          ensemble.ensemble_softmax_xent_train,
          ensemble.ensemble_softmax_xent_eval,
          ensemble.label_pred_ensemble_softmax),
      'ensemble_sigmoid_xent': (
          ensemble.ensemble_sigmoid_xent_train,
          ensemble.ensemble_sigmoid_xent_eval,
          ensemble.label_pred_ensemble_sigmoid),
  }
  if name in loss_fns:
    train_loss_fn, eval_loss_fn, pred_fn = loss_fns[name]
    train_loss_fn = functools.partial(train_loss_fn, **kwargs)
    eval_loss_fn = functools.partial(eval_loss_fn, **kwargs)
    if name.startswith('ensemble_'):
      # Pass ensemble_size to the pred_fn.
      pred_fn = functools.partial(pred_fn, **kwargs)
    return train_loss_fn, eval_loss_fn, pred_fn
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
            **kwargs))
  else:
    raise ValueError(f'Unknown initialization method: {name!r}')


def mixup(
    rng: PRNGKey,
    tree: PyTree,
    *,
    concentration: float,
    shape: Tuple[int, ...] = (1, 2),
    roll_axis: int = 0,
    partition_spec: Optional[PartitionSpec] = None,
) -> Tuple[PRNGKey, PyTree]:
  """Performs mixup on each array of the given tree.

  For details on mixup, check "mixup: Beyond Empirical Risk Minimization"
  (https://arxiv.org/abs/1710.09412).

  The same mixing weights are used for all leaves of the tree, which is
  convenient to mix several arrays in the same way (e.g. images and labels).

  Args:
    rng: PRNGKey used to generate the mixup weights.
    tree: Tree with array leaves. The size of the first dimension of all trees
      must be equal, i.e. the batch_size.
    concentration: Dirichlet concentration parameter, used to sample the
      mixing weights. Must be a positive float.
    shape: Shape of the samples from the Dirichlet distribution.
    roll_axis: Axis to roll to mix examples. This is typically the 'batch'
      axis from your input arrays. It must be [0, len(shape) - 1).
    partition_spec: Optional PartitionSpec used to annotate the sampled values
      from the Dirichlet distribution.

  Returns:
    A tree with the mixed arrays.
  """
  arrays, treedef = jax.tree_flatten(tree)
  if len(shape) < 2:
    raise ValueError(f"Mixup 'shape' has length {len(shape)}, but it must have "
                     'length >= 2.')
  # Check that all arrays have the same batch size.
  batch_ndim = len(shape) - 1
  for i, x in enumerate(arrays):
    try:
      _ = jnp.broadcast_shapes(x.shape[:batch_ndim], shape[:-1])
    except ValueError:
      raise ValueError(f'Mixup with inconsistent shapes. The shape of the '
                       f'{i}-th array is {x.shape}, but the first {batch_ndim} '
                       f'dims must be broadcastable to {shape[:-1]}') from None
    if x.shape[:batch_ndim] != arrays[0].shape[:batch_ndim]:
      raise ValueError(f'Mixup with inconsistent shapes. The shape of the '
                       f'{i}-th array is {x.shape}, but the first {batch_ndim} '
                       f'dims must be equal to {arrays[0].shape[:batch_ndim]}')
  if not 0 <= roll_axis < batch_ndim:
    raise ValueError("Mixup 'roll_axis' must be an integer in "
                     f'[0, {batch_ndim}), but got roll_axis = {roll_axis}')
  # Check mixup parameters.
  if concentration <= 0.:
    raise ValueError(f"Mixup 'concentration' must be greater than 0, but got "
                     f'concentration = {concentration}')
  # Generate alphas (weights) for the mixup.
  concentration = jnp.full(shape, concentration)
  concentration = partitioning.with_sharding_constraint(
      concentration, partition_spec)
  alpha = jax.random.dirichlet(rng, concentration)
  alpha = partitioning.with_sharding_constraint(alpha, partition_spec)
  # Put the largest weight of each example in the first position of alpha.
  # This avoids destroying examples due to permuations of the weights.
  # If one samples an alpha for every example, then chances are that many of
  # the original examples are lost (with mixup_size = 2).
  # Suppose that we have concentration approaching 0, and mixup_size = 2.
  # Then each alpha (Dirichlet sample) will be either (1, 0) or (0, 1).
  # If the batch is [1, 2], with alpha [(1, 0), (0, 1)], then the output will
  # be [1, 1]. If the sampled alpha is [(0, 1), (1, 0)], the output is [2, 2].
  alpha = -jnp.sort(-alpha, axis=-1)
  # Reshape alphas to (mixup_size, batch_size, ...) for convenience.
  alpha = jnp.moveaxis(alpha, -1, 0)
  # Mix each sample with the next `mixup_size` samples within the same group.
  def mix(x):
    a = alpha.reshape(alpha.shape + (1,) * (x.ndim - batch_ndim))
    return sum(a[i] * jnp.roll(x, -i, axis=roll_axis) for i in range(shape[-1]))
  arrays = list(map(mix, arrays))
  return jax.tree_unflatten(treedef, arrays)


def override_base_config(
    base: ml_collections.ConfigDict,
    override: Optional[ml_collections.ConfigDict],
) -> ml_collections.ConfigDict:
  output = base.copy_and_resolve_references()
  with output.unlocked():
    if override:
      output.update(override)
  return output


def train_step(
    state: TrainState,
    images: Array,
    labels: Array,
    loss_fn: Callable[[Array, Array], Array],
    summarizer: Optional[TreeSummarizer] = None,
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
  # Update train state.
  state = state.apply_gradients(grads=grads, rngs=next_rngs)

  if summarizer:
    # Summarize arrays in the gradients tree or the train state.
    state_flat = flax.traverse_util.flatten_dict(
        flax.serialization.to_state_dict(state), sep='/')
    state_flat['params_grads'] = flax.traverse_util.flatten_dict(grads, sep='/')
    metrics.update(summarizer(state_flat))

  return state, metrics


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  # Set logical device mesh globally.
  mesh = partitioning.get_auto_logical_mesh(config.num_expert_partitions,
                                            jax.devices())
  partitioning.log_logical_mesh(mesh)
  with mesh:
    return _train_and_evaluate(config, workdir, mesh)


def _train_and_evaluate(config: ml_collections.ConfigDict, workdir: str,
                        mesh: Mesh):
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

  # Get the global shape of the image array.
  element_spec: ArraySpecDict = datasets['train'].element_spec  # pytype: disable=annotation-type-mismatch
  train_image_shape = tuple(element_spec['image'].shape)
  train_image_shape = (train_batch_size,) + train_image_shape[1:]
  # Get the PartitionSpec for the inputs. By default, the first axis (batch)
  # is split among all axes of the logical device mesh, meaning that data is
  # fully partitioned, as usual.
  input_axis_resources = config.get('input_axis_resources', (mesh.axis_names,))
  input_axis_resources = partitioning.parse_partition_spec(input_axis_resources)

  train_state_initialize_fn = create_train_state_initialize_from_scratch_fn(
      model=create_flax_model(config=config.model, deterministic=False),
      optimizer_config=config.optimizer,
      input_shape=train_image_shape,
      input_axis_resources=input_axis_resources,
      train_steps=train_steps)
  train_state_rngs = utils.make_rngs(
      ('params',) + tuple(config.get('extra_rng_keys', [])),
      config.get('seed', 0))
  train_state_axis_resources = partitioning.tree_axis_resources_from_regexes(
      tree=jax.eval_shape(train_state_initialize_fn, train_state_rngs),
      axis_resources_regexes=config.params_axis_resources)
  train_state = restore_or_create_train_state(
      prefix=os.path.join(workdir, 'ckpt'),
      initialize_fn=train_state_initialize_fn,
      axis_resources=train_state_axis_resources,
      rngs=train_state_rngs,
      thread_pool=ThreadPool(),
      initialization_kwargs=config.get('initialization'))
  init_step = int(train_state.step)
  train_loss_fn, eval_loss_fn, label_pred_fn = get_loss_fn(**config.loss)
  summarizer = create_tree_summarizer(config.get('summarize_arrays'))
  train_step_fn = functools.partial(
      train_step,
      loss_fn=train_loss_fn,
      summarizer=summarizer)
  if config.get('adversarial', {}):
    adversarial_config = config.adversarial.to_dict()
    train_step_fn = wrap_train_step_with_adversarial_attack(
        train_step_fn, train_loss_fn, **adversarial_config)
  # If mixup options are defined, wrap the train_step_fn with mixup.
  if config.get('mixup', {}):
    mixup_config = config.mixup.to_dict()
    train_step_fn = wrap_train_step_with_mixup(
        train_step_fn, partition_spec=input_axis_resources, **mixup_config)

  train_step_pjit = pjit.pjit(
      fun=train_step_fn,
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
  evaluation_hook, config_model_eval = create_evaluation_hook(
      base_model_config=config.model.copy_and_resolve_references(),
      writer=writer,
      progress_hook=progress_hook,
      datasets={name: ds for name, ds in datasets.items() if name != 'train'},
      loss_fn=eval_loss_fn,
      label_pred_fn=label_pred_fn,
      params_axis_resources=train_state_axis_resources.params,
      input_axis_resources=input_axis_resources,
      first_step=init_step + 1,
      train_steps=train_steps,
      extra_rng_keys=config.get('extra_rng_keys', []),
      **config.get('evaluate', {}))
  fewshot_hook, _ = create_fewshot_hook(
      base_model_config=config_model_eval,
      writer=writer,
      progress_hook=progress_hook,
      variables_axis_resources={'params': train_state_axis_resources.params},
      input_axis_resources=input_axis_resources,
      first_step=init_step + 1,
      train_steps=train_steps,
      extra_rng_keys=config.get('extra_rng_keys', []),
      **config.get('fewshot', {}))
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
        iterator=datasets['train'],
        axis_resources={'image': input_axis_resources,
                        'labels': input_axis_resources},
        size=config.dataset.train.get('prefetch_device'))
    for step, batch in zip(range(init_step + 1, train_steps + 1), tr_iter):
      profile_hook(step)
      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        train_state, metrics = train_step_pjit(train_state, batch['image'],
                                               batch['labels'])
      progress_hook(
          step, scalar_metrics={f'train/{k}': v for k, v in metrics.items()})
      checkpoint_hook(step, state=train_state)
      evaluation_hook(step, params=train_state.params)
      fewshot_hook(step, variables={'params': train_state.params})
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


def tree_global_to_local_shape(tree, axis_resources, mesh):
  """Maps a tree from global to local shapes."""
  # Note: This requires some low-level understanding about pjit. Replace this
  # if we switch to Global Device Array (GDA) in the future.
  # See more information in checkpoints/partitioned.py.
  leaves, struct_tree = jax.tree_flatten(tree)
  axis_resources, struct_axis_resources = jax.tree_flatten(axis_resources)
  if struct_tree != struct_axis_resources:
    raise ValueError(f'The tree structs do not match.\n'
                     f'index: {struct_tree}\n'
                     f'axis_resources: {struct_axis_resources}')
  global_shapes = [jax.ShapedArray(x.shape, x.dtype) for x in leaves]
  positional_semantics = [_PositionalSemantics.LOCAL for _ in global_shapes]
  shardings = [NamedSharding(mesh, spec) for spec in axis_resources]
  shardings = [
      pjit.to_op_sharding_sharding(s, a.ndim)
      for a, s in zip(global_shapes, shardings)
  ]
  local_shapes = pjit.global_to_local(positional_semantics, global_shapes,
                                      shardings, mesh)
  return struct_tree.unflatten([
      jax.ShapeDtypeStruct(shape=s.shape, dtype=s.dtype)
      for s in local_shapes
  ])


def wrap_train_step_with_adversarial_attack(
    train_step_fn: TrainStepFn,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    *,
    attack_auxiliary_loss: bool,
    max_epsilon: float,
    num_updates: int,
) -> TrainStepFn:
  """Wraps a train_step_fn to perform PGD adversarial training.

  Arguments:
    train_step_fn: Training step function to wrap.
    loss_fn: Loss function to attack.
    attack_auxiliary_loss: Whether to attack the MoE auxiliary loss or not.
    max_epsilon: Maximum change for each pixel.
    num_updates: Number of PGD updates to perform in the attack.

  Returns:
    A new train_step_fn.
  """
  if num_updates <= 0 or max_epsilon <= 0.0:
    return train_step_fn

  def loss_with_auxiliary_attack_fn(logits, labels, metrics):
    loss = loss_fn(logits, labels)
    if attack_auxiliary_loss:
      loss = loss + jnp.mean(metrics['auxiliary_loss'])
    return loss

  def new_train_step_fn(
      state: TrainState,
      images: jnp.ndarray,
      labels: jnp.ndarray,
      *args,
      **kwargs,
  ) -> Tuple[TrainState, Mapping[str, Any]]:
    def apply_fn(x, **kwargs):
      return state.apply_fn({'params': state.params}, x, **kwargs)
    images, rngs = adversarial_attacks.stateless_attack_pgd(
        images=images, labels=labels, rngs=state.rngs, apply_fn=apply_fn,
        loss_fn=loss_with_auxiliary_attack_fn, max_epsilon=max_epsilon,
        num_updates=num_updates)
    state = state.replace(rngs=rngs)
    return train_step_fn(state, images, labels, *args, **kwargs)

  return new_train_step_fn


def wrap_train_step_with_mixup(
    train_step_fn: TrainStepFn,
    *,
    concentration: Optional[float] = None,
    granularity: str = 'device',
    size: int = 2,
    partition_spec: Optional[PartitionSpec] = None,
) -> TrainStepFn:
  """Wraps a train step function with mixup.

  Args:
    train_step_fn: Training step function to wrap.
    concentration: Dirichlet concentration parameter, used to sample the
      mixing weights. Must be a positive float.
    granularity: Granularity of the mixing weights. 'batch' samples a single set
      of weights for all the examples in the batch, 'device' samples a set of
      weights for all the examples in each device, and 'example' samples weights
      for each example independently.
    size: Number of original examples used in each new mixed example.
    partition_spec: Optional PartitionSpec used to annotate the sampled values
      from the Dirichlet distribution. Only used if granularity is 'example'.

  Returns:
    A new train_step_fn that does mixup before calling the original one.
  """
  if concentration is None or concentration == 0.:
    return train_step_fn

  if granularity == 'batch':
    partition_spec = None

  if granularity == 'example':
    logging.warning(
        "You are using granularity = 'example'. This is extremely inneficient "
        'since the current implementation of jax.random.gamma does not work '
        "with pjit's sharding. Use granularity = 'device' meanwhile.")

  def new_train_step_fn(
      state: TrainState,
      images: Array,
      labels: Array,
      *args,
      **kwargs,
  ) -> Tuple[TrainState, Mapping[str, Any]]:
    if 'mixup' not in state.rngs:
      raise ValueError(f'Mixup requires a PRNGKey with the name "mixup", but '
                       f'only these were given: {sorted(state.rngs.keys())}')
    frozen_rngs = isinstance(state.rngs, flax.core.FrozenDict)
    rngs = flax.core.unfreeze(state.rngs) if frozen_rngs else state.rngs
    # Apply mixup on images and labels.
    mixup_rng = rngs.pop('mixup')
    mixup_rng, next_mixup_rng = jax.random.split(mixup_rng)
    # Reshape images and labels to perform mixup on each device independently,
    # to avoid communication among devices due to mixup's jnp.roll().
    images = images.reshape((jax.device_count(), -1) + images.shape[1:])
    labels = labels.reshape((jax.device_count(), -1) + labels.shape[1:])
    if granularity == 'batch':
      mixup_shape = (1, 1, size)
    elif granularity == 'device':
      mixup_shape = (images.shape[0], 1, size)
    elif granularity == 'example':
      mixup_shape = (*images.shape[:2], size)
    else:
      raise ValueError(f'Unknown granularity: {granularity!r}')
    (images, labels) = mixup(
        rng=mixup_rng, tree=(images, labels), concentration=concentration,
        shape=mixup_shape, roll_axis=1, partition_spec=partition_spec)
    images = images.reshape((-1,) + images.shape[2:])
    labels = labels.reshape((-1,) + labels.shape[2:])
    # Call the given `train_step_fn` passing the state without mixup rngs, and
    # the modified images and labels.
    state = state.replace(rngs=flax.core.freeze(rngs) if frozen_rngs else rngs)
    state, metrics = train_step_fn(state, images, labels, *args, **kwargs)
    # Add the mixup PRNGKey back to the state.
    rngs = flax.core.unfreeze(state.rngs) if frozen_rngs else state.rngs
    rngs['mixup'] = next_mixup_rng
    state = state.replace(rngs=flax.core.freeze(rngs) if frozen_rngs else rngs)
    return state, metrics

  return new_train_step_fn
