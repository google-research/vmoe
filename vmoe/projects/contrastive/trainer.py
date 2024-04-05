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

"""Classes and functions used for training (from-scratch and fine-tuning)."""
import functools
import multiprocessing.pool
import os
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from absl import logging
from clu import metric_writers
import flax
import flax.serialization
import flax.training.train_state
import flax.traverse_util
import jax
import jax.numpy as jnp
import ml_collections
import tensorflow as tf
from vmoe import multihost_utils
from vmoe import partitioning
from vmoe import utils
from vmoe.data import input_pipeline
from vmoe.data import pjit_utils
from vmoe.evaluate import fewshot
from vmoe.projects.contrastive import evaluators
from vmoe.train import periodic_actions as train_periodic_actions
from vmoe.train import train_state as train_state_module
from vmoe.train import trainer
from vmoe.train import tree_summarizer


Array = jax.numpy.ndarray
DatasetIterator = input_pipeline.DatasetIterator
Mesh = partitioning.Mesh
ReportProgress = train_periodic_actions.ReportProgress
ThreadPool = multiprocessing.pool.ThreadPool
TrainState = train_state_module.TrainState
TreeSummarizer = tree_summarizer.TreeSummarizer

accumulate_gradients_and_metrics = trainer.accumulate_gradients_and_metrics
create_checkpoint_manager = trainer.create_checkpoint_manager
create_flax_model = trainer.create_flax_model
create_profile_hook = trainer.create_profile_hook
create_progress_hook = trainer.create_progress_hook
create_tree_summarizer = trainer.create_tree_summarizer
get_dataset_iterator = trainer.get_dataset_iterator
get_train_steps_and_epochs = trainer.get_train_steps_and_epochs
make_create_train_state_fn = trainer.make_create_train_state_fn
make_train_cost_fn = trainer.make_train_cost_fn
override_base_config = trainer.override_base_config
restore_or_create_train_state = trainer.restore_or_create_train_state


def create_fewshot_hook(
    *,
    base_model_config: ml_collections.ConfigDict,
    writer: metric_writers.MetricWriter,
    progress_hook: ReportProgress,
    first_step: int,
    train_steps: int,
    extra_rng_keys: Sequence[str],
    model_overrides: Optional[ml_collections.ConfigDict] = None,
    **kwargs) -> Callable[..., Any]:
  """Returns a hook to run fewshot evaluation of a model periodically."""
  model_config = override_base_config(base_model_config, model_overrides)
  # Few-shot eval requires additional mandatory parameters. If none of those is
  # given, we assume that no few-shot eval should be done.
  if not kwargs:
    return (lambda *args, **kw: None)
  model = create_flax_model(
      config=model_config.to_dict(), deterministic=True)
  # Apply function only embeds images.
  apply_fn = lambda p, x, **kw: model.apply(p, images=x, texts=None, **kw)
  on_steps = set(kwargs.pop('on_steps', []))
  # Always evaluate on the first and last step.
  on_steps.update([first_step, train_steps])
  periodic_action = fewshot.FewShotPeriodicAction(
      metric_writer=writer,
      apply_fn=apply_fn,
      rng_keys=extra_rng_keys,
      report_progress=progress_hook,
      report_progress_name='fewshot',
      on_steps=on_steps,
      **kwargs)
  return periodic_action


def create_retrieval_hook(
    *,
    base_model_config: ml_collections.ConfigDict,
    writer: metric_writers.MetricWriter,
    progress_hook: ReportProgress,
    first_step: int,
    train_steps: int,
    every_steps: Optional[int] = None,
    every_secs: Optional[int] = None,
    datasets: Optional[Mapping[str, Mapping[str, Any]]] = None,
    model_overrides: Optional[ml_collections.ConfigDict] = None,
    data_sharding: jax.sharding.NamedSharding,
    **kwargs) -> Callable[..., Any]:
  """Returns a hook to run retrieval evaluation of a model periodically."""
  model_config = override_base_config(base_model_config, model_overrides)
  model = create_flax_model(
      config=model_config.to_dict(), deterministic=True)
  # Always evaluate on the first and last step.
  on_steps = set(kwargs.pop('on_steps', []))
  on_steps.update([first_step, train_steps])

  # Make the apply_fn function conform with Big Vision's evaluator expected
  # inputs and outputs.
  def apply_fn(v, input_dict):
    img = input_dict.get('image')
    txt = input_dict.get('labels')
    if (img is None) == (txt is None):
      raise ValueError('One and only of images or text must be None.')
    z, _ = model.apply(v, images=img, texts=txt)
    return (None, z, None) if img is None else (z, None, None)

  datasets = datasets or {}
  if isinstance(datasets, ml_collections.ConfigDict):
    datasets = datasets.to_dict()
  try:
    # Instantiate hooks for each of the tasks to evaluate.
    hooks = [
        evaluators.RetrievalPeriodicAction(
            metric_writer=writer,
            apply_fn=apply_fn,
            task=task,
            data_sharding=data_sharding,
            every_steps=every_steps,
            every_secs=every_secs,
            on_steps=on_steps,
            report_progress=progress_hook,
            **kwargs,
            **bv_kw)
        for task, bv_kw in datasets.items()
    ]
    def periodic_action(*a, **kw):
      for hook in hooks:
        hook(*a, **kw)
    return periodic_action
  except NotImplementedError as e:
    logging.warning('%s', str(e))
    return (lambda *a, **kw: None)


def create_zeroshot_hook(
    *,
    base_model_config: ml_collections.ConfigDict,
    writer: metric_writers.MetricWriter,
    progress_hook: ReportProgress,
    first_step: int,
    train_steps: int,
    every_steps: Optional[int] = None,
    every_secs: Optional[int] = None,
    datasets: Optional[Mapping[str, Mapping[str, Any]]] = None,
    model_overrides: Optional[ml_collections.ConfigDict] = None,
    data_sharding: jax.sharding.NamedSharding,
    **kwargs) -> Callable[..., Any]:
  """Returns a hook to run zeroshot evaluation of a model periodically."""
  model_config = override_base_config(base_model_config, model_overrides)
  model = create_flax_model(
      config=model_config.to_dict(), deterministic=True)
  # Always evaluate on the first and last step.
  on_steps = set(kwargs.pop('on_steps', []))
  on_steps.update([first_step, train_steps])

  # Make the apply_fn function conform with Big Vision's evaluator expected
  # inputs and outputs.
  def apply_fn(v, input_dict):
    img = input_dict.get('image')
    txt = input_dict.get('labels')
    if (img is None) == (txt is None):
      raise ValueError('One and only of images or text must be None.')
    z, _ = model.apply(v, images=img, texts=txt)
    return (None, z, None) if img is None else (z, None, None)

  datasets = datasets or {}
  if isinstance(datasets, ml_collections.ConfigDict):
    datasets = datasets.to_dict()
  if not datasets:
    return (lambda *a, **kw: None)

  try:
    return evaluators.ZeroShotPeriodicAction(
        metric_writer=writer,
        apply_fn=apply_fn,
        data_sharding=data_sharding,
        every_steps=every_steps,
        every_secs=every_secs,
        on_steps=on_steps,
        report_progress=progress_hook,
        dataset_names=tuple(datasets.keys()),
        dataset_overrides=datasets,
        **kwargs)
  except NotImplementedError as e:
    logging.warning('%s', str(e))
    return (lambda *a, **kw: None)


def sigmoid_loss(logits: Array):
  if logits.ndim < 2 or logits.shape[-1] != logits.shape[-2]:
    raise ValueError(
        f'Last two dims of logits must be equal, but got {logits.shape=}')
  # SigLIP loss, as described in https://arxiv.org/pdf/2303.15343.pdf.
  # Positives are in the diagonal, negatives are off-diagonal.
  z = 2. * jnp.eye(logits.shape[-1], dtype=logits.dtype) - 1.
  log_lkh = jax.nn.log_sigmoid(jnp.einsum('...mn,mn->...mn', logits, z))
  # Normalize by npos per column, but that's one, so just sum.
  return -jnp.sum(log_lkh, axis=-1)


def train_step(
    state: TrainState,
    images: Array,
    texts: Array,
    loss_fn: Callable[[Array], Array],
    microsteps: Optional[int] = None,
    summarizer: Optional[TreeSummarizer] = None,
) -> Tuple[TrainState, Mapping[str, Any]]:
  """Performs one update step of the given TrainState object ."""

  @functools.partial(jax.grad, has_aux=True)
  def compute_grads_and_metrics(params, images, texts, rngs):
    rngs, next_rngs = utils.tree_rngs_split(rngs)
    logits, metrics = state.apply_fn(
        {'params': params}, images, texts, rngs=rngs)
    metrics = dict(**metrics)
    metrics['main_loss'] = jnp.mean(loss_fn(logits))
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    total_loss = metrics['main_loss'] + metrics.get('auxiliary_loss', 0.0)
    metrics['total_loss'] = total_loss
    return total_loss, (next_rngs, metrics)

  compute_grads_and_metrics = accumulate_gradients_and_metrics(
      compute_grads_and_metrics, microsteps)
  grads, (next_rngs, metrics) = compute_grads_and_metrics(
      state.params, images, texts, state.rngs)
  state, global_norms = state.apply_gradients_and_compute_global_norms(
      grads, rngs=next_rngs)
  metrics.update({f'global_norm/{k}': v for k, v in global_norms.items()})

  if summarizer:
    # Summarize arrays in the gradients tree or the train state.
    state_flat = flax.traverse_util.flatten_dict(
        flax.serialization.to_state_dict(state), sep='/')
    state_flat['params_grads'] = flax.traverse_util.flatten_dict(grads, sep='/')
    metrics.update(summarizer(state_flat))

  return state, metrics


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str,
                       mesh: Mesh, writer: metric_writers.MetricWriter):
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
  dataset_element_shape_dtype = pjit_utils.get_dataset_shape_dtype_struct(
      datasets['train'])

  ckpt_manager = create_checkpoint_manager(
      workdir=workdir, **config.get('save_checkpoint', {}))
  train_state_initialize_fn = make_create_train_state_fn(
      model=create_flax_model(config=config.model, deterministic=False),
      optimizer_config=config.optimizer,
      input_shape_dtypes=(dataset_element_shape_dtype['image'],
                          dataset_element_shape_dtype['text']),
      train_steps=train_steps,
      extra_rng_keys=tuple(config.get('extra_rng_keys', [])),
      seed=config.get('seed', 0))
  train_state, last_seen_index = restore_or_create_train_state(
      ckpt_manager=ckpt_manager,
      initialize_fn=train_state_initialize_fn,
      axis_resources_regexes=config.params_axis_resources,
      thread_pool=ThreadPool(),
      initialization_kwargs=config.get('initialization'))
  init_step = int(train_state.step)
  logging.info('Initial step = %d', init_step)
  tr_iter = get_dataset_iterator(
      dataset=datasets['train'],
      prefetch_size=config.dataset.train.get('prefetch_device', 1),
      mesh=mesh,
      last_seen_index=last_seen_index)
  summarizer = create_tree_summarizer(config.get('summarize_arrays'))
  train_step_fn = functools.partial(
      train_step,
      loss_fn=sigmoid_loss,
      microsteps=config.get('microsteps'),
      summarizer=summarizer)

  train_step_pjit = jax.jit(
      fun=train_step_fn,
      out_shardings=(
          jax.tree_util.tree_map(lambda x: x.sharding, train_state),
          None,
      ),
      donate_argnums=(0, 1, 2),
  )

  # Setup hooks.
  profile_hook = create_profile_hook(
      workdir=workdir, **config.get('profile', {}))
  progress_hook = create_progress_hook(
      writer=writer, first_step=init_step + 1, train_steps=train_steps,
      **config.get('report_progress', {}))
  fewshot_hook = create_fewshot_hook(
      base_model_config=config.model.copy_and_resolve_references(),
      writer=writer,
      progress_hook=progress_hook,
      first_step=init_step + 1,
      train_steps=train_steps,
      extra_rng_keys=config.get('extra_rng_keys', []),
      **config.get('fewshot', {}))
  retrieval_hook = create_retrieval_hook(
      data_sharding=dataset_element_shape_dtype['image'].sharding,
      base_model_config=config.model.copy_and_resolve_references(),
      writer=writer,
      progress_hook=progress_hook,
      first_step=init_step + 1,
      train_steps=train_steps,
      **config.get('retrieval', {}))
  zeroshot_hook = create_zeroshot_hook(
      data_sharding=dataset_element_shape_dtype['image'].sharding,
      base_model_config=config.model.copy_and_resolve_references(),
      writer=writer,
      progress_hook=progress_hook,
      first_step=init_step + 1,
      train_steps=train_steps,
      **config.get('zeroshot', {}))
  # Run checkpoint hook just before starting the loop. This will save the train
  # state at initialization.
  def _save_checkpoint(step, ts, it, force=False):
    last_seen_index = step * train_batch_size
    with progress_hook.timed('ckpt', wait_jax_async_dispatch=False):
      ckpt_manager.save(
          step,
          items={
              'state': ts,
              'dataset_iterator': {'last_seen_index': last_seen_index},
          },
          force=force)
  if init_step == 0 and not tf.io.gfile.exists(os.path.join(workdir, 'ckpt/0')):
    multihost_utils.sync_devices('training:ckpt-first')
    _save_checkpoint(init_step, train_state, tr_iter, force=True)
  # Explicitly compile train_step here and report the compilation time.
  t0 = time.time()
  train_step_pjit = train_step_pjit.lower(
      train_state,
      dataset_element_shape_dtype['image'],
      dataset_element_shape_dtype['text']).compile()
  t1 = time.time()
  # Report compilation time, and flops and optimal seconds per step and device.
  writer.write_scalars(init_step + 1, {'train/compile_secs': t1 - t0})
  train_step_flops_per_device, train_step_seconds_per_device = (
      utils.get_flops_and_seconds_per_device(train_step_pjit))
  if train_step_flops_per_device:
    writer.write_scalars(
        init_step + 1,
        {'train/step_flops_per_device': train_step_flops_per_device})
  if train_step_seconds_per_device:
    writer.write_scalars(
        init_step + 1,
        {'train/step_seconds_per_device': train_step_seconds_per_device})
  train_cost_fn = make_train_cost_fn(train_step_pjit)
  for step, batch in zip(range(init_step + 1, train_steps + 1), tr_iter):
    profile_hook(step)
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_state, metrics = train_step_pjit(train_state, batch['image'],
                                             batch['text'])
    progress_hook(step, scalar_metrics=(
        train_cost_fn(step) | {f'train/{k}': v for k, v in metrics.items()}
    ))
    _save_checkpoint(step, train_state, tr_iter)
    fewshot_hook(step, variables={'params': train_state.params},
                 **train_cost_fn(step))
    retrieval_hook(step, variables={'params': train_state.params},
                   **train_cost_fn(step))
    zeroshot_hook(step, variables={'params': train_state.params},
                  **train_cost_fn(step))
  ckpt_manager.wait_until_finished()
  if not tf.io.gfile.exists(os.path.join(workdir, f'ckpt/{train_steps}')):
    multihost_utils.sync_devices('training:ckpt-last')
    _save_checkpoint(train_steps, train_state, tr_iter, force=True)
    ckpt_manager.wait_until_finished()
  multihost_utils.sync_devices('training:completed')
  logging.info('Training completed.')
