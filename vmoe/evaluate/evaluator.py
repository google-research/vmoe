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

"""Classes and functions useful for evaluating models."""
import functools
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import cachetools
from clu import metric_writers
from clu import periodic_actions
import clu.data
import flax.core
import flax.struct
import jax
from jax.experimental import pjit
import jax.numpy as jnp
from vmoe import utils
from vmoe.data import input_pipeline
from vmoe.data import pjit_utils

Array = jnp.ndarray
DatasetIterator = clu.data.DatasetIterator
EvalStepPjitFn = Callable[['EvalState', 'PyTree', Array, Array, Array],
                          'EvalState']
PartitionSpec = jax.sharding.PartitionSpec
PRNGKey = jnp.ndarray
PyTree = Any
VALID_KEY = input_pipeline.VALID_KEY


class EvalState(flax.struct.PyTreeNode):
  """Evaluation state."""
  num: int
  sum_correct: int
  sum_loss: float
  rngs: Dict[str, PRNGKey]

  def update(self, num, correct, loss, rngs):
    num = self.num + num
    sum_correct = self.sum_correct + jnp.sum(correct)
    sum_loss = self.sum_loss + jnp.sum(loss)
    return self.replace(
        num=num, sum_correct=sum_correct, sum_loss=sum_loss, rngs=rngs)


class EvaluateMultipleDatasets(periodic_actions.PeriodicCallback):
  """Periodic action that evaluates a model on multiple datasets.

  Usage:
    eval_action = EvaluateMultipleDatasets(
      apply_fn=model.apply, datasets=eval_datasets, every_steps=10, ...)
    for step in range(100):
      params = train_step(params, ...)
      eval_action(step, params=params)  # Runs at steps 10, 20, 30, ...
  """

  def __init__(
      self,
      *,
      apply_fn: Callable[..., Any],
      loss_fn: Callable[[Array, Array], Array],
      label_pred_fn: Callable[[Array], Array],
      params_axis_resources: PyTree,
      input_axis_resources: PartitionSpec,
      datasets: Mapping[str, DatasetIterator],
      metric_writer: metric_writers.MetricWriter,
      rng_keys: Sequence[str],
      seed: int = 0,
      every_steps: Optional[int] = None,
      every_secs: Optional[float] = None,
      on_steps: Optional[Iterable[int]] = None,
      report_progress: Optional[periodic_actions.ReportProgress] = None,
      report_progress_name: str = 'eval'):
    """Initializer.

    Args:
      apply_fn: Function used to apply the model on a batch of inputs. This
        typically is the `apply` method of a Linen module. It must return an
        array (outputs of the model) and a dictionary with additional metrics.
      loss_fn: Loss function used with the model. Its arguments are
        (outputs, labels) and return the loss for each input example.
      label_pred_fn: Function that determines how to go from the outputs of the
        model to the predicted label. A common case is argmax(logits, -1).
      params_axis_resources: PyTree with the same structure as the params passed
        to the model, but with PartitionSpec leaves indicating how each array is
        partitioned.
      input_axis_resources: PartitionSpec indicating how the input to the model
        is partitioned.
      datasets: Mapping from names to DatasetIterator objects with the datasets
        to evaluate.
      metric_writer: CLU metric_writer object, used to report the evaluation
        metrics.
      rng_keys: Collection of PRNG names that the `apply_fn` expects. It can be
        empty if the evaluation is deterministic.
      seed: Seed used to create PRNGKeys (default = 0).
      every_steps: Run evaluation on all datasets every `every_steps` steps.
      every_secs: Run evaluation on all datasets every `every_secs` seconds.
      on_steps: Run evaluation on all datasets on these particular steps.
      report_progress: When given, the `timed()` method of this `ReportProgress`
        is used to time the evaluation of multiple datasets.
      report_progress_name: Name used by `ReportProgress.timed()`.
    """
    callback = self._make_callback_fn(
        apply_fn=apply_fn,
        loss_fn=loss_fn,
        label_pred_fn=label_pred_fn,
        params_axis_resources=params_axis_resources,
        input_axis_resources=input_axis_resources,
        metric_writer=metric_writer,
        datasets=datasets,
        rng_keys=tuple(rng_keys),
        seed=seed,
        report_progress=report_progress,
        report_progress_name=report_progress_name)
    super().__init__(
        every_steps=every_steps,
        every_secs=every_secs,
        on_steps=on_steps,
        callback_fn=callback,
        execute_async=False,
        pass_step_and_time=True)

  @classmethod
  def _make_callback_fn(cls, *, apply_fn, loss_fn, label_pred_fn,
                        params_axis_resources, input_axis_resources, datasets,
                        metric_writer, rng_keys, seed, report_progress,
                        report_progress_name):
    datasets_element_shape_dtype = {
        name: pjit_utils.get_dataset_shape_dtype_struct(
            datasets[name], input_axis_resources)
        for name in datasets
    }

    # Note: We create the eval_step_pjit here to avoid multiple compilation
    # steps. If the shapes of inputs/outputs for all datasets is the same, this
    # will be only compiled once.
    eval_step_pjit = make_eval_step_pjit(
        params_axis_resources=params_axis_resources,
        input_axis_resources=input_axis_resources,
        apply_fn=apply_fn,
        loss_fn=loss_fn,
        label_pred_fn=label_pred_fn,
        rng_keys=rng_keys)

    @functools.partial(
        pjit.pjit,
        in_axis_resources=(), out_axis_resources=None, static_argnums=(0,))
    def make_eval_state_pjit(seed):
      rngs = utils.make_rngs(rng_keys, seed)
      return EvalState(
          num=jnp.zeros((), dtype=jnp.float32),
          sum_correct=jnp.zeros((), dtype=jnp.float32),
          sum_loss=jnp.zeros((), dtype=jnp.float32),
          rngs=rngs)

    @cachetools.cached(
        cache={}, key=lambda name, *_: cachetools.keys.hashkey(name))
    def compile_for_dataset(name, params, train_step):
      # Note: This is not the initial EvalState, this only serves to compile the
      # eval step for a given dataset.
      eval_state = EvalState(
          num=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
          sum_correct=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
          sum_loss=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
          rngs=jax.eval_shape(lambda: utils.make_rngs(rng_keys, 0)))
      t0 = time.time()
      eval_step_pjit_ds = eval_step_pjit.lower(
          eval_state,
          params,
          datasets_element_shape_dtype[name]['image'],
          datasets_element_shape_dtype[name]['labels'],
          datasets_element_shape_dtype[name][VALID_KEY]).compile()
      t1 = time.time()
      metric_writer.write_scalars(train_step, {f'{name}/compile_secs': t1 - t0})
      return eval_step_pjit_ds

    def callback_fn(step: int, t: Optional[float], params: PyTree):
      del t  # Unused.
      for name, dataset in datasets.items():
        eval_step_pjit_ds = compile_for_dataset(name, params, step)
        # NOTE: Fold-in the dataset name and/or the train_step to the seed
        # in order to use different initial seeds for each dataset and/or
        # evaluation run. Notice that each eval step will use a different seed,
        # since it's updated in the EvalState (see evaluate_step).
        eval_state = jax.tree_util.tree_map(lambda x: x.block_until_ready(),
                                            make_eval_state_pjit(seed))
        ds_iter = pjit_utils.prefetch_to_device(
            iter(dataset), input_axis_resources, size=0)
        t0 = time.time()
        eval_state = evaluate_dataset(eval_step_pjit=eval_step_pjit_ds,
                                      eval_state=eval_state,
                                      dataset=ds_iter,
                                      params=params)
        t1 = time.time()
        with jax.spmd_mode('allow_all'):
          metric_writer.write_scalars(step, {
              f'{name}/prec@1': eval_state.sum_correct / eval_state.num,
              f'{name}/loss': eval_state.sum_loss / eval_state.num,
              f'{name}/duration_secs': t1 - t0,
          })
        # Reset iterator for the next evaluation.
        dataset.reset()

    if report_progress is None:
      return callback_fn
    else:
      return report_progress.timed(
          report_progress_name, wait_jax_async_dispatch=False)(callback_fn)


def evaluate_dataset(
    *,
    eval_step_pjit: EvalStepPjitFn,
    eval_state: EvalState,
    dataset: Iterable[PyTree],
    params: PyTree,
) -> EvalState:
  """Evaluates a given model on the given dataset."""
  for batch in dataset:
    eval_state = eval_step_pjit(eval_state, params, batch['image'],
                                batch['labels'], batch[VALID_KEY])
  return jax.tree_map(lambda x: x.block_until_ready(), eval_state)


def evaluate_step(
    state: EvalState,
    params: PyTree,
    images: Array,
    labels: Array,
    valid: Array,
    apply_fn: Callable[..., Any],
    loss_fn: Callable[[Array, Array], Array],
    label_pred_fn: Callable[[Array], Array],
) -> EvalState:
  """Performs one evaluation step, updating the given state."""
  rngs, next_rngs = utils.tree_rngs_split(state.rngs)
  logits, _ = apply_fn({'params': params}, images, rngs=rngs)
  loss = loss_fn(logits, labels)
  loss = valid * jnp.sum(loss, axis=tuple(range(1, loss.ndim)))
  correct = (valid[:, None] * labels *
             jax.nn.one_hot(label_pred_fn(logits), labels.shape[1]))
  num_valid = jnp.sum(valid, dtype=jnp.float32)
  return state.update(num_valid, correct, loss, next_rngs)


def make_eval_step_pjit(
    params_axis_resources: PyTree,
    input_axis_resources: PartitionSpec,
    apply_fn: Callable[..., Any],
    loss_fn: Callable[[Array, Array], Array],
    label_pred_fn: Callable[[Array], Array],
    rng_keys: Sequence[str],
) -> EvalStepPjitFn:
  """Create a pjitted function that performs one evaluation step."""
  eval_state_axis_resources = EvalState(
      num=PartitionSpec(),
      sum_correct=PartitionSpec(),
      sum_loss=PartitionSpec(),
      rngs={key: PartitionSpec() for key in rng_keys})
  eval_step_pjit = pjit.pjit(
      fun=functools.partial(
          evaluate_step,
          apply_fn=apply_fn,
          loss_fn=loss_fn,
          label_pred_fn=label_pred_fn),
      in_axis_resources=(
          eval_state_axis_resources,  # state
          params_axis_resources,      # params
          input_axis_resources,       # images
          input_axis_resources,       # labels
          input_axis_resources,       # valid
      ),
      out_axis_resources=eval_state_axis_resources,
      donate_argnums=(0, 2, 3, 4))
  return eval_step_pjit
