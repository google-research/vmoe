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

"""Classes used for the few-shot evaluation of models.

The code used for few-shot evaluation is heavily inspired by the work and code
of A. Kolesnikov, L. Beyer, and X. Zhai.
"""
import functools
import itertools
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import cachetools
from clu import metric_writers
from clu import periodic_actions
import flax.struct
import jax
import jax.experimental.pjit
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets.public_api as tfds
import vmoe.data.input_pipeline
import vmoe.data.pjit_utils
import vmoe.utils

Array = Union[jax.numpy.ndarray, np.ndarray]
DatasetIterator = vmoe.data.input_pipeline.DatasetIterator
PartitionSpec = jax.sharding.PartitionSpec
PyTree = Any
PRNGKey = jax.Array

BIAS_CONSTANT = 100.0
VALID_KEY = vmoe.data.input_pipeline.VALID_KEY


class FewShotState(flax.struct.PyTreeNode):
  rngs: Dict[str, PRNGKey]


class FewShotPeriodicAction(periodic_actions.PeriodicCallback):
  """Periodic action that runs the pjit-friendly FewShotEvaluator repeatedly."""

  def __init__(
      self,
      *,
      shots: Sequence[int],
      l2_regs: Sequence[float],
      metric_writer: metric_writers.MetricWriter,
      datasets: Mapping[str, Tuple[str, str, str]],
      apply_fn: Callable[..., Any],
      rng_keys: Sequence[str],
      seed: int = 0,
      seed_fold_in_step: bool = False,
      seeds_per_step: int = 1,
      main_task: Optional[Tuple[str, int]] = None,
      main_task_prefix: str = '0',
      every_steps: Optional[int] = None,
      every_secs: Optional[float] = None,
      on_steps: Optional[Iterable[int]] = None,
      report_progress: Optional[periodic_actions.ReportProgress] = None,
      report_progress_name: str = 'fewshot',
      prefetch_to_device: Optional[int] = None,
      **dataset_kwargs):
    """Initializer.

    Args:
      shots: Sequence of integers with the number of training examples per class
        to use in each of the k-shot linear regression runs.
      l2_regs: Sequence of floats with the L2 regularization hyper-parameters to
        explore for linear regression.
      metric_writer: CLU metric_writer object, used to report the evaluation
        metrics.
      datasets: Mapping from names to tuples with three string elements:
        ('tfds_name', 'train_split', 'test_split'), as expect by the
        FewShotEvaluator class.
      apply_fn: Function used to apply the model on a batch of inputs. This
        typically is the `apply` method of a Linen module. It must return an
        array (features extract by the model) and a dict with metrics.
      rng_keys: Collection of PRNG names that the `apply_fn` expects. It can be
        empty if the evaluation is deterministic.
      seed: Seed used to create PRNGKeys (default = 0).
      seed_fold_in_step: If True, folds the step number in the seed at each
        evaluation.
      seeds_per_step: Number of seeds to use in each evaluation run.
      main_task: Optional pair (name, shot) indicating the main task to report
        separately to the metric_writer.
      main_task_prefix: Prefix used to report the accuracy of the main task.
        Default is '0', meaning that the main task's accuracy will be reported
        using the key `f'0/{name}/{shot}shot'`.
      every_steps: Run evaluation on all datasets every `every_steps` steps.
      every_secs: Run evaluation on all datasets every `every_secs` seconds.
      on_steps: Run evaluation on all datasets on these particular steps.
      report_progress: When given, the `timed()` method of this `ReportProgress`
        is used to time the evaluation of multiple datasets.
      report_progress_name: Name used by `ReportProgress.timed()`.
      prefetch_to_device: Optional number of batches prefetched to the devices.
      **dataset_kwargs: Additional kwargs passed to the dataset.
    """
    # TODO(jpuigcerver): This is a hack. Split this class into a evaluator and
    # a periodic action that uses that evaluator.
    self.last_metrics = None
    callback = self._make_callback_fn(
        # Used to obtain the representation/labels/mask of a batch of images.
        representation_fn=_make_fewshot_step_pjit(
            apply_fn=apply_fn,
            rng_keys=rng_keys),
        shots=shots,
        l2_regs=l2_regs,
        datasets_info=_get_datasets(datasets, **dataset_kwargs),
        prefetch_to_device=prefetch_to_device,
        rng_keys=tuple(rng_keys),
        seed=seed,
        seed_fold_in_step=seed_fold_in_step,
        seeds_per_step=seeds_per_step,
        metric_writer=metric_writer,
        report_progress=report_progress,
        report_progress_name=report_progress_name,
        main_task=main_task,
        main_task_prefix=main_task_prefix)
    super().__init__(
        every_steps=every_steps,
        every_secs=every_secs,
        on_steps=on_steps,
        callback_fn=callback,
        execute_async=False,
        pass_step_and_time=True)

  def _make_callback_fn(self, *, representation_fn, shots, l2_regs,
                        datasets_info, prefetch_to_device,
                        rng_keys, seed, seed_fold_in_step, seeds_per_step,
                        metric_writer, report_progress, report_progress_name,
                        main_task, main_task_prefix):
    # Make an iterator over a dataset with optional device prefetching.
    def make_dataset_iterator(dataset_iterator: DatasetIterator):
      return vmoe.data.pjit_utils.prefetch_to_device(
          iterator=dataset_iterator,
          size=prefetch_to_device)

    @functools.partial(
        jax.experimental.pjit.pjit, in_shardings=(None, None, None),
        out_shardings=None)
    def make_fewshot_state_pjit(seed, sub_seed, step):
      if rng_keys:
        rng = jax.random.PRNGKey(seed)
        rng = jax.random.fold_in(rng, sub_seed)
        if seed_fold_in_step:
          rng = jax.random.fold_in(rng, step)
        rngs = jax.random.split(rng, len(rng_keys))
        rngs = dict(zip(rng_keys, rngs))
      else:
        rngs = {}
      return FewShotState(rngs=rngs)

    def callback_fn(step: int, t: Optional[float], variables: PyTree, **kwargs):
      del t  # Unused.
      # Two-level dict: first is dataset name, second level is (shot, l2_reg).
      all_results = {}
      metrics = {}
      t0 = time.time()
      for sub_seed in range(seeds_per_step):
        # This is used to permute the fewshot examples.
        permute_seed = hash(
            (seed, step, sub_seed) if seed_fold_in_step else (seed, sub_seed))
        for name, (tr_ds, te_ds, num_classes) in datasets_info.items():
          t0_d = time.time()
          # Compute fewshot metrics.
          state = make_fewshot_state_pjit(seed, sub_seed, step)
          name = name if seeds_per_step == 1 else f'{name}-seed-{sub_seed}'
          all_results[name] = _compute_fewshot_metrics(
              representation_fn, shots, l2_regs, num_classes, state, variables,
              make_dataset_iterator(tr_ds), make_dataset_iterator(te_ds),
              permute_seed % 2**31)
          t1_d = time.time()
          # Report few-shot eval duration for each dataset.
          metrics[f'{report_progress_name}/{name}/duration_secs'] = t1_d - t0_d
          # Reset iterators for the next eval.
          tr_ds.reset()
          te_ds.reset()
      t1 = time.time()
      metrics[f'{report_progress_name}/duration_secs'] = t1 - t0

      # For each dataset, report only the result with the best l2_reg.
      best_l2 = _find_best_l2_reg(all_results, shots, l2_regs)
      metrics.update({
          f'{report_progress_name}/{shot}shot_best_l2': l2
          for shot, l2 in best_l2.items()
      })
      metrics.update({
          f'{report_progress_name}/{name}/{shot}shot':
          all_results[name][shot, best_l2[shot]]
          for name, shot in itertools.product(all_results, shots)
      })
      # If a main task is specified, report the corresponding accuracy too.
      if main_task:
        name, shot = main_task
        if seeds_per_step == 1:
          accuracy = all_results[name][shot, best_l2[shot]]
        else:
          accuracy = sum(
              all_results[f'{name}-seed-{sub_seed}'][shot, best_l2[shot]]
              for sub_seed in range(seeds_per_step)
          ) / seeds_per_step
        metrics[f'{main_task_prefix}/{name}/{shot}shot'] = accuracy
      metrics = metrics | {k: v for k, v in kwargs.items() if v is not None}
      metric_writer.write_scalars(step, metrics)
      self.last_metrics = metrics

    if report_progress is None:
      return callback_fn
    else:
      return report_progress.timed(
          report_progress_name, wait_jax_async_dispatch=False)(callback_fn)


def _compute_fewshot_metrics(
    representation_fn: Callable[..., Any],
    shots: Sequence[int],
    l2_regs: Sequence[float],
    num_classes: int,
    state: FewShotState,
    variables: PyTree,
    tr_iter: Iterable[Mapping[str, Array]],
    te_iter: Iterable[Mapping[str, Array]],
    seed: int = 0,
):
  """Computes few-shot accuracy and other metrics for a given dataset."""
  state, tr_features, tr_labels = _compute_representation(
      representation_fn, state, variables, tr_iter)
  state, te_features, te_labels = _compute_representation(
      representation_fn, state, variables, te_iter)

  rng = np.random.default_rng(seed)
  class_indices = [
      rng.permutation(np.where(tr_labels == c)[0]) for c in range(num_classes)]
  results = {}
  for shot in shots:
    all_idx = [indices[:shot] for indices in class_indices]
    all_idx = np.concatenate(all_idx, axis=0)
    x, y = tr_features[all_idx], tr_labels[all_idx]
    cache = _precompute_cache(x, y, num_classes)
    for l2_reg in l2_regs:
      acc = _eig_fewshot_acc_fn(cache, te_features, te_labels, l2_reg)
      results[shot, l2_reg] = np.array(acc)

  return results


def _compute_representation(
    representation_fn: Callable[..., Any],
    state: FewShotState,
    variables: PyTree,
    iterator: Iterable[Mapping[str, Array]],
) -> Tuple[FewShotState, Array, Array]:
  """Computes the representation of multiple batches."""
  features_list = []
  labels_list = []
  for batch in iterator:
    state, features, labels, mask = representation_fn(
        state, variables, batch['image'], batch['label'], batch[VALID_KEY])
    mask = np.asarray(mask).astype(np.bool_)
    features_list.append(np.asarray(features)[mask])
    labels_list.append(np.asarray(labels)[mask])
  features = np.concatenate(features_list, axis=0)
  labels = np.concatenate(labels_list, axis=0)
  return state, features, labels


def _fewshot_step(
    state: FewShotState,
    variables: PyTree,
    images: Array,
    label: Array,
    valid: Array,
    apply_fn: Callable[..., Any],
) -> Tuple[FewShotState, Array, Array, Array]:
  """Applies `apply_fn` function and updates the state."""
  rngs, next_rngs = vmoe.utils.tree_rngs_split(state.rngs)
  features, _ = apply_fn(variables, images, rngs=rngs)
  return FewShotState(rngs=next_rngs), features, label, valid


def _find_best_l2_reg(all_results, shots, l2_regs):
  """Finds the best L2 regularization param across all datasets, per-shot."""
  # Similar to ATARI benchmark requiring one single hyper-param across tasks,
  # or BiT-HyperRule defining one clear thing. Avoids over-fitting to a single
  # task by selecting on test there, while also avoiding the need to do
  # cross-validation runs for each task.
  best_l2 = {}
  for shot in shots:
    reg_ranks = []
    for res in all_results.values():
      reg_accus = [res[shot, l2] for l2 in l2_regs]
      reg_ranks.append(np.argsort(np.argsort(reg_accus)))
    best_l2[shot] = l2_regs[np.argmax(np.mean(reg_ranks, axis=0))]
  return best_l2


def _get_datasets(
    datasets: Mapping[str, Tuple[str, str, str]],
    **dataset_kwargs,
) -> Mapping[str, Tuple[DatasetIterator, DatasetIterator, int]]:
  """Returns a dict with the train&test datasets, and the num_classes of each fewshot dataset."""

  @cachetools.cached(cache={})
  def _get_dataset(name: str, split: str) -> DatasetIterator:
    return vmoe.data.input_pipeline.get_dataset(
        variant='fewshot', name=name, split=split, **dataset_kwargs)

  @cachetools.cached(cache={})
  def _get_num_classes(name: str):
    return tfds.builder(name).info.features['label'].num_classes

  return {
      key: (_get_dataset(name, tr_split), _get_dataset(name, te_split),
            _get_num_classes(name))
      for key, (name, tr_split, te_split) in datasets.items()
  }


def _make_fewshot_step_pjit(
    apply_fn: Callable[..., Any],
    rng_keys: Sequence[str],
):
  """Wraps _fewshot_step with pjit."""
  state_axis_resources = FewShotState(  # pytype: disable=wrong-arg-types
      rngs={key: PartitionSpec() for key in rng_keys})
  fewshot_step_pjit = jax.experimental.pjit.pjit(
      functools.partial(_fewshot_step, apply_fn=apply_fn),
      out_shardings=(
          state_axis_resources,  # state
          # Note: we dont partition these so that each host gets a copy of all
          # data.
          PartitionSpec(),  # features
          PartitionSpec(),  # label
          PartitionSpec(),  # valid
      ),
  )
  return fewshot_step_pjit


@functools.partial(jax.jit, backend='cpu')
def _eig_fewshot_acc_fn(cache, x_test, y_test, l2_reg):
  """Computes (x,y) linear regression accuracy on (x_test, y_test)."""

  x_test = (x_test - cache['mean']) / cache['std']
  x_test = jnp.pad(x_test, ((0, 0), (0, 1)), constant_values=BIAS_CONSTANT)

  rhs = cache['rhs']
  lhs = cache['lhs']
  eigs = cache['eigs']

  # See comments in _precompute_cache for context about the formula.
  scaling = 1.0 / (eigs + l2_reg * jnp.ones_like(eigs))
  scaling = scaling.reshape((1, -1))
  w = (lhs * scaling) @ rhs

  # Predict test-set values and measure their accuracy
  preds = jnp.argmax(x_test @ w, axis=1)
  return jnp.mean(preds == y_test)


# Setup function for few-shot regression on CPU to avoid using extra TPU memory.
@functools.partial(jax.jit, backend='cpu', static_argnums=(2,))
def _precompute_cache(x, y, num_classes):
  """Cache quantities to speed-up the computation of L2-regularized least-sq."""
  # Whiten
  mean = jnp.mean(x, axis=0, keepdims=True)
  std = jnp.std(x, axis=0, keepdims=True) + 1e-5
  x = (x - mean) / std

  # Add a constant feature for the bias, large so it's almost unregularized:
  x = jnp.pad(x, ((0, 0), (0, 1)), constant_values=BIAS_CONSTANT)

  # To one-hot representation rescaled into {-1, 1}
  y = 2.0 * jax.nn.one_hot(y, num_classes) - 1.0

  num_points, dim = x.shape
  # Let N be the number of points, D the dimension and C the number of classes.
  # We have x of shape (N, D) and y of shape (N, C).
  # For least-squares, we can compute
  #
  #   (A) when N >= D, (x^T x + l2 Id)^{-1} x^T y
  #   (B) when D > N, x^T  (x x^T + l2 Id)^{-1} y
  #
  # We pre-compute the eigen-decomposition of either x^T x or x x^T which
  # becomes q diag(eigs) q^T with q unitary matrix either (D, D) or (N, N)
  # and eigs a vector (D,) or (N,).
  #
  # For any l2 > 0, we can compute (x^T x + l2 Id)^{-1} or (x x^T + l2 Id)^{-1}
  # by simply computing q (diag(eigs) + l2 Id)^{-1} q^T.
  # (SVD would be more natural here, but it proved slower, so we use eigh)
  #
  # Both cases (A) and (B) can be viewed as lhs (diag(eigs) + l2 Id)^{-1} rhs,
  # where lhs/rhs are pre-computed left/right-hand sides to specify.
  #
  if num_points >= dim:
    eigs, q = jnp.linalg.eigh(x.T @ x)
    rhs = q.T @ (x.T @ y)
    lhs = q
  else:
    eigs, q = jnp.linalg.eigh(x @ x.T)
    rhs = q.T @ y
    lhs = x.T @ q

  cache = {'eigs': eigs, 'rhs': rhs, 'lhs': lhs, 'mean': mean, 'std': std}
  return cache
