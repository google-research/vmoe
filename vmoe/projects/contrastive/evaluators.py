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

"""Evaluators used during contrastive training."""
import time
from typing import Any, Callable, Iterable, Optional, Tuple

from clu import metric_writers
from clu import periodic_actions
import jax

# pylint: disable=g-import-not-at-top
try:
  from big_vision.evaluators.proj.image_text import discriminative_classifier as bv_discriminative
except ImportError:
  bv_discriminative = None


try:
  from big_vision.evaluators.proj.image_text import retrieval as bv_retrieval
except ImportError:
  bv_retrieval = None
# pylint: enable=g-import-not-at-top

Array = jax.Array
PyTree = Any


class ZeroShotPeriodicAction(periodic_actions.PeriodicCallback):
  """Periodic action that runs Big Vision's Retrieval evaluator repeatedly."""

  def __init__(
      self,
      *,
      metric_writer: metric_writers.MetricWriter,
      apply_fn: Callable[..., Tuple[Array, Array, Any]],
      data_sharding: jax.sharding.NamedSharding,
      every_steps: Optional[int] = None,
      every_secs: Optional[float] = None,
      on_steps: Optional[Iterable[int]] = None,
      report_progress: Optional[periodic_actions.ReportProgress] = None,
      report_progress_name: str = 'zeroshot',
      **bv_evaluator_kwargs,
  ):
    """Constructor."""
    if bv_discriminative is None:
      raise NotImplementedError(
          'Big Vision must be installed to run the discriminative evaluation.')
    bv_evaluator = bv_discriminative.Evaluator(
        predict_fn=apply_fn,
        devices=list(data_sharding.mesh.devices.flatten()),
        **bv_evaluator_kwargs,
    )
    callback = self._make_callback_fn(
        evaluator=bv_evaluator,
        metric_writer=metric_writer,
        report_progress=report_progress,
        report_progress_name=report_progress_name,
    )
    super().__init__(
        every_steps=every_steps,
        every_secs=every_secs,
        on_steps=on_steps,
        callback_fn=callback,
        execute_async=False,
        pass_step_and_time=True)

  def _make_callback_fn(
      self, *, evaluator, metric_writer, report_progress,
      report_progress_name):

    def callback_fn(step: int, t: Optional[float], variables: PyTree, **kwargs):
      del t  # Unused.
      metrics = {}
      t0 = time.time()
      for task in evaluator.datasets:
        acc = evaluator.evaluate(variables, task)['accuracy']
        t1 = time.time()
        metrics[f'{report_progress_name}/{task}/accuracy'] = acc
        metrics[f'{report_progress_name}/{task}/duration_secs'] = t1 - t0
      metrics = metrics | {k: v for k, v in kwargs.items() if v is not None}
      metric_writer.write_scalars(step, metrics)

    if report_progress is None:
      return callback_fn
    else:
      return report_progress.timed(
          report_progress_name, wait_jax_async_dispatch=False)(callback_fn)


class RetrievalPeriodicAction(periodic_actions.PeriodicCallback):
  """Periodic action that runs Big Vision's Retrieval evaluator repeatedly."""

  def __init__(
      self,
      *,
      metric_writer: metric_writers.MetricWriter,
      apply_fn: Callable[..., Tuple[Array, Array, Any]],
      task: str,
      data_sharding: jax.sharding.NamedSharding,
      every_steps: Optional[int] = None,
      every_secs: Optional[float] = None,
      on_steps: Optional[Iterable[int]] = None,
      report_progress: Optional[periodic_actions.ReportProgress] = None,
      report_progress_name: str = 'retrieval',
      **bv_evaluator_kwargs,
  ):
    """Constructor."""
    if bv_retrieval is None:
      raise NotImplementedError(
          'Big Vision must be installed to run the retrieval evaluation.')
    bv_evaluator = bv_retrieval.Evaluator(
        predict_fn=apply_fn,
        devices=list(data_sharding.mesh.devices.flatten()),
        **bv_evaluator_kwargs,
    )
    callback = self._make_callback_fn(
        evaluator=bv_evaluator,
        task=task,
        metric_writer=metric_writer,
        report_progress=report_progress,
        report_progress_name=report_progress_name,
    )
    super().__init__(
        every_steps=every_steps,
        every_secs=every_secs,
        on_steps=on_steps,
        callback_fn=callback,
        execute_async=False,
        pass_step_and_time=True)

  def _make_callback_fn(
      self, *, evaluator, task, metric_writer, report_progress,
      report_progress_name):

    def callback_fn(step: int, t: Optional[float], variables: PyTree, **kwargs):
      del t  # Unused.
      metrics = {}
      t0 = time.time()
      bv_metrics = evaluator.evaluate(variables)
      metrics.update({
          f'{report_progress_name}/{task}/txt2img/{k}': v
          for k, v in bv_metrics['txt2img'].items()
      })
      metrics.update({
          f'{report_progress_name}/{task}/img2txt/{k}': v
          for k, v in bv_metrics['img2txt'].items()
      })
      t1 = time.time()
      metrics[f'{report_progress_name}/{task}/duration_secs'] = t1 - t0
      metrics = metrics | {k: v for k, v in kwargs.items() if v is not None}
      metric_writer.write_scalars(step, metrics)

    if report_progress is None:
      return callback_fn
    else:
      return report_progress.timed(
          report_progress_name, wait_jax_async_dispatch=False)(callback_fn)
