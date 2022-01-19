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

"""PeriodicAction that saves checkpoints periodically."""
import os
from typing import Iterable, Optional

from clu import periodic_actions
import jax
import numpy as np

from vmoe import multihost_utils
from vmoe.checkpoints import base as checkpoints_base
from vmoe.checkpoints import partitioned as checkpoints_partitioned


AsyncResult = checkpoints_partitioned.AsyncResult
Mesh = checkpoints_partitioned.Mesh
PyTree = checkpoints_partitioned.PyTree
ThreadPool = checkpoints_partitioned.ThreadPool


class PeriodicSaveCheckpoint(periodic_actions.PeriodicCallback):
  """Saves checkpoints of a partitioned training state periodically.

  Example:
    saver = PeriodicSaveCheckpoint(
      prefix='/tmp/ckpt',
      state_axis_resources=state_axis_resources,
      every_steps=10)
    for step in range(100):
      state = update_state(...)
      saver(step=step, state=state)  # Saves at steps 0, 10, 20, 30, ...
  """

  def __init__(
      self,
      *,
      prefix: str,
      state_axis_resources: PyTree,
      mesh: Optional[Mesh] = None,
      num_shards: int = 0,
      num_threads: Optional[int] = None,
      wait_seconds: Optional[int] = None,
      every_steps: Optional[int] = None,
      every_secs: Optional[float] = None,
      on_steps: Optional[Iterable[int]] = None,
      keep_last: Optional[int] = None,
      keep_steps_multiple_of: Optional[int] = None,
      execute_async: bool = True,
      report_progress: Optional[periodic_actions.ReportProgress] = None,
      report_progress_name: str = 'ckpt'):
    """Initializer.

    Args:
      prefix: Prefix for the checkpoint files. The step number is appended to
        this when a checkpoint is written (e.g. prefix='ckpt_' gives checkpoints
        'ckpt_1', 'ckpt_2', ...).
      state_axis_resources: PyTree with PartitionSpec leaves, with the same
        structure as the `state` to checkpoint in every step, indicating how
        each axis of the corresponding array is partitioned across the axes of
        the logical device mesh.
      mesh: Logical device mesh used with pjit. If None, the active mesh will
        be used.
      num_shards: Number of checkpoint shards. If `num_shards <= 0`, the minimum
        number of shards will be used. If `num_shards > 0`, this number is only
        tentative.
      num_threads: Number of threads to use for writing checkpoint shards. If
        None, `multiprocessing.pool.cpu_count()` is used.
      wait_seconds: If given, we wait at most this number of seconds for the
        checkpoint writing to complete. Otherwise, TimeoutError is raised.
      every_steps: If given, writes a checkpoint every `every_steps` steps.
      every_secs: If given, writes a checkpoint every `every_secs` seconds.
      on_steps: If given, writes a checkpoint on these particular steps.
      keep_last: If given, we only keep the last `keep_last` checkpoints.
        If None, only the last checkpoint is kept.
      keep_steps_multiple_of: If given, all steps multiple of this number are
        kept (in addition to the `keep_last` steps).
      execute_async: If True, writes checkpoints shards asynchronously.
        If False, waits `wait_seconds` for the writing to complete. Note that,
        even if this is True, we always wait up to `wait_seconds` between two
        consecutive checkpointing steps.
      report_progress: When given, the `timed()` method of this `ReportProgress`
        is used to time the saving of checkpoints.
      report_progress_name: Name used by `ReportProgress.timed()`.
    """
    self._thread_pool = ThreadPool(processes=num_threads)
    self._async_result = None  # type: Optional[AsyncResult]
    self._wait_seconds = wait_seconds
    self._makedirs(os.path.dirname(prefix))
    keep_last = max(keep_last or 1, 1)
    keep_multiple = max(keep_steps_multiple_of or 0, 0)
    super().__init__(
        every_steps=every_steps,
        every_secs=every_secs,
        on_steps=on_steps,
        callback_fn=self._make_callback_fn(
            prefix, state_axis_resources, mesh, num_shards, wait_seconds,
            keep_last, keep_multiple, execute_async, self._thread_pool,
            report_progress, report_progress_name),
        # Note: save_checkpoint() is still asynchronous. This just means that
        # we wait until the callback_fn returns.
        execute_async=False,
        pass_step_and_time=True)

  def __del__(self):
    if self._async_result:
      self._block_async_result(self._wait_seconds)
    self._thread_pool.close()

  @classmethod
  def _makedirs(cls, workdir: str):
    # Process 0 creates the workdir if it doesn't exist. All processes wait
    # until this is done.
    if jax.process_index() == 0 and not os.path.exists(workdir):
      checkpoints_base.gfile.makedirs(workdir)
    multihost_utils.sync_devices(f'checkpoints:mkdir:{workdir}')

  @classmethod
  def _remove_old_checkpoints(cls, prefix: str, keep_last: int,
                              keep_multiple: int, thread_pool: ThreadPool):

    def _parse_step_from_filepath(filepath):
      m = checkpoints_base.CHECKPOINT_REGEX.fullmatch(filepath)
      step_str = m.group(2) if m else None
      return int(step_str[1:]) if step_str else None

    def _find_step_numbers(filepaths):
      for step in map(_parse_step_from_filepath, filepaths):
        if step is not None:
          yield step

    def _remove():
      # Find step number of pending shards.
      workdir = os.path.dirname(prefix)
      basename = os.path.basename(prefix)
      prefix_tmp = os.path.join(workdir, f'.tmp.{basename}') + '*'
      checkpoints_tmp = checkpoints_base.gfile.glob(prefix_tmp)
      pending_steps = set(_find_step_numbers(checkpoints_tmp))
      # Find all completed shards.
      checkpoints = checkpoints_base.gfile.glob(prefix + '*')
      completed_steps = set(_find_step_numbers(checkpoints))
      # Keep `keep_last` completed steps.
      keep_steps = set(sorted(completed_steps - pending_steps)[-keep_last:])
      # Keep steps multiple of `keep_multiple`.
      if keep_multiple > 0:
        keep_steps.update([
            step for step in completed_steps if step % keep_multiple == 0])
      # Always keep pending steps.
      keep_steps.update(pending_steps)
      # Remove checkpoints.
      def match_remove_fn(filepath):
        # Returns True (to remove) if the step is not in `keep_steps`.
        step = _parse_step_from_filepath(filepath)
        return (step not in keep_steps) if step is not None else False
      checkpoints_base.remove_checkpoints(
          checkpoints, match_remove_fn, thread_pool=thread_pool)

    # Only process 0 removes files. All processes wait untils this is done.
    if jax.process_index() == 0:
      _remove()
    multihost_utils.sync_devices(f'checkpoints:remove:{prefix}')

  def _block_async_result(self, wait_seconds: Optional[int]):
    try:
      self._async_result.get(wait_seconds)
      self._async_result = None
    except TimeoutError:
      raise TimeoutError('Timeout while writing checkpoint files after '
                         f'{wait_seconds} seconds.')

  def _make_callback_fn(self, prefix, state_axis_resources, mesh, num_shards,
                        wait_seconds, keep_last, keep_multiple, execute_async,
                        thread_pool, report_progress, report_progress_name):

    def callback_fn(step: int, t: float, state: PyTree):
      del t  # Unused.
      # Wait up to `wait_seconds` seconds, until the previous checkpoint is
      # completed before starting to write a new checkpoint. If the timeout
      # expires, an exception is raised. This is to avoid having multiple copies
      # of the model in the CPU memory.
      if self._async_result:
        self._block_async_result(wait_seconds)
        multihost_utils.sync_devices(f'checkpoints:sync_pending:{prefix}')
      # Remove outdated checkpoints before starting writing new ones.
      self._remove_old_checkpoints(
          prefix, keep_last, keep_multiple, thread_pool)
      # Save new checkpoint.
      self._async_result = checkpoints_partitioned.save_checkpoint(
          prefix=f'{prefix}_{step}',
          # Note: saving is faster if we transfer the data from device to CPU
          # in one go.
          tree=jax.tree_map(np.asarray, state),
          axis_resources=state_axis_resources,
          mesh=mesh,
          num_shards=num_shards,
          thread_pool=thread_pool,
          makedirs=False,
          overwrite=True)
      # Optionally, wait `wait_seconds` until the checkpointing is done, or
      # raise an exception if writing doesn't finish in `wait_seconds`.
      if not execute_async:
        self._block_async_result(wait_seconds)
        multihost_utils.sync_devices(f'checkpoints:no_async:{prefix}')

    if report_progress is None:
      return callback_fn
    else:
      return report_progress.timed(
          report_progress_name, wait_jax_async_dispatch=False)(callback_fn)
