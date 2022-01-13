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

"""Base functions for checkpointing."""
import functools
import itertools
import multiprocessing.pool
import os
import re
from typing import Any, Callable, Iterable, Iterator, Sequence, Mapping, Optional

from tensorflow.io import gfile
from vmoe.checkpoints import serialization

AsyncResult = multiprocessing.pool.AsyncResult
ThreadPool = multiprocessing.pool.ThreadPool

# Allows checkpoints patterns such as:
# ckpt, ckpt.foo, ckpt-0-of-3, ckpt_1, ckpt_1.idx, ckpt_1.data-00-of-10.
CHECKPOINT_REGEX = re.compile(
    r'^(.*?)(_[0-9]+)?(\.[a-zA-Z]+)?(-[0-9]+-of-[0-9]+)?$')


def add_shard_suffix(filepath: str, shard: int, shard_count: int) -> str:
  return f'{filepath}-{shard:05d}-of-{shard_count:05d}'


def find_latest_complete_checkpoint_for_prefix(
    prefix: str, suffixes: Optional[Sequence[str]] = None) -> Optional[str]:
  """Returns the latest complete checkpoint matching a given prefix.

  Args:
    prefix: Prefix of the checkpoint file (e.g. '/tmp/ckpt').
    suffixes: Collection of required suffixes for the checkpoints.

  Returns:
    Latest available checkpoint (if any). E.g. '/tmp/ckpt_2500'.
  """
  for step in iterate_complete_steps_for_prefix(
      prefix, suffixes=suffixes, decreasing=True):
    return prefix + f'_{step}'
  return None


def iterate_complete_steps_for_prefix(
    prefix: str,
    suffixes: Optional[Sequence[str]] = None,
    decreasing: bool = False) -> Iterator[int]:
  """Iterates over steps with complete checkpoints from a given prefix.

  Complete steps are those for which there are not incomplete (temp) checkpoint
  shards and for which all suffixes are present.

  E.g. If the prefix is '/dir/ckpt', the suffixes are ('.index', '.data') and
  the files in '/dir/' are: '/dir/ckpt_1.index', '/dir/ckpt_1.data',
  '/dir/ckpt_2.index', '/dir/.tmp.ckpt_2.data', '/dir/ckpt_3.data'. Then, the
  only completed step is 1, since there is one incomplete shard for step 2
  (i.e. '/dir/.tmp.ckpt_2.data') and there is one suffix missing for step 3
  (i.e. '/dir/ckpt_3.index').

  Args:
    prefix: Prefix of the checkpoint file (e.g. '/tmp/ckpt').
    suffixes: Collection of required suffixes for the checkpoints.
    decreasing: If True, iterates the step numbers in decreasing order.

  Yields:
    Integers corresponding to the completed step numbers for the given prefix.
  """
  if not suffixes:
    suffixes = (None,)
  suffixes = set(suffixes)

  def _parse_step_and_suffix_or_error(filepath):
    m = CHECKPOINT_REGEX.fullmatch(filepath)
    assert m is not None, (
        f'Filepath {filepath!r} does not match CHECKPOINT_REGEX. '
        f'This should not happen.')
    if m.group(2) is None:
      raise ValueError(f'Filepath {filepath!r} does not contain a step number.')
    step = int(m.group(2)[1:])
    suffix = m.group(3)
    return step, suffix

  # Find set of (step, suffix) from the given prefix.
  steps_and_suffixes = set(
      map(_parse_step_and_suffix_or_error, gfile.glob(prefix + '*')))
  # Remove any steps where there is an associated temp file.
  workdir = os.path.dirname(prefix)
  pattern_tmp = os.path.join(workdir, f'.tmp.{os.path.basename(prefix)}') + '*'
  incomplete_steps_and_suffixes = set(
      map(_parse_step_and_suffix_or_error, gfile.glob(pattern_tmp)))

  for step, group in itertools.groupby(
      sorted(steps_and_suffixes - incomplete_steps_and_suffixes,
             reverse=decreasing),
      lambda x: x[0]):
    if set(x[1] for x in group) == suffixes:
      yield step


def remove_checkpoints(filepaths: Iterable[str],
                       match_fn: Callable[[str], bool],
                       *,
                       thread_pool: Optional[ThreadPool] = None):
  """Removes checkpoints for which `match_fn` returns True."""

  def remove(filepath):
    if match_fn(filepath):
      gfile.remove(filepath)

  thread_pool = ThreadPool() if thread_pool is None else thread_pool
  thread_pool.map(remove, filepaths)


def remove_shard_suffix(filepath: str) -> str:
  return CHECKPOINT_REGEX.sub(r'\1\2\3', filepath)


def restore_checkpoint(filepath: str, tree: Optional[Any] = None) -> Any:
  with gfile.GFile(filepath, 'rb') as fp:
    checkpoint_contents = fp.read()
  if tree is None:
    return serialization.msgpack_restore(checkpoint_contents)
  else:
    return serialization.from_bytes(tree, checkpoint_contents)


def restore_multiple_checkpoints(
    filepath_tree_map: Mapping[str, Any],
    *,
    thread_pool: Optional[ThreadPool] = None) -> Mapping[str, Any]:
  thread_pool = thread_pool or ThreadPool()
  restored_trees = thread_pool.map(
      lambda item: restore_checkpoint(item[0], item[1]),
      filepath_tree_map.items())
  return dict(zip(filepath_tree_map, restored_trees))


def save_checkpoint(filepath: str,
                    tree: Any,
                    *,
                    overwrite: bool = True,
                    makedirs: bool = True) -> str:
  """Saves the given PyTree in the given location."""
  wdir = os.path.dirname(filepath)
  if makedirs and not gfile.exists(wdir):
    gfile.makedirs(wdir)

  temp_filepath = os.path.join(wdir, '.tmp.' + os.path.basename(filepath))
  with gfile.GFile(temp_filepath, 'wb') as fp:
    fp.write(serialization.to_bytes(tree))
  gfile.rename(temp_filepath, filepath, overwrite=overwrite)
  return filepath


def save_checkpoint_async(
    filepath: str,
    tree: Any,
    *,
    overwrite: bool = True,
    makedirs: bool = True,
    thread_pool: Optional[ThreadPool] = None) -> AsyncResult:
  """Saves the given PyTree in the given location, asynchronously."""
  thread_pool = thread_pool or ThreadPool()
  return thread_pool.apply_async(
      save_checkpoint,
      args=(filepath, tree),
      kwds=dict(overwrite=overwrite, makedirs=makedirs))


def save_multiple_checkpoints_async(
    filepath_tree_map: Mapping[str, Any],
    *,
    overwrite: bool = True,
    makedirs: bool = True,
    thread_pool: Optional[ThreadPool] = None) -> AsyncResult:
  thread_pool = thread_pool or ThreadPool()
  fn = functools.partial(
      save_checkpoint, overwrite=overwrite, makedirs=makedirs)
  return thread_pool.map_async(
      lambda args: fn(*args), filepath_tree_map.items())
