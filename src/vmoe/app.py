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

"""Generic entry point for V-MoE projects that require their own main script.

This provides run() which performs some initialization and then calls the
provided main with the ConfigDict, the working directory, a JAX sharding Mesh,
and a CLU MetricWriter. We expect each project to have its own main.py.

Usage in your main.py:
  from vmoe import app

  def main(config: ml_collections.ConfigDict,
           workdir: str,
           mesh: jax.sharding.Mesh,
           writer: metric_writers.MetricWriter):
    # ...

  if __name__ == '__main__':
    app.run(main)
"""
import functools

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf
from vmoe import partitioning

flags.DEFINE_string('workdir', None, 'Directory to store logs and model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)
flags.mark_flags_as_required(['config', 'workdir'])
FLAGS = flags.FLAGS


def run(main):
  jax.config.config_with_absl()
  app.run(functools.partial(_main, main=main))


def _main(argv, *, main) -> None:
  """Runs the `main` method after some initial setup."""
  del argv
  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.set_visible_devices([], 'GPU')
  # Log JAX compilation steps.
  jax.config.update('jax_log_compiles', True)
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  # Log useful information to identify the process running in the logs.
  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())
  jax_xla_backend = ('None' if FLAGS.jax_xla_backend is None else
                     FLAGS.jax_xla_backend)
  logging.info('Using JAX XLA backend %s', jax_xla_backend)
  # Log the configuration passed to the main script.
  logging.info('Config: %s', FLAGS.config)
  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')
  # CLU metric writer.
  logdir = FLAGS.workdir
  writer = metric_writers.create_default_writer(
      logdir=logdir, just_logging=jax.process_index() > 0)
  # Set logical device mesh globally.
  mesh = partitioning.get_auto_logical_mesh(FLAGS.config.num_expert_partitions,
                                            jax.devices())
  partitioning.log_logical_mesh(mesh)
  with metric_writers.ensure_flushes(writer):
    with mesh:
      main(FLAGS.config, FLAGS.workdir, mesh, writer)
