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

"""Library to perform adversarial attacks on V-MoE and ViT models."""
import collections
import io
import math
import os

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import jax
from jax.experimental import pjit
import ml_collections
import numpy as np
from vmoe import partitioning
from vmoe import utils
from vmoe.data import input_pipeline
from vmoe.data import pjit_utils
from vmoe.projects.adversarial_attacks import attacks
from vmoe.projects.adversarial_attacks import restore


ArraySpecDict = input_pipeline.ArraySpecDict
DatasetIterator = input_pipeline.DatasetIterator
Mesh = partitioning.Mesh
PartitionSpec = partitioning.PartitionSpec
gfile = input_pipeline.tf.io.gfile
VALID_KEY = input_pipeline.VALID_KEY


class ExampleStorage:
  """Storage to write examples to disk."""
  # TODO(jpuigcerver): This may cause OOM if the dataset is too big. Fix.

  def __init__(self, filepath: str, transfer_every_steps: int = 1):
    self._filepath = filepath
    self._transfer_every_steps = transfer_every_steps
    self._data = collections.defaultdict(list)
    self._temp = collections.defaultdict(list)

  def _transfer(self):
    for key, values in self._temp.items():
      self._data[key].extend(jax.tree_util.tree_map(np.asarray, values))
      self._temp[key] = []

  @property
  def filepath(self) -> str:
    return self._filepath

  def store(self, step: int, **arrays):
    for key, value in arrays.items():
      self._temp[key].append(value)
    if step % self._transfer_every_steps == 0:
      self._transfer()

  def write(self):
    self._transfer()
    if self._data:
      savez_compressed(
          self._filepath, **{
              key: np.concatenate(values, axis=0)
              for key, values in self._data.items()
          })


def get_dataset(config: ml_collections.ConfigDict) -> DatasetIterator:
  config = config.to_dict()
  _ = config.pop('prefetch_device', None)
  # Note: the variant name is not important as long it isn't "train".
  return input_pipeline.get_dataset(variant='adversarial', **config)


def savez_compressed(filepath: str, **data):
  """Wraps Numpy's savez to be able to work with non-seekable file systems."""
  outdir = os.path.dirname(filepath)
  if not gfile.exists(outdir):
    gfile.makedirs(outdir)
  # savez assumes that the file is seekable, which might not be always true.
  buffer = io.BytesIO()
  np.savez_compressed(buffer, **data)
  with gfile.GFile(filepath, 'wb') as fp:
    fp.write(buffer.getvalue())


def run_pgd_attack(
    config: ml_collections.ConfigDict, workdir: str, mesh: Mesh,
    writer: metric_writers.MetricWriter):
  """Run PGD attack on an entire dataset, using a model from an XM experiment."""
  del mesh
  # Setup dataset and get the global shape of the image array.
  dataset = get_dataset(config.dataset)
  element_spec: ArraySpecDict = dataset.element_spec  # pytype: disable=annotation-type-mismatch
  image_shape = tuple(element_spec['image'].shape)
  image_shape = (config.dataset.batch_size,) + image_shape[1:]
  num_examples = input_pipeline.get_data_num_examples(config.dataset)
  num_steps = int(math.ceil(num_examples / config.dataset.batch_size))
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_steps, on_steps=[], every_secs=60.0)
  # Restore the model and parameters to attack.
  if config.restore.get('from') == 'checkpoint':
    mesh = restore.create_mesh(config.get('num_expert_partitions', 1))
    partitioning.log_logical_mesh(mesh)
    (flax_module, variables, variables_axis_resources, loss_fn, router_keys,
     rng_keys) = restore.restore_from_config(
         config, config.restore.prefix, image_shape, mesh)
  else:
    raise ValueError('config.restore.from = '
                     f'{config.restore.get("from")!r} is not supported.')
  # Store examples before/after the attack. IMPORTANT: This may cause OOM if one
  # is processing a huge dataset.
  if config.get('store_examples'):
    examples_kwargs = config.store_examples.to_dict()
    examples_prefix_default = os.path.join(workdir, 'examples')
    examples_prefix = examples_kwargs.pop('prefix', examples_prefix_default)
    examples_filepath = f'{examples_prefix}-{jax.process_index():03d}-of-{jax.process_count():03d}'
    example_storage = ExampleStorage(
        filepath=examples_filepath, **examples_kwargs)
  else:
    example_storage = None
  # Prepare iterator to process the entire dataset, optionally pre-fetching data
  # to the TPU devices.
  data_axis_resources = PartitionSpec(mesh.axis_names)
  dataset_iter = pjit_utils.prefetch_to_device(
      iterator=dataset,
      size=config.dataset.get('prefetch_device'),
      mesh=mesh)
  # Initialize PGD state to track statistics over the entire dataset.
  def init_state():
    rngs = utils.make_rngs(rng_keys, config.get('seed', 0))
    return attacks.AttackState.create(
        max_updates=config.num_updates, router_keys=router_keys, rngs=rngs)
  state_axis_resources = jax.tree_util.tree_map(lambda _: PartitionSpec(),
                                                jax.eval_shape(init_state))
  init_state_pjit = pjit.pjit(
      init_state, in_shardings=(), out_shardings=state_axis_resources
  )

  def stateful_attack(variables, state, x, y, valid):

    def apply_fn(images, **kwargs):
      return flax_module.apply(variables, images, **kwargs)

    def stateless_attack_fn(images, labels, rngs):
      return attacks.stateless_attack_pgd(
          images, labels, rngs, apply_fn=apply_fn, loss_fn=loss_fn,
          max_epsilon=config.max_epsilon,
          num_updates=config.num_updates)

    def compute_loss_predict_correct_cw_fn(images, labels, rngs):
      return restore.compute_loss_predict_cw_fn(
          images, labels, rngs, apply_fn=apply_fn, loss_fn=loss_fn)

    return attacks.stateful_attack(
        state, x, y, valid, stateless_attack_fn=stateless_attack_fn,
        compute_loss_predict_correct_cw_fn=compute_loss_predict_correct_cw_fn)

  # Wrap pgd_attack function with pjit.
  stateful_attack_pjit = pjit.pjit(
      fun=stateful_attack,
      in_shardings=(
          variables_axis_resources,  # variables.
          state_axis_resources,  # input state.
          data_axis_resources,  # input images.
          data_axis_resources,  # true labels.
          data_axis_resources,  # valid mask.
      ),
      out_shardings=(
          state_axis_resources,  # output state
          data_axis_resources,  # images after the attack.
          data_axis_resources,  # loss before the attack.
          data_axis_resources,  # loss after the attack.
          data_axis_resources,  # predictions before the attack.
          data_axis_resources,  # predictions after the attack.
          data_axis_resources,  # combine weights before the attack.
          data_axis_resources,  # combine weights after the attack.
      ),
  )
  with mesh:
    state = init_state_pjit()
    for step, batch in enumerate(dataset_iter, 1):
      report_progress(step)
      # x_0, x_m, y_0, y_m, cw_0, cw_m: original and final images (x),
      # predictions (y) and combine weights for each MoE layer (cw, if any).
      x_0, y, valid = batch['image'], batch['labels'], batch[VALID_KEY]
      state, x_m, l_0, l_m, y_0, y_m, cw_0, cw_m = stateful_attack_pjit(
          variables, state, x_0, y, valid)
      if example_storage:
        example_storage.store(
            step=step, x_0=x_0, y_0=y_0, l_0=l_0, x_m=x_m, y_m=y_m, l_m=l_m,
            y=y, valid=valid,
            **{f'cw_0/{k}': v for k, v in cw_0.items()},
            **{f'cw_m/{k}': v for k, v in cw_m.items()})
  # Copy state from device to CPU and convert to numpy arrays.
  state = jax.tree_util.tree_map(np.asarray, state)
  # Process with index=0 saves the PGD state (it's the same for all processes).
  if jax.process_index() == 0:
    state_filepath = os.path.join(workdir, 'pgd_state.npz')
    logging.info('Saving PGDState to %r...', state_filepath)
    savez_compressed(state_filepath,
                     num_images=state.num_images,
                     num_changes=state.num_changes,
                     num_correct=state.num_correct,
                     sum_loss=state.sum_loss,
                     **{f'sum_iou_experts/{k}': v
                        for k, v in state.sum_iou_experts.items()})
  with jax.spmd_mode('allow_all'):
    # Statistics before the adversarial attack.
    writer.write_scalars(0, {
        'error': 1 - state.num_correct[0] / state.num_images,
        'loss': state.sum_loss[0] / state.num_images,
    })
    # Statistics after the adversarial attack.
    writer.write_scalars(config.num_updates, {
        'error': 1 - state.num_correct[1] / state.num_images,
        'loss': state.sum_loss[1] / state.num_images,
        'prediction_changes': state.num_changes / state.num_images,
        **{f'routing_changes/{k}': 1 - v / state.num_images
           for k, v in state.sum_iou_experts.items()}
    })
  # Each process optionally stores the processed examples.
  if example_storage:
    logging.info('Saving examples to %r...', example_storage.filepath)
    example_storage.write()
