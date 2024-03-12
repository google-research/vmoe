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

"""Classes and functions for performing adversarial attacks."""
from typing import Dict, Iterable, Mapping

import flax.struct
import jax
import jax.numpy as jnp


class AttackState(flax.struct.PyTreeNode):  # pytype: disable=invalid-function-definition  # dataclass_transform
  """State used to aggregate attack statistics over several batches of images.

  Attributes:
    max_updates: Integer, max number of updates during the attack.
    num_images: Scalar ndarray, total number of images processed.
    num_changes: Scalar ndarray, total number of prediction changes after the
      attack (w.r.t. the original prediction).
    num_correct: Vector ndarray of size 2, total number of correct predictions
      before and after the attack.
    sum_loss: Vector ndarray of size 2, with the sum of the losses of all
      examples before and after the attack.
    sum_iou_experts: Mapping from router names to scalar ndarrays, with the
      sum of average (over tokens) intersection over union of the experts
      selected before/after the attack.
    rngs: Dictionary of PRNGKeys to use if the model is not deterministic or
      in the adversarial attacks.
  """
  max_updates: int = flax.struct.field(pytree_node=False)
  num_images: jnp.ndarray
  num_changes: jnp.ndarray
  num_correct: jnp.ndarray
  sum_loss: jnp.ndarray
  sum_iou_experts: Mapping[str, jnp.ndarray]
  rngs: Dict[str, jnp.ndarray]

  @classmethod
  def create(
      cls, max_updates: int, router_keys: Iterable[str],
      rngs: Dict[str, jnp.ndarray]) -> 'AttackState':
    return AttackState(
        max_updates=max_updates,
        num_images=jnp.zeros((), dtype=jnp.int32),
        num_changes=jnp.zeros((), dtype=jnp.int32),
        num_correct=jnp.zeros((2,), dtype=jnp.int32),
        sum_loss=jnp.zeros((2,), dtype=jnp.float32),
        sum_iou_experts={
            key: jnp.zeros((), dtype=jnp.float32) for key in router_keys
        },
        rngs=rngs)

  def update(self, *, rngs, **kwargs) -> 'AttackState':
    new_kwargs = {}
    for k, v in kwargs.items():
      new_kwargs[k] = jax.tree_util.tree_map(
          lambda x, y: x + y, getattr(self, k), v)
    return AttackState(max_updates=self.max_updates, rngs=rngs, **new_kwargs)


def stateful_attack(attack_state, x, y, valid, *,
                    stateless_attack_fn, compute_loss_predict_correct_cw_fn):
  """Performs an adversarial attack on a batch of images, updating the state.

  Arguments:
    attack_state: AttackState object with the current stats to update.
    x: Input batch of images.
    y: One-hot encoding of the labels of the input batch of images.
    valid: Vector of booleans indicating whether the corresponding input image
      is valid (True) or fake (False).
    stateless_attack_fn: Function that takes the images, labels and a dictionary
      of PRNGKeys, and outputs the adversarial images and new PRNGKeys.
    compute_loss_predict_correct_cw_fn:  Function that takes the images, labels
      and a dictionary of PRNGKeys, and outputs for every element in the batch:
      the loss, the label prediction, whether the prediction is right or not,
      and the combine weights for every MoE layer.

  Returns:
    A new AttackState, the image after the attack, and then the loss, the
    predictions and the combine weights for every MoE layer before/after the
    attack.
  """
  assert x.ndim >= 1, f'{x.ndim=} is not greater than or equal to 1!'
  batch_size = x.shape[0]

  def mask(a):
    """Masks the input array with the (optional) valid array."""
    if valid is None:
      return a
    assert a.ndim >= 1, f'{a.ndim=} is not greater than or equal to 1!'
    assert valid.shape[0] == a.shape[0], (
        f'{valid.shape=} is not broadcastable to {a.shape=}')
    v = valid.reshape((batch_size,) + (1,) * max(a.ndim - valid.ndim, 0))
    return a * v

  # Attack the input images using the provide stateless attack function.
  x_0 = x
  rngs = attack_state.rngs
  x_m, new_rngs = stateless_attack_fn(x_0, y, rngs)
  # Compute the loss, the predictions, the correct predictions and the combine
  # weights of every MoE layer (if any).
  l_0, y_0, c_0, cw_0 = compute_loss_predict_correct_cw_fn(x_0, y, rngs)
  l_m, y_m, c_m, cw_m = compute_loss_predict_correct_cw_fn(x_m, y, new_rngs)
  # Mask outputs from the compute_loss_predict_correct_cw_fn function.
  l_0, y_0, c_0, cw_0 = jax.tree_util.tree_map(mask, (l_0, y_0, c_0, cw_0))
  l_m, y_m, c_m, cw_m = jax.tree_util.tree_map(mask, (l_m, y_m, c_m, cw_m))
  # Update AttackState.
  num_images = batch_size if valid is None else jnp.sum(valid)
  num_changes = jnp.sum(y_m != y_0, dtype=jnp.int32)
  num_correct = jnp.stack([c_0.sum(), c_m.sum()], axis=0)
  sum_loss = jnp.stack([l_0.sum(), l_m.sum()], axis=0)
  sum_iou_experts = jax.tree_util.tree_map(
      sum_intersection_over_union, cw_0, cw_m)
  new_attack_state = attack_state.update(
      rngs=new_rngs, num_images=num_images, num_changes=num_changes,
      num_correct=num_correct, sum_loss=sum_loss,
      sum_iou_experts=sum_iou_experts)
  return new_attack_state, x_m, l_0, l_m, y_0, y_m, cw_0, cw_m


def stateless_attack_pgd(
    images, labels, rngs, *, max_epsilon, num_updates, apply_fn, loss_fn):
  """Performs a PGD attack on a batch of images and returns the modified ones.

  Arguments:
    images: Input batch of images.
    labels: Labels of the input batch of images.
    rngs: PyTree containing the PRNGKeys passed to the apply_fn. It can be None.
    max_epsilon: Maximum change for each pixel.
    num_updates: Number of updates performed in the attack.
    apply_fn: Function that applies a model on a batch of images. It must return
      an array of logits and a dictionary with metrics.
    loss_fn: The loss to adversarially attack, including the auxiliary loss
      terms (if applicable).

  Returns:
    The images after the last adversarial attack, a PyTree of PRNGKeys.
  """
  # How much we can change the image on each update.
  delta = max_epsilon / num_updates
  # If rngs are given, split them in num_updates + 1. The last one will be
  # returned at the end of this function, the others are used in each update.
  if rngs is not None:
    rngs = jax.tree_util.tree_map(
        lambda rng: jax.random.split(rng, num_updates +1), rngs)
  # This computes gradients of loss_fn w.r.t. the images.
  @jax.grad
  def compute_loss_grads(x, rngs):
    logits, metrics = apply_fn(x, rngs=rngs)
    loss = jnp.mean(loss_fn(logits, labels, metrics))
    return loss
  # Performs an adversarial update on the given images.
  def update(i, x_c):
    rngs_c = jax.tree_util.tree_map(
        lambda x: x[i], rngs) if rngs is not None else None
    dx = compute_loss_grads(x_c, rngs_c)
    x_n = x_c + delta * jnp.sign(dx)
    return x_n
  # Performs num_updates on the original images.
  images = jax.lax.fori_loop(0, num_updates, update, images)
  rngs = jax.tree_util.tree_map(
      lambda x: x[-1], rngs) if rngs is not None else None
  return images, rngs


def sum_intersection_over_union(a, b):
  """Returns the sum over examples of the intersection over union."""
  # a, b have shape (batch_size, num_tokens, num_experts).
  assert a.ndim == 3, f'a.ndim = {a.ndim}'
  assert a.shape == b.shape, f'a.shape = {a.shape} vs. b.shape = {b.shape}'
  num_tokens = a.shape[1]
  a = jnp.asarray(a > 0.0, dtype=jnp.float32)
  b = jnp.asarray(b > 0.0, dtype=jnp.float32)
  # The shape of the following arrays is (batch_size, num_tokens).
  a_intersection_b = jnp.sum(a * b, axis=-1)
  a_union_b = jnp.sum(a + b, axis=-1) - a_intersection_b
  a_iou_b = jnp.where(a_union_b > 0, a_intersection_b / a_union_b, 0)
  # This is the sum (over examples) of the average IoU (over tokens) of the
  # selected experts.
  return jnp.sum(a_iou_b) / num_tokens
