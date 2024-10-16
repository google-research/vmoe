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

"""Functions to evaluate ensembles."""
from typing import Callable

import jax
import jax.numpy as jnp
import optax

Array = jnp.ndarray


def _multiply_no_nan(x: Array, y: Array) -> Array:
  """Multiplies x and y and returns 0 if x is 0, even if y is not finite."""
  x_ok = x != 0.
  safe_x = jnp.where(x_ok, x, 1.)
  safe_y = jnp.where(x_ok, y, 1.)
  return jnp.where(x_ok, jax.lax.mul(safe_x, safe_y), jnp.zeros_like(x))


def _ensemble_log_mean(
    logits: Array, ensemble_size: int,
    log_normalization_fn: Callable[[Array], Array]) -> Array:
  """Computes the log ensemble probability.

  Args:
    logits: 2D tensor of shape [batch size * ensemble size, #classes]. It is
      assumed that the batches for each ensemble member are stacked with a
      jnp.repeat(..., ensemble_size, axis=0).
    ensemble_size: The size of the ensemble.
    log_normalization_fn: Normalization function used to normalize the last axis
      of the logits into a probability distribution. Common choices are
      `jax.nn.log_softmax` or `jax.nn.log_sigmoid`.
  Returns:
    log ensemble probability.
  """
  _, num_classes = logits.shape
  logits = jnp.reshape(logits, (-1, ensemble_size, num_classes))
  log_ens_size = jnp.log(ensemble_size)
  log_p = log_normalization_fn(logits)
  return jax.nn.logsumexp(log_p, axis=1) - log_ens_size


def ensemble_softmax_xent_train(repeated_logits: Array, labels: Array,
                                ensemble_size: int) -> Array:
  """At train time, the ensemble uses the standard softmax cross entropy.

  The logits of the ensemble model already account for the ensemble size. This
  is not the case of the labels that need to be repeated.

  Args:
    repeated_logits: 2D tensor of shape [ensemble size * batch size, #classes].
      It is assumed that the batches for each ensemble member are stacked with a
      jnp.repeat(..., ensemble_size, axis=0).
    labels: 2D tensor of labels [batch size, #classes].
    ensemble_size: The size of the ensemble.
  Returns:
    ensemble softmax cross entropy used at training time.
  """
  repeated_labels = jnp.repeat(labels, ensemble_size, axis=0)
  return optax.softmax_cross_entropy(repeated_logits, repeated_labels)  # pytype: disable=bad-return-type  # numpy-scalars


def ensemble_sigmoid_xent_train(repeated_logits: Array, labels: Array,
                                ensemble_size: int) -> Array:
  """At train time, the ensemble uses the standard sigmoid cross entropy.

  The logits of the ensemble model already account for the ensemble size. This
  is not the case of the labels that need to be repeated.

  Args:
    repeated_logits: 2D tensor of shape [ensemble size * batch size, #classes].
      It is assumed that the batches for each ensemble member are stacked with a
      jnp.repeat(..., ensemble_size, axis=0).
    labels: 2D tensor of labels [batch size, #classes].
    ensemble_size: The size of the ensemble.
  Returns:
    ensemble sigmoid cross entropy used at training time.
  """
  repeated_labels = jnp.repeat(labels, ensemble_size, axis=0)
  losses = optax.sigmoid_binary_cross_entropy(repeated_logits, repeated_labels)
  return jnp.sum(losses, axis=-1)


def ensemble_softmax_xent_eval(logits: Array, labels: Array,
                               ensemble_size: int) -> Array:
  """Ensemble version of the softmax cross entropy.

  Args:
    logits: 2D tensor of shape [ensemble size * batch size, #classes]. It is
      assumed that the batches for each ensemble member are stacked with a
      jnp.repeat(..., ensemble_size, axis=0).
    labels: 2D tensor of labels [batch size, #classes].
    ensemble_size: The size of the ensemble.
  Returns:
    ensemble softmax cross entropy (typically used at evaluation time).
  """
  log_p = _ensemble_log_mean(logits, ensemble_size, jax.nn.log_softmax)
  xent = _multiply_no_nan(labels, log_p)  # Shape: (B, C).
  return -jnp.sum(xent, axis=-1)  # Shape: (B,).


def ensemble_sigmoid_xent_eval(logits: Array, labels: Array,
                               ensemble_size: int) -> Array:
  """Ensemble version of the sigmoid cross entropy.

  Args:
    logits: 2D tensor of shape [ensemble size * batch size, #classes]. It is
      assumed that the batches for each ensemble member are stacked with a
      jnp.repeat(..., ensemble_size, axis=0).
    labels: 2D tensor of labels [batch size, #classes].
    ensemble_size: The size of the ensemble.
  Returns:
    ensemble sigmoid cross entropy (typically used at evaluation time).
  """
  log_p = _ensemble_log_mean(logits, ensemble_size, jax.nn.log_sigmoid)
  log_not_p = _ensemble_log_mean(-logits, ensemble_size, jax.nn.log_sigmoid)
  return -jnp.sum(labels * log_p + (1. - labels) * log_not_p, axis=-1)


def label_pred_ensemble_softmax(logits: Array, ensemble_size: int) -> Array:
  """Function to select the predicted labels for the ensemble softmax CE.

  Args:
    logits: 2D tensor of shape [ensemble size * batch size, #classes]. It is
      assumed that the batches for each ensemble member are stacked with a
      jnp.repeat(..., ensemble_size, axis=0).
    ensemble_size: The size of the ensemble.
  Returns:
    The class labels to predict.
  """
  log_p = _ensemble_log_mean(logits, ensemble_size, jax.nn.log_softmax)
  return jnp.argmax(log_p, axis=1)  # Shape: (B,).


def label_pred_ensemble_sigmoid(logits: Array, ensemble_size: int) -> Array:
  """Function to select the predicted labels for the ensemble sigmoid CE.

  Args:
    logits: 2D tensor of shape [ensemble size * batch size, #classes]. It is
      assumed that the batches for each ensemble member are stacked with a
      jnp.repeat(..., ensemble_size, axis=0).
    ensemble_size: The size of the ensemble.
  Returns:
    The class labels to predict.
  """
  log_p = _ensemble_log_mean(logits, ensemble_size, jax.nn.log_sigmoid)
  return jnp.argmax(log_p, axis=1)  # Shape: (B,).
