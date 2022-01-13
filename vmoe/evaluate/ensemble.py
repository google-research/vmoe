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

"""Functions to evaluate ensembles."""
import jax
import jax.numpy as jnp

Array = jnp.ndarray


def _multiply_no_nan(x: Array, y: Array) -> Array:
  """Multiplies x and y and returns 0 if x is 0, even if y is not finite."""
  x_ok = x != 0.
  safe_x = jnp.where(x_ok, x, 1.)
  safe_y = jnp.where(x_ok, y, 1.)
  return jnp.where(x_ok, jax.lax.mul(safe_x, safe_y), jnp.zeros_like(x))


def _get_log_ensemble_softmax(logits: Array, ensemble_size: int) -> Array:
  """Computes the log ensemble softmax probability.

  Args:
    logits: 2D tensor of shape [ensemble size * batch size, #classes]. It is
      assumed that the batches for each ensemble member are stacked with a
      jnp.repeat(..., ensemble_size) pattern and can thus be recovered by
      an appropriate slicing ...[::ensemble_size].
    ensemble_size: The size of the ensemble.
  Returns:
    log ensemble softmax probability.
  """
  logits3d = [logits[e::ensemble_size] for e in range(ensemble_size)]
  logits3d = jnp.asarray(logits3d)  # Shape: (E, B, C).
  log_ens_size = jnp.log(ensemble_size)
  # E = ensemble size, B = batch size, C = number of classes.
  log_p = jax.nn.log_softmax(logits3d)  # Shape: (E, B, C).
  log_p = jax.nn.logsumexp(log_p, axis=0) - log_ens_size  # Shape: (B, C).
  return log_p


def ensemble_softmax_xent(logits: Array, labels: Array,
                          ensemble_size: int) -> Array:
  """Ensemble version of the softmax cross entropy.

  Args:
    logits: 2D tensor of shape [ensemble size * batch size, #classes]. It is
      assumed that the batches for each ensemble member are stacked with a
      jnp.repeat(..., ensemble_size) pattern and can thus be recovered by
      an appropriate slicing ...[::ensemble_size].
    labels: 2D tensor of labels [batch size, #classes].
    ensemble_size: The size of the ensemble.
  Returns:
    ensemble softmax cross entropy.
  """
  log_p = _get_log_ensemble_softmax(logits, ensemble_size)  # Shape: (B, C).
  xent = _multiply_no_nan(labels, log_p)  # Shape: (B, C).
  return -jnp.sum(xent, axis=-1)  # Shape: (B,).


def label_pred_ensemble_softmax(logits: Array, ensemble_size: int) -> Array:
  """Function to select the predicted labels for the ensemble softmax CE.

  Args:
    logits: 2D tensor of shape [ensemble size * batch size, #classes]. It is
      assumed that the batches for each ensemble member are stacked with a
      jnp.repeat(..., ensemble_size) pattern and can thus be recovered by
      an appropriate slicing ...[::ensemble_size].
    ensemble_size: The size of the ensemble.
  Returns:
    The class labels to predict.
  """
  log_p = _get_log_ensemble_softmax(logits, ensemble_size)  # Shape: (B, C).
  return jnp.argmax(log_p, axis=1)  # Shape: (B,).
