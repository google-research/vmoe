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

"""External models wrapped to work with the V-MoE codebase."""
import flax.linen as nn
import ml_collections
from vit_jax import models


class MlpMixer(models.MlpMixer):
  """Official implementation of the MLP-Mixer."""
  deterministic: bool = False

  def __post_init__(self):
    # Note: The base class assumes that patches is a ConfigDict.
    self.patches = ml_collections.ConfigDict(self.patches)
    super().__post_init__()

  @nn.compact
  def __call__(self, inputs):
    return super().__call__(inputs, train=not self.deterministic), {}


class VisionTransformer(models.VisionTransformer):
  """Official implementation of the Vision Transformer."""
  deterministic: bool = False

  def __post_init__(self):
    # Note: The base class assumes that patches and resnet are ConfigDicts.
    self.patches = ml_collections.ConfigDict(self.patches)
    if self.resnet is not None:
      self.resnet = ml_collections.ConfigDict(self.resnet)
    super().__post_init__()

  @nn.compact
  def __call__(self, inputs):
    return super().__call__(inputs, train=not self.deterministic), {}
