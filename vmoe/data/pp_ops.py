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

"""Implementation of data processing ops.

All ops should return data processing functors. Data examples are represented
as a dictionary of tensors.

Most of these were originally implemented by: Lucas Beyer, Alex Kolesnikov,
Xiaohua Zhai and other collaborators from Google Brain Zurich.
"""
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

try:
  from cloud_tpu.models.efficientnet import autoaugment  # pylint: disable=g-import-not-at-top
except ImportError:
  autoaugment = None


class InKeyOutKey(object):
  """Decorator for preprocessing ops, which adds `inkey` and `outkey` arguments.

  Note: Only supports single-input single-output ops.
  """

  def __init__(self, indefault='image', outdefault='image'):
    self.indefault = indefault
    self.outdefault = outdefault

  def __call__(self, orig_get_pp_fn):

    def get_ikok_pp_fn(*args, key=None,
                       inkey=self.indefault, outkey=self.outdefault, **kw):
      orig_pp_fn = orig_get_pp_fn(*args, **kw)
      def _ikok_pp_fn(data):
        data[key or outkey] = orig_pp_fn(data[key or inkey])
        return data

      return _ikok_pp_fn

    return get_ikok_pp_fn


@InKeyOutKey()
def central_crop(crop_size):
  """Makes central crop of a given size.

  Args:
    crop_size: either an integer H, where H is both the height and width of the
      central crop, or a list or tuple [H, W] of integers, where H and W are
      height and width of the central crop respectively.

  Returns:
    A function, that applies central crop.
  """
  if isinstance(crop_size, int):
    crop_size = (crop_size, crop_size)
  crop_size = tuple(crop_size)

  def _crop(image):
    h, w = crop_size[0], crop_size[1]
    dy = (tf.shape(image)[0] - h) // 2
    dx = (tf.shape(image)[1] - w) // 2
    return tf.image.crop_to_bounding_box(image, dy, dx, h, w)

  return _crop


def copy(inkey, outkey):
  """Copies value of `inkey` into `outkey`."""

  def _copy(data):
    data[outkey] = data[inkey]
    return data

  return _copy


@InKeyOutKey()
def decode(channels=3):
  """Decodes an encoded image string, see tf.io.decode_image."""

  def _decode(image):
    return tf.io.decode_image(image, channels=channels, expand_animations=False)

  return _decode


@InKeyOutKey()
def decode_jpeg_and_inception_crop(resize_size=None, area_min=5, area_max=100):
  """Decodes jpeg string and makes inception-style image crop.

  See `inception_crop` for details.

  Args:
    resize_size: Resize image to this size after crop.
    area_min: minimal crop area.
    area_max: maximal crop area.

  Returns:
    A function, that applies inception crop.
  """

  def _inception_crop(image_data):  # pylint: disable=missing-docstring
    shape = tf.image.extract_jpeg_shape(image_data)
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(area_min / 100, area_max / 100),
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(begin)
    target_height, target_width, _ = tf.unstack(size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_data, crop_window, channels=3)

    if resize_size:
      image = resize(resize_size)({'image': image})['image']

    return image

  return _inception_crop


@InKeyOutKey()
def flip_lr():
  """Flips an image horizontally with probability 50%."""

  def _random_flip_lr_pp(image):
    return tf.image.random_flip_left_right(image)

  return _random_flip_lr_pp


@InKeyOutKey()
def inception_crop(resize_size=None, area_min=5, area_max=100,
                   resize_method='bilinear'):
  """Makes inception-style image crop.

  Inception-style crop is a random image crop (its size and aspect ratio are
  random) that was used for training Inception models, see
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

  Args:
    resize_size: Resize image to this size after crop.
    area_min: minimal crop area.
    area_max: maximal crop area.
    resize_method: rezied method, see tf.image.resize docs for options.

  Returns:
    A function, that applies inception crop.
  """

  def _inception_crop(image):  # pylint: disable=missing-docstring
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(area_min / 100, area_max / 100),
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)
    crop = tf.slice(image, begin, size)
    # Unfortunately, the above operation loses the depth-dimension. So we need
    # to restore it the manual way.
    crop.set_shape([None, None, image.shape[-1]])
    if resize_size:
      crop = resize(resize_size, resize_method)({'image': crop})['image']
    return crop

  return _inception_crop


def keep(*keys):
  """Keeps only the given keys."""

  def _keep(data):
    return {k: v for k, v in data.items() if k in keys}

  return _keep


@InKeyOutKey(indefault='labels', outdefault='labels')
def onehot(depth, multi=True, on=1.0, off=0.0):
  """One-hot encodes the input.

  Args:
    depth: Length of the one-hot vector (how many classes).
    multi: If there are multiple labels, whether to merge them into the same
      "multi-hot" vector (True) or keep them as an extra dimension (False).
    on: Value to fill in for the positive label (default: 1).
    off: Value to fill in for negative labels (default: 0).

  Returns:
    Data dictionary.
  """

  def _onehot(label):
    # When there's more than one label, this is significantly more efficient
    # than using tf.one_hot followed by tf.reduce_max; we tested.
    if label.shape.rank > 0 and multi:
      x = tf.scatter_nd(label[:, None],
                        tf.ones(tf.shape(label)[0]), (depth,))
      x = tf.clip_by_value(x, 0, 1) * (on - off) + off
    else:
      x = tf.one_hot(label, depth, on_value=on, off_value=off)
    return x

  return _onehot


@InKeyOutKey()
def randaug(num_layers: int = 2, magnitude: int = 10):
  """Creates a function that applies RandAugment.

  RandAugment is from the paper https://arxiv.org/abs/1909.13719,

  Args:
    num_layers: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper. Usually best
      values will be in the range [1, 3].
    magnitude: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range
      [5, 30].

  Returns:
    a function that applies RandAugment.
  """
  if autoaugment is None:
    raise ValueError(
        "In order to use RandAugment you need to install the 'cloud_tpu' "
        "package. Clone the https://github.com/tensorflow/tpu repository, "
        "name it 'cloud_tpu', and add the corresponding directory to your "
        "PYTHONPATH.")

  def _randaug(image):
    return autoaugment.distort_image_with_randaugment(
        image, num_layers, magnitude)

  return _randaug


@InKeyOutKey()
def resize(resize_size, resize_method='bilinear'):
  """Resizes image to a given size.

  Args:
    resize_size: either an integer H, where H is both the new height and width
      of the resized image, or a list or tuple [H, W] of integers, where H and W
      are new image"s height and width respectively.
    resize_method: rezied method, see tf.image.resize docs for options.

  Returns:
    A function for resizing an image.

  """
  if isinstance(resize_size, int):
    resize_size = (resize_size, resize_size)
  resize_size = tuple(resize_size)

  def _resize(image):
    # Note: use TF-2 version of tf.image.resize as the version in TF-1 is
    # buggy: https://github.com/tensorflow/tensorflow/issues/6720.
    # In particular it was not equivariant with rotation and lead to the network
    # to learn a shortcut in self-supervised rotation task, if rotation was
    # applied after resize.
    dtype = image.dtype
    image = tf2.image.resize(image, resize_size, resize_method)
    return tf.cast(image, dtype)

  return _resize


@InKeyOutKey()
def resize_small(smaller_size):
  """Resizes the smaller side to `smaller_size` keeping aspect ratio.

  Args:
    smaller_size: an integer, that represents a new size of the smaller side of
      an input image.

  Returns:
    A function, that resizes an image and preserves its aspect ratio.
  """

  def _resize_small(image):  # pylint: disable=missing-docstring
    h, w = tf.shape(image)[0], tf.shape(image)[1]

    # Figure out the necessary h/w.
    ratio = (
        tf.cast(smaller_size, tf.float32) /
        tf.cast(tf.minimum(h, w), tf.float32))
    h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
    w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)

    return tf.image.resize_area(image[None], [h, w])[0]

  return _resize_small


@InKeyOutKey()
def value_range(vmin, vmax, in_min=0, in_max=255.0, clip_values=False):
  """Transforms a [in_min,in_max] image to [vmin,vmax] range.

  Input ranges in_min/in_max can be equal-size lists to rescale the invidudal
  channels independently.

  Args:
    vmin: A scalar. Output max value.
    vmax: A scalar. Output min value.
    in_min: A scalar or a list of input min values to scale. If a list, the
      length should match to the number of channels in the image.
    in_max: A scalar or a list of input max values to scale. If a list, the
      length should match to the number of channels in the image.
    clip_values: Whether to clip the output values to the provided ranges.

  Returns:
    A function to rescale the values.
  """

  def _value_range(image):
    """Scales values in given range."""
    in_min_t = tf.constant(in_min, tf.float32)
    in_max_t = tf.constant(in_max, tf.float32)
    image = tf.cast(image, tf.float32)
    image = (image - in_min_t) / (in_max_t - in_min_t)
    image = vmin + image * (vmax - vmin)
    if clip_values:
      image = tf.clip_by_value(image, vmin, vmax)
    return image

  return _value_range
