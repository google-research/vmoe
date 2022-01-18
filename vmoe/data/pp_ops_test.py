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

"""Tests for pp_ops."""
import io

import imageio
import numpy as np
import tensorflow.compat.v1 as tf
from vmoe.data import pp_ops


class PreprocessOpsTest(tf.test.TestCase):

  def get_data(self, dtype=tf.uint8):
    return {'image': tf.random.uniform([640, 480, 3], 0, 255)}

  def test_resize(self):
    data = self.get_data()
    data = pp_ops.resize([120, 80])(data)
    self.assertEqual(data['image'].numpy().shape, (120, 80, 3))

  def test_resize_small(self):
    data = self.get_data()
    data = pp_ops.resize_small(240)(data)
    self.assertEqual(data['image'].numpy().shape, (320, 240, 3))

  def test_decode_jpeg_and_inception_crop(self):
    f = io.BytesIO()
    imageio.imsave(
        f,
        np.random.randint(0, 256, [224, 224, 3]).astype('uint8'),
        format='jpg')
    data = {'image': f.getvalue()}
    output = pp_ops.decode_jpeg_and_inception_crop(128)(data)['image']
    self.assertEqual(output.shape, (128, 128, 3))

  def test_inception_crop(self):
    image = np.random.randint(0, 256, [224, 224, 3]).astype('uint8')
    output = pp_ops.inception_crop(128)({'image': image})['image'].numpy()
    self.assertEqual(output.shape, (128, 128, 3))

  def test_central_crop(self):
    data = self.get_data()
    data = pp_ops.central_crop([120, 80])(data)
    self.assertEqual(data['image'].numpy().shape, (120, 80, 3))

  def test_flip_lr(self):
    data = self.get_data()
    data_after_pp = pp_ops.flip_lr()(data)
    self.assertTrue(
        np.all(data['image'].numpy() == data_after_pp['image'].numpy()) or
        np.all(data['image'][:, ::-1].numpy() ==
               data_after_pp['image'].numpy()))

  def test_value_range(self):
    data = self.get_data()
    data = pp_ops.value_range(-0.5, 0.5)(data)
    self.assertLessEqual(np.max(data['image'].numpy()), 0.5)
    self.assertGreaterEqual(np.min(data['image'].numpy()), -0.5)

  def test_value_range_custom_input_range(self):
    data = self.get_data()
    data = pp_ops.value_range(-0.5, 0.5, -256, 255, True)(data)
    self.assertLessEqual(np.max(data['image'].numpy()), 0.5)
    self.assertGreaterEqual(np.min(data['image'].numpy()), 0.0)

  def test_keep(self):
    data = {'image': 1, 'labels': 2, 'something': 3}

    data_keep = pp_ops.keep('image', 'labels')(data)
    self.assertAllEqual(list(data_keep.keys()), ['image', 'labels'])

  def test_onehot(self):
    data = {'labels': tf.constant(np.asarray(2), dtype=tf.int64)}
    output_data = pp_ops.onehot(4, multi=True)(data)
    self.assertAllClose(output_data['labels'].numpy(), np.asarray(
        [0., 0., 1., 0.], dtype=np.float32))

  def test_onehot_multi(self):
    data = {'labels': tf.constant(np.asarray([2, 3, 0]), dtype=tf.int64)}
    output_data = pp_ops.onehot(4, multi=False)(data)
    self.assertAllClose(output_data['labels'].numpy(), np.asarray([
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.]], dtype=np.float32))

    data = {'labels': tf.constant(np.asarray([2, 3, 0]), dtype=tf.int64)}
    output_data = pp_ops.onehot(4, multi=True)(data)
    self.assertAllClose(output_data['labels'].numpy(),
                        np.asarray([1., 0., 1., 1.], dtype=np.float32))

  def test_onehot_smoothing(self):
    data = {'labels': tf.constant(np.asarray([2, 3, 0]), dtype=tf.int64)}
    output_data = pp_ops.onehot(4, multi=False, on=0.8, off=0.1)(data)
    self.assertAllClose(output_data['labels'].numpy(), np.asarray([
        [0.1, 0.1, 0.8, 0.1],
        [0.1, 0.1, 0.1, 0.8],
        [0.8, 0.1, 0.1, 0.1]], dtype=np.float32))

    data = {'labels': tf.constant(np.asarray([2, 3, 0]), dtype=tf.int64)}
    output_data = pp_ops.onehot(4, multi=True, on=0.8, off=0.1)(data)
    self.assertAllClose(output_data['labels'].numpy(),
                        np.asarray([0.8, 0.1, 0.8, 0.8], dtype=np.float32))


if __name__ == '__main__':
  tf.test.main()
