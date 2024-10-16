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

"""Tests for utils."""
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from vmoe import utils


class ParseCallTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_name', 'parse_call', 'vmoe.utils', utils.parse_call, (), {}),
      ('_attribute', 'numpy.ceil', 'foo', np.ceil, (), {}),
  )
  def test(self, string, default_module, expect_obj, expect_args, expect_kwarg):
    obj, args, kwargs = utils.parse_call(string, default_module)
    self.assertIs(obj, expect_obj)
    self.assertEqual(args, expect_args)
    self.assertEqual(kwargs, expect_kwarg)

  @parameterized.named_parameters(
      ('expression', '2 + 2', 'foo'),
      ('callable_in_attribute', 'foo(16).bar.baz', 'foo'),
  )
  def test_raises(self, string, default_module):
    with self.assertRaises(ValueError):
      utils.parse_call(string, default_module)


class UtilsTest(absltest.TestCase):

  def test_multiply_no_nan(self):
    x = np.asarray([0.0, 0.0, 2.0])
    y = np.asarray([np.nan, np.inf, 3.0])
    np.testing.assert_allclose(utils.multiply_no_nan(x, y), [0.0, 0.0, 6.0])
    dx = jax.grad(lambda x: utils.multiply_no_nan(x, y).sum())(x)
    np.testing.assert_allclose(dx, [0.0, 0.0, 3.0])
    dy = jax.grad(lambda y: utils.multiply_no_nan(x, y).sum())(y)
    np.testing.assert_allclose(dy, [0.0, 0.0, 2.0])

  def test_partialclass(self):

    @dataclasses.dataclass
    class A:
      a: int
      b: int

    cls = utils.partialclass(A, a=1)
    self.assertTrue(issubclass(cls, A))
    obj = cls(b=2)
    self.assertIsInstance(obj, A)
    self.assertEqual(obj.a, 1)
    self.assertEqual(obj.b, 2)

  def test_safe_map(self):
    self.assertEqual(
        list(utils.safe_map(lambda x, y: x + y, [1, 3], [2, 4])),
        [1 + 2, 3 + 4])

  def test_safe_zip(self):
    self.assertIsInstance(utils.safe_zip([], []), utils.SafeZipIterator)
    self.assertEqual(list(utils.safe_zip([], [])), [])
    self.assertEqual(list(utils.safe_zip([1])), [(1,)])
    self.assertEqual(list(utils.safe_zip([1, 2], ['a', 'b'])),
                     [(1, 'a'), (2, 'b')])
    with self.assertRaises(utils.SafeZipIteratorError):
      list(utils.safe_zip([], ['a'], [1.]))
    with self.assertRaises(utils.SafeZipIteratorError):
      list(utils.safe_zip([1], [], [1.]))
    with self.assertRaises(utils.SafeZipIteratorError):
      list(utils.safe_zip([1], ['a'], []))

  def test_make_match_fn_from_regex_list(self):
    regexes = ['ab', 'c']
    match_fn = utils.make_match_fn_from_regex_list(regexes)
    self.assertFalse(match_fn('d'))
    self.assertFalse(match_fn('adb'))
    self.assertTrue(match_fn('acb'))    # c is the matching regex.
    self.assertTrue(match_fn('abd'))   # ab is the matching regex.
    self.assertTrue(match_fn('dabd'))  # ab is the matching regex.

    regexes = 'ab'
    match_fn = utils.make_match_fn_from_regex_list(regexes)
    self.assertFalse(match_fn('cb'))
    self.assertFalse(match_fn('ac'))
    self.assertTrue(match_fn('aab'))    # ab is the matching regex.
    self.assertTrue(match_fn('abd'))   # ab is the matching regex.


if __name__ == '__main__':
  absltest.main()
