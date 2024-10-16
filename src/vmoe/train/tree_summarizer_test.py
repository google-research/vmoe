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

from absl.testing import absltest
from absl.testing import parameterized
import chex
import numpy as np

from vmoe.train import tree_summarizer


class TreeSummarizerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # Summarize the entire input array into a single value.
      ('none', [[1], [2], [3]], (), '', [[1], [2], [3]]),
      ('sum', [[1], [2], [3]], ('sum',), '_sum', 6),
      ('min', [[1], [2], [3]], ('min',), '_min', 1),
      ('max', [[1], [2], [3]], ('max',), '_max', 3),
      ('mean', [[1], [2], [3]], ('mean',), '_mean', 2),
      ('std', [[1], [2], [3]], ('std',), '_std', np.sqrt(2 / 3)),
      ('norm', [[1], [2], [3]], ('norm',), '_norm', np.sqrt(1 + 4 + 9)),
      # Summarize an array along a subset of dimensions.
      ('_norm_0', [[1, 2], [3, 4]], (('norm', 0),), '_norm(0,)',
       np.sqrt([[10, 20]])),
      ('_norm_(1,)', [[1, 2], [3, 4]], (('norm', (1,)),), '_norm(1,)',
       np.sqrt([[5], [25]])),
      ('norm_(1, 0)', [[1], [2], [3]], (('norm', (1, 0)),), '_norm(1, 0)',
       np.sqrt(1 + 4 + 9)),
      # Sequence of transformations to summarize an array.
      ('_min_0_max', [[1, 2], [3, 4]], (('min', 0), 'max'), '_min(0,)_max', 2),
  )
  def test_transform(self, value, transforms, expected_prefix, expected_output):
    value = np.asarray(value, dtype=np.float32)
    expected_output = np.asarray(expected_output, dtype=np.float32)
    prefix, output = tree_summarizer.TreeSummarizer._transform(value,
                                                               transforms)
    self.assertEqual(prefix, expected_prefix)
    chex.assert_trees_all_close(output, expected_output)

  def test_transform_raises(self):
    with self.assertRaisesRegex(ValueError, "Operation op='foo'"):
      tree_summarizer.TreeSummarizer._transform(np.zeros(5,), ('foo',))

  @parameterized.named_parameters(
      ('no_values', [], {}),
      ('single_value', [[3.]], {'foo': 3.}),
      ('multiple_values', [[3.], [4.]], {'foo[0]': 3., 'foo[1]': 4.}),
  )
  def test_yield(self, value, expected):
    prefix, value = 'foo', np.asarray(value, dtype=np.float32)
    output = dict(tree_summarizer.TreeSummarizer._yield(prefix, value))
    chex.assert_trees_all_close(output, expected)

  def test_summarize(self):
    rules = [
        '.*/learning_rate',
        ('.*/dense/w', 'norm'),
        ('.*/moe/w', ('norm', (1,))),
    ]
    tree = {
        'params': {
            'dense': {'w': np.asarray([[1, 2, 3, 4]], dtype=np.float32)},
            'moe': {'w': np.asarray([[1, 2], [3, 4]], dtype=np.float32)},
        },
        'opt_state': {
            'learning_rate': np.asarray(1, dtype=np.float32),
        }
    }
    summarizer = tree_summarizer.TreeSummarizer(
        rules=rules,
        # We will print two values for each moe array, update this so it doesn't
        # complain.
        max_summary_values=2,
    )
    output = dict(summarizer(tree))
    expected_output = {
        'params/dense/w_norm': np.sqrt(30),
        'params/moe/w_norm(1,)[0]': np.sqrt(5),
        'params/moe/w_norm(1,)[1]': np.sqrt(25),
        'opt_state/learning_rate': 1.,
    }
    chex.assert_trees_all_close(output, expected_output)

  def test_summarize_raises(self):
    rules = [('.*/bar',)]
    tree = {'foo': {'bar': np.asarray([1, 2], dtype=np.float32)}}
    summarizer = tree_summarizer.TreeSummarizer(rules)
    with self.assertRaisesRegex(ValueError, 'generates 2 > 1 summary values'):
      _ = dict(summarizer(tree))


if __name__ == '__main__':
  absltest.main()
