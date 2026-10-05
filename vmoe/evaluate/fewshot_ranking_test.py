# Copyright 2026 Google LLC.
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

"""Tests for fewshot."""

"""Tie-aware selection tests for shared few-shot regularization."""

import itertools
import numpy as np
import pytest
from scipy import stats
from vmoe.evaluate import fewshot


@pytest.mark.parametrize(
    "order", list(itertools.permutations([0.01, 0.1, 1.0]))
)
def test_tied_datasets_cannot_overrule_consistently_better_regularizer(order):
  results = {
      "tie-a": {(1, reg): 0.5 for reg in order},
      "tie-b": {(1, reg): 0.5 for reg in order},
      "informative": {(1, 0.01): 0.9, (1, 0.1): 0.8, (1, 1.0): 0.6},
  }
  assert fewshot._find_best_l2_reg(results, [1], list(order)) == {1: 0.01}


def test_each_shot_uses_its_own_tie_aware_average_ranks():
  regs = [0.01, 0.1, 1.0, 10.0]
  scores = {
      "a": {1: [0.5, 0.5, 0.5, 0.5], 5: [0.3, 0.5, 0.5, 0.3]},
      "b": {1: [0.9, 0.8, 0.7, 0.6], 5: [0.1, 0.2, 0.9, 0.7]},
      "c": {1: [0.4, 0.4, 0.4, 0.4], 5: [0.4, 0.4, 0.4, 0.4]},
  }
  results = {
      key: {
          (shot, reg): values[shot][i]
          for shot in [1, 5]
          for i, reg in enumerate(regs)
      }
      for key, values in scores.items()
  }
  expected = {}
  for shot in [1, 5]:
    ranks = np.array(
        [
            stats.rankdata(values[shot], method="average")
            for values in scores.values()
        ]
    )
    expected[shot] = regs[np.argmax(ranks.mean(axis=0))]
  assert fewshot._find_best_l2_reg(results, [1, 5], regs) == expected


def test_unique_accuracy_ranks_are_unchanged():
  rng = np.random.default_rng(5)
  regs = [0.01, 0.1, 1.0, 10.0]
  for _ in range(25):
    scores = rng.random((3, 4))
    results = {
        str(i): {(1, reg): score for reg, score in zip(regs, row)}
        for i, row in enumerate(scores)
    }
    ranks = np.argsort(np.argsort(scores, axis=1), axis=1)
    expected = regs[np.argmax(ranks.mean(axis=0))]
    assert fewshot._find_best_l2_reg(results, [1], regs) == {1: expected}


def test_all_tied_scores_keep_the_first_regularizer_tiebreak():
  regs = [10.0, 0.1, 1.0]
  results = {
      "a": {(1, reg): 0.5 for reg in regs},
      "b": {(1, reg): 0.5 for reg in regs},
  }
  assert fewshot._find_best_l2_reg(results, [1], regs) == {1: regs[0]}


def test_single_regularizer_and_empty_shot_list():
  assert fewshot._find_best_l2_reg({"a": {(1, 0.1): 0.5}}, [1], [0.1]) == {
      1: 0.1
  }
  assert fewshot._find_best_l2_reg({}, [], [0.1]) == {}