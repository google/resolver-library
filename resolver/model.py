# Copyright 2013 Google Inc. All Rights Reserved.
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

"""A base class for statistical models for contributor behavior."""


import itertools
import random

import numpy


def ExtractResolutions(data):
  """Extracts a questions-to-resolutions dict from a ResolverData object."""
  return dict((question, resolution_map.copy())
              for question, (_, resolution_map) in data.iteritems())


class StatisticalModel(object):
  """A base class to model contributor behavior for discrete answer spaces."""

  @staticmethod
  def ExtractResolutions(data):
    """Made available here so users of subclasses don't have to import model."""
    return ExtractResolutions(data)

  @staticmethod
  def InitializeResolutions(data,
                            overwrite_all_resolutions=False,
                            judgments_to_answers=None):
    """Stores initial resolution guesses in a ResolverData object.

    The initial resolution guesses are simply weighted maps in which each
    judgment counts as one vote.  A question with one judgment of 'A', two of
    'B', one of 'C', and one of 'D'  would have its resolution initialized to:
      {'A': 0.2, 'B': 0.4, 'C': 0.2, 'D': 0.2}
    This assumes that the judgment and answer spaces are equal.  If this is not
    the case, use the judgments_to_answers parameter to specify the
    relationship between the two spaces.  For example,
      judgments_to_answers = {'A': 1, 'B': 1, 'C': None, 'D': 2}
    maps judgments 'A' and 'B' to answer 1 and judgment 'D' to answer 2,
    ignoring judgment 'C'.  In this case the above question would have its
    resolution initialized to:
      {1: 0.75, 2: 0.25}
    This feature can be paired with the ConfusionMatrices model to produce
    non-square confusion matrices.  The above example would result in confusion
    matrices with two rows and four columns.

    Args:
      data:  A ResolverData object.
      overwrite_all_resolutions:  A flag to overwrite any resolutions in data.
        - If False (the default), this constructor will set initial guesses for
          questions without resolutions in the data set.  Each answer in the
          initial guess for a question is given probability proportional to the
          number of judgments cast for that answer.  This provides a good
          starting point for resolution algorithms.
        - If True, this constructor will set these initial guesses for every
          question in the data set, even those that already have entries for
          their resolutions.
      judgments_to_answers:  An optional dict mapping from the judgment space
                             to the answer space.
        - Any judgment appearing as a key in the map will be treated as the
          corresponding answer for the purpose of computing initial resolution
          guesses.
        - Mapping a judgment to None will cause it to be ignored in this
          computation.
        - This does not directly restrict the behavior of models or resolution
          algorithms; it affects only their starting point.
    """

    if judgments_to_answers is None:
      judgments_to_answers = {}

    for responses, resolution_map in data.itervalues():
      # For questions which don't already have resolutions (golden data or
      # guesses) in the data object, we'll approximate initial resolutions
      # using voting.  This is to provide a starting point for algorithms such
      # as expectation-maximization, and also to ensure that a resolution
      # algorithm doesn't prematurely zero out any priors or parameters.
      if overwrite_all_resolutions or not resolution_map:
        resolution_map.clear()
        norm = 0.0
        for _, judgment, _ in responses:
          # If judgment is in judgments_to_answers, answer is the corresponding
          # value; otherwise, answer is identical to judgment:
          answer = judgments_to_answers.get(judgment, judgment)
          if answer is not None:
            norm += 1.0
            resolution_map[answer] = resolution_map.get(answer, 0.0) + 1.0
        if norm:
          for answer in resolution_map:
            resolution_map[answer] /= norm

  @staticmethod
  def RandomAnswer(resolution_map):
    """Returns an answer drawn randomly from resolution_map.

    Args:
      resolution_map:  A map from answers to probabilities.

    Returns:
      A key from resolution_map.
    """
    answers = resolution_map.keys()
    answer_probabilities = resolution_map.values() + [0.0]
    answer_index = next(itertools.compress(
        range(len(answers) + 1),
        numpy.random.multinomial(1, answer_probabilities)))
    if answer_index < len(answers):
      return answers[answer_index]
    return None

  @staticmethod
  def MostLikelyResolution(resolution_map):
    """Returns the most probable answer from a resolution map.

    Args:
      resolution_map:  A map from answers to probabilities.

    Returns:
      The answer with the highest probability.

    Example:
      r = {0: 0.4, 1: 0.5, 2: 0.1}
      MostLikelyResolution(r) == 1
    """
    if resolution_map:
      # Sort (for deterministic tests) and then shuffle, to break ties randomly:
      resolution_items = sorted(resolution_map.items())
      random.shuffle(resolution_items)
      return max(resolution_items, key=lambda x: x[1])[0]
    return None

  @staticmethod
  def MostLikelyResolutions(data):
    """Returns the most probable answer for each question.

    Args:
      data: A ResolverData object.

    Returns:
      A dict mapping from question to the most probable answer to each question.

    Example:
      t = {"Question 1": {0: 0.3, 1: 0.4, "foo": 0.3},
           "Question 2": {0: 0.9, 1: 0.1},
           "Question 3": {"wingnut": 0.8, "radguy": 0.2}}
      MostLikelyResolutions(t) == {"Question 1": 1,
                                   "Question 2": 0,
                                   "Question 3": "wingnut"]
    """
    return dict([(question,
                  StatisticalModel.MostLikelyResolution(resolution_map))
                 for question, (_, resolution_map) in data.iteritems()])

  @staticmethod
  def ResolutionDistanceSquared(x, y):
    """Returns the squared distance between two resolution maps.

    This is useful for convergence testing.

    Args:
      x:  a map from answers to probabilities.
      y:  a map from answers to probabilities.

    Returns:
      The sum of squared differences between x and y, with missing answers
      in either map being treated as having value 0.0.
    """
    return sum([(x.get(key, 0.0) - y.get(key, 0.0)) ** 2
                for key in set(x.keys() + y.keys())])

  def SetMLEParameters(self, data):
    raise NotImplementedError

  def SetSampleParameters(self, data):
    raise NotImplementedError

  def QuestionEntropy(self, data):
    raise NotImplementedError

  def PointwiseMutualInformation(self, data):
    raise NotImplementedError

  def MutualInformation(self, data):
    raise NotImplementedError

  def ResolveQuestion(self, question):
    raise NotImplementedError
