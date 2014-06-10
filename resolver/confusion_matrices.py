# Copyright 2012 Google Inc. All Rights Reserved.
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

"""Confusion matrices for judgment resolution.

This model gives each contributor a confusion matrix which gives the
probability that the contributor will yield a given judgment in response
to a question with a given answer.
"""

__author__ = 'tpw@google.com (Tamsyn Waterhouse)'

import collections
import math

import numpy

from resolver import digamma
from resolver import model


####################
# Module functions #
####################


def MLEResolutionPriors(data, question_weights=None):
  """Given resolutions, returns the MLE of the resolution priors.

  Args:
    data: A ResolverData object.
    question_weights: Used by decision_tree.py.  An optional dict from question
                      to a float.  Questions will be weighted accordingly in
                      computing the priors, with a default weight of 1.0.

  Returns:
    Maximum-likelihood resolution priors keyed by answer.
  """
  if question_weights is None:
    question_weights = {}

  # The MLE for the priors is simply the normalized resolution counts;
  # this is equation (2.4) of Dawid and Skene or equation (TODO(tpw)) in
  # RCJ:
  norm = 0.0
  priors = collections.defaultdict(float)
  for question, (_, resolution_map) in data.iteritems():
    weight = question_weights.get(question, 1.0)
    for answer in resolution_map:
      probability = resolution_map[answer] * weight
      priors[answer] += probability
      norm += probability
  assert norm or not priors  # Either norm is nonzero or priors is empty
  for answer in priors:
    priors[answer] /= norm
  return priors


def MLEConfusionMatrices(data, question_weights=None):
  """Given resolutions and judgments, returns the MLE confusion matrices.

  Args:
    data: A ResolverData object.
    question_weights: Used by decision_tree.py.  An optional dict from question
                      to a float.  Questions will be weighted accordingly in
                      computing the priors, with a default weight of 1.0.

  Returns:
    Maximum-likelihood confusion matrices keyed by contributor, answer,
    and judgment.
  """
  if question_weights is None:
    question_weights = {}

  # The MLE for a confusion matrix is computed by counting judgments; this is
  # equation (2.3) in Dawid and Skene or equation (TODO(tpw)) in RCJ:
  confusion_matrices = collections.defaultdict(
      lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
  # We'll need to normalize each row of each confusion matrix:
  norm = collections.defaultdict(lambda: collections.defaultdict(float))
  # Start by computing the sums of correct judgments:
  for question, (responses, resolution_map) in data.iteritems():
    weight = question_weights.get(question, 1.0)
    for contributor, judgment, _ in responses:
      for answer in resolution_map:
        probability = resolution_map[answer] * weight
        if probability:
          confusion_matrices[contributor][answer][judgment] += probability
          norm[contributor][answer] += probability
  # Now normalize each row of each matrix:
  for contributor in confusion_matrices:
    for answer in confusion_matrices[contributor]:
      assert norm[contributor][answer]
      for judgment in confusion_matrices[contributor][answer]:
        confusion_matrices[contributor][answer][judgment] /= (
            norm[contributor][answer])
  return confusion_matrices


def ResolutionPriorsDirichletParameters(data, question_weights=None):
  """Given resolutions, returns the Dirichlet parameters for the answer priors.

  Args:
    data: A ResolverData object.
    question_weights: Used by decision_tree.py.  An optional dict from question
                      to a float.  Questions will be weighted accordingly in
                      computing the priors, with a default weight of 1.0.

  Returns:
    The Dirichlet parameter vector describing P(priors|resolutions),
    keyed by answer.  (This can be used by the functions below.)
  """
  if question_weights is None:
    question_weights = {}

  # Compute the Dirichlet parameter vector describing P(priors|resolutions):
  dirichlet = collections.defaultdict(lambda: 1.0)
  for question, (_, resolution_map) in data.iteritems():
    weight = question_weights.get(question, 1.0)
    for answer in resolution_map:
      dirichlet[answer] += resolution_map[answer] * weight
  # dirichlet[answer] is now 1.0 plus the sum of all
  # resolution_map[answer] in data.
  return dirichlet


def VariationalResolutionPriors(dirichlet):
  """Given a Dirichlet parameter vector, returns variational prior parameters.

  Args:
    dirichlet: A dict mapping from answers to their corresponding Dirichlet
               parameters.

  Returns:
    Variational mean answer priors p, keyed by answer.
    This is the exponential of equation 18 in Simpson et al. (2011),
    http://www.orchid.ac.uk/eprints/7/, except that we have dropped the
    unnecessary constant e^psi(alpha_0) term.
    The short story is that these values for the resolution priors are weighted
    means that minimize the difference between our estimated resolutions and
    the full resolution inference integral.
  """
  # We seek the log-mean of the corresponding Dirichlet distribution:
  priors = {}
  for answer in dirichlet:
    priors[answer] = math.exp(digamma.Digamma(dirichlet[answer]))
  return priors


def SampleResolutionPriors(dirichlet):
  """Given a Dirichlet parameter vector, returns sample answer priors.

  Args:
    dirichlet: A dict mapping from answers to their corresponding Dirichlet
               parameters.

  Returns:
    Sample answer priors p drawn from the distribution described by dirichlet,
    keyed by answer.
  """

  # Draw and return a random prior vector using the Dirichlet distribution:
  return dict(zip(dirichlet.iterkeys(),
                  numpy.random.dirichlet(dirichlet.values())))


def ConfusionMatricesDirichletParameters(data, question_weights=None):
  """Given resolutions and judgments, returns the Dirichlet parameters for CMs.

  Args:
    data: A ResolverData object.
    question_weights: Used by decision_tree.py.  An optional dict from question
                      to a float.  Questions will be weighted accordingly in
                      computing the priors, with a default weight of 1.0.

  Returns:
    The Dirichlet parameters describing P(confusion_matrices|data), keyed by
    answer, judgment, and contributor.  (This can be used by the functions
    below.)
  """
  if question_weights is None:
    question_weights = {}

  # First we compute the Dirichlet parameter vectors, one for each
  # contributor-answer pair, that describe P(confusion_matrices|data):
  dirichlet = collections.defaultdict(
      lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
  # As we go, we'll keep track of all the contributors, answers, and judgments
  # we've seen.  We'll use this below to fill out the dirichlet matrix.
  contributor_space = set()
  answer_space = set()
  judgment_space = set()
  for question, (responses, resolution_map) in data.iteritems():
    weight = question_weights.get(question, 1.0)
    for contributor, judgment, _ in responses:
      judgment_space.add(judgment)
      for answer in resolution_map:
        dirichlet[contributor][answer][judgment] += (resolution_map[answer] *
                                                     weight)
        contributor_space.add(contributor)
        answer_space.add(answer)
  # We must ensure that dirichlet has an entry (and a +1 term) for each possible
  # contributor-answer-judgment triplet, in order for our confusion matrices to
  # have the possibility of a nonzero entry in each cell.  (Otherwise, these
  # would be "confusion maps" rather than confusion matrices.)
  for contributor in contributor_space:
    for answer in answer_space:
      for judgment in judgment_space:
        dirichlet[contributor][answer][judgment] += 1.0
  # Now dirichlet[contributor][answer][judgment] is 1.0 plus the sum of
  # resolution_map[answer] (the probability that answer is the correct answer
  # to a question) for each time contributor gave judgment as a response, over
  # all questions.
  # Each map dirichlet[contributor][answer] is thus a Dirichlet parameter vector
  # expressing P(confusion_matrices|data) for one row of one contributor's
  # confusion matrix.
  return dirichlet


def VariationalConfusionMatrices(dirichlet):
  """Given Dirichlet parameters, returns variational CM parameters.

  Args:
    dirichlet: A dict-of-dicts-of-dicts mapping from contributors, answers, and
               judgments to their corresponding Dirichlet parameters.

  Returns:
    Variational mean confusion matrices, keyed by answer, judgment, and
    contributor.
    This is the exponential of equation 23 in Simpson et al. (2011),
    http://www.orchid.ac.uk/eprints/7/
    The short story is that these values for the confusion matrices are
    weighted means that minimize the difference between our estimated
    resolutions and the full resolution inference integral.
  """

  # We seek the log-mean of the corresponding Dirichlet distribution for each
  # row of each confusion matrix:
  confusion_matrices = collections.defaultdict(
      lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
  for contributor in dirichlet:
    for answer in dirichlet[contributor]:
      # alpha_0 is the sum of the Dirichlet parameters,
      alpha_0 = sum(dirichlet[contributor][answer].itervalues())
      # and digamma_alpha_0 is the digamma function applied to it:
      digamma_alpha_0 = digamma.Digamma(alpha_0)
      for judgment in dirichlet[contributor][answer]:
        alpha = dirichlet[contributor][answer][judgment]
        assert alpha
        confusion_matrices[contributor][answer][judgment] = math.exp(
            digamma.Digamma(alpha) - digamma_alpha_0)
  return confusion_matrices


def SampleConfusionMatrices(dirichlet):
  """Given Dirichlet parameters, returns sample confusion matrices.

  Args:
    dirichlet: A dict-of-dicts-of-dicts mapping from contributors, answers, and
               judgments to their corresponding Dirichlet parameters.

  Returns:
    A set of sample confusion matrices drawn from the distribution described by
    dirichlet, keyed by answer, judgment, and contributor.
  """

  # Draw random confusion matrix rows from the Dirichlet distribution:
  confusion_matrices = collections.defaultdict(
      lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
  for contributor in dirichlet:
    for answer in dirichlet[contributor]:
      confusion_matrices[contributor][answer] = dict(zip(
          dirichlet[contributor][answer].iterkeys(),
          numpy.random.dirichlet(dirichlet[contributor][answer].values())))
  return confusion_matrices


###########################
# ConfusionMatrices class #
###########################


class ConfusionMatrices(model.StatisticalModel):
  """Models rater behaviour using one confusion matrix for each contributor."""

  def SetMLEParameters(self, data, question_weights=None):
    """Given resolutions and judgments, updates self with their MLEs.

    Args:
      data: A ResolverData object.
      question_weights: Used by decision_tree.py.  An optional dict from
                        question to a float.  Questions will be weighted
                        accordingly in computing the priors, with a default
                        weight of 1.0.
    """
    self.priors = MLEResolutionPriors(data, question_weights=question_weights)
    self.confusion_matrices = MLEConfusionMatrices(
        data, question_weights=question_weights)

  def SetVariationalParameters(self, data, question_weights=None):
    """Given resolutions and judgments, updates self with variational params.

    Args:
      data: A ResolverData object.
      question_weights: Used by decision_tree.py.  An optional dict from
                        question to a float.  Questions will be weighted
                        accordingly in computing the priors, with a default
                        weight of 1.0.
    """
    self.priors = VariationalResolutionPriors(
        ResolutionPriorsDirichletParameters(data,
                                            question_weights=question_weights))
    self.confusion_matrices = VariationalConfusionMatrices(
        ConfusionMatricesDirichletParameters(data,
                                             question_weights=question_weights))

  def SetSampleParameters(self, data, question_weights=None):
    """Given resolutions and judgments, updates self with sampled parameters.

    Args:
      data: A ResolverData object.
      question_weights: Used by decision_tree.py.  An optional dict from
                        question to a float.  Questions will be weighted
                        accordingly in computing the priors, with a default
                        weight of 1.0.
    """
    self.priors = SampleResolutionPriors(
        ResolutionPriorsDirichletParameters(data,
                                            question_weights=question_weights))
    self.confusion_matrices = SampleConfusionMatrices(
        ConfusionMatricesDirichletParameters(data,
                                             question_weights=question_weights))

  def QuestionEntropy(self, radix=2):
    """Calculates and returns per-question entropy, given self.priors.

    Args:
      radix:  The units in which to measure information (2 for bits, etc.).

    Returns:
      Entropy in units of base equal to the parameter radix (bits by default).
    """
    # self.priors is not guaranteed to be normalized, so normalize its values:
    norm = sum(self.priors.itervalues())
    normalized_probabilities = [probability / norm
                                for probability in self.priors.itervalues()]
    return sum(-probability * math.log(probability, radix)
               if probability else 0.0
               for probability in normalized_probabilities)

  def PointwiseMutualInformation(self, contributor, answer, judgment,
                                 previous_responses=None, radix=2):
    """Returns the pointwise (conditional) mutual information for one judgment.

    This is the amount by which the information content (surprise) of learning
    the correct answer is decreased by having this judgment.

    Args:
      contributor:  The contributor providing the new judgment.
      answer:  The correct answer to the question.
      judgment:  The judgment given by the contributor.
      previous_responses:  An optional iterable of previous (contributor,
                           judgment, metadata) tuples for this question.  If
                           provided, the pointwise mutual information is
                           computed conditioned on this set of previous
                           judgments.
      radix:  The units in which to measure information (2 for bits, etc.).

    Returns:
      The pointwise mutual information
      log(p(answer|judgment,previous_responses) / p(answer|previous_responses))
      of the contributor giving judgment <judgment> when the correct answer is
      <answer>.
    """
    if previous_responses is None:
      previous_responses = []

    # Compute the answer probabilities before the contributor gave the judgment:
    resolution_before = self.ResolveQuestion(previous_responses)
    probability_before = resolution_before.get(answer, 0.0)

    # And after:
    resolution_after = self.ResolveQuestion(previous_responses +
                                            [(contributor, judgment, {})])
    probability_after = resolution_after.get(answer, 0.0)

    # Compute the reduction in information resulting from this judgment:
    # Note:  If probability_before is zero, then probability_after must also be
    #        zero.  And if probability_after is zero, then we've concluded that
    #        answer cannot be correct.  In theory this means we should return
    #        negative infinity, but we'll just return zero instead in such
    #        cases.
    if probability_before and probability_after:
      return math.log(probability_after / probability_before, radix)
    # else:
    return 0.0

  def MutualInformation(self, contributor, previous_responses=None, radix=2):
    """Returns the (conditional) mutual information for one contributor.

    This is the expected amount by which the information content of learning
    the correct answer is decreased by obtaining a judgment from a given
    contributor, optionally conditioned on a set of previous judgments.

    Args:
      contributor:  The contributor.
      previous_responses:  An optional iterable of previous (contributor,
                           judgment, metadata) tuples for this question.  If
                           provided, the mutual information is computed
                           conditioned on this set of previous judgments.
      radix:  The units in which to measure information (2 for bits, etc.).

    Returns:
      Mutual information in units of base equal to the parameter radix
      (bits by default).
    """
    if previous_responses is None:
      previous_responses = []

    resolution_via_previous_responses = self.ResolveQuestion(previous_responses)

    if contributor in self.confusion_matrices:
      return sum([resolution_via_previous_responses.get(answer, 0.0) *
                  self.confusion_matrices[contributor][answer][judgment] *
                  self.PointwiseMutualInformation(
                      contributor, answer, judgment,
                      previous_responses=previous_responses, radix=radix)
                  for answer in self.confusion_matrices[contributor]
                  for judgment in self.confusion_matrices[contributor][answer]])
    # else:
    return 0.0

  def ResolveQuestion(self, responses):
    """Returns one resolution map based on judgments and current parameters.

    Args:
      responses:  A list of (contributor, judgment, metadata) tuples for this
                  question.

    Returns:
      A map from answers to probabilities.
    """
    # Use Bayes to calculate the resolution probabilities.
    # unnormalized_posterior is P in equation (TODO(tpw)) in RCJ, and the
    # resolution probabilities are equation (TODO(tpw)) in RCJ:

    # Start with the priors:
    unnormalized_posterior = self.priors.copy()
    for contributor, judgment, _ in responses:
      if contributor in self.confusion_matrices:
        for answer in self.confusion_matrices[contributor]:
          # For each judgment we have for this question, the posterior gets a
          # factor proportional to the confusion matrix entry for that
          # contributor-answer-judgment triplet:
          if answer in unnormalized_posterior:
            if judgment in self.confusion_matrices[contributor][answer]:
              unnormalized_posterior[answer] *= (
                  self.confusion_matrices[contributor][answer][judgment])
            else:
              del unnormalized_posterior[answer]
    resolution_map = {}
    norm = sum(unnormalized_posterior.itervalues())
    if norm:
      for answer in unnormalized_posterior:
        resolution_map[answer] = unnormalized_posterior[answer] / norm
    return resolution_map
