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

"""The one-parameter-per-contributor model for judgment resolution.

This model gives each contributor a single number pi that is that contributor's
probability of responding to any question with a judgment equal to the correct
answer.  (The model therefore assumes that the answer space is a subset of the
judgment space.  This particular implementation further assumes the two spaces
are identical, building both from the set of observed judgments.)
"""

__author__ = 'tpw@google.com (Tamsyn Waterhouse)'

import collections
import math

import numpy

from resolver import digamma
from resolver import model


INFINITY = float('inf')


####################
# Module functions #
####################


def MLEProbabilityCorrect(data, question_weights=None):
  """Given resolutions and judgments, returns MLE contributor parameters.

  Args:
    data: a ResolverData object.
    question_weights: Used by decision_tree.py.  An optional dict from question
                      to a float.  Questions will be weighted accordingly in
                      computing the parameters, with a default weight of 1.0.

  Returns:
    Maximum-likelihood probabilities-correct keyed by contributor.
  """
  if question_weights is None:
    question_weights = {}

  # The MLE for a contributor's probability-correct parameter is simply the
  # proportion of that contributor's judgments that are correct; this is
  # equation (TODO(tpw)) in RCJ:
  total_judgments = collections.defaultdict(float)
  correct_judgments = collections.defaultdict(float)
  for question, (responses, resolution_map) in data.iteritems():
    weight = question_weights.get(question, 1.0)
    for contributor, judgment, _ in responses:
      total_judgments[contributor] += weight
      if judgment in resolution_map:
        # Add to correct_judgments the probability that this judgment was
        # correct (that is, correct_judgments is really an expectation):
        correct_judgments[contributor] += resolution_map[judgment] * weight
  probability_correct = {}
  for contributor in total_judgments:
    assert total_judgments[contributor]
    probability_correct[contributor] = (
        correct_judgments[contributor] / total_judgments[contributor])
  return probability_correct


def ProbabilityCorrectBetaParameters(data, question_weights=None):
  """Given resolutions and judgments, returns beta parameters for P(pi|data).

  Args:
    data: a ResolverData object.
    question_weights: Used by decision_tree.py.  An optional dict from question
                      to a float.  Questions will be weighted accordingly in
                      computing the parameters, with a default weight of 1.0.

  Returns:
    A map to alpha and a map to beta parameters, each keyed by contributor,
    that describe the likelihood function for each contributor's
    probability-correct value.
  """
  if question_weights is None:
    question_weights = {}

  # Compute the pairs of beta parameters that together describe P(pi|data):
  # alpha is 1.0 plus the number of correct judgments from a given contributor:
  alpha = collections.defaultdict(lambda: 1.0)
  # beta is 1.0 plus the number of correct judgments from a given contributor:
  beta = collections.defaultdict(lambda: 1.0)
  for question, (responses, resolution_map) in data.iteritems():
    weight = question_weights.get(question, 1.0)
    for contributor, judgment, _ in responses:
      probability_correct = resolution_map.get(judgment, 0.0)
      alpha[contributor] += probability_correct * weight
      beta[contributor] += (1.0 - probability_correct) * weight
  return alpha, beta


def SampleProbabilityCorrect((alpha, beta)):
  """Given alpha and beta parameters, returns sample parameters from P(pi|data).

  Args:
    (alpha, beta):  Maps from contributor to alpha and to beta, respectively.

  Returns:
    Sample probabilities-correct drawn from the beta distribution P(pi|data),
    keyed by contributor.
  """
  # Draw a random probability-correct parameter for each contributor from the
  # beta distribution:
  probability_correct = {}
  assert alpha.keys() == beta.keys()  # each contributor should appear in both
  for contributor in alpha:
    probability_correct[contributor] = numpy.random.beta(alpha[contributor],
                                                         beta[contributor])
  return probability_correct


def VariationalProbabilityCorrect((alpha, beta)):
  """Given alpha and beta parameters, returns variational parameters.

  Args:
    (alpha, beta):  Maps from contributor to alpha and to beta, respectively.

  Returns:
    Probability-correct values corresponding to geometric-mean likelihood
    conditional probabilities, keyed by contributor.  See RCJ for details.
  """
  probability_correct = {}
  assert alpha.keys() == beta.keys()  # each contributor should appear in both
  for contributor in alpha:
    exp_psi_alpha = math.exp(digamma.Digamma(alpha[contributor]))
    exp_psi_beta = math.exp(digamma.Digamma(beta[contributor]))
    probability_correct[contributor] = (
        exp_psi_alpha / (exp_psi_alpha + exp_psi_beta))
  return probability_correct


def AnswerSpaceSize(data):
  """Given judgments, returns the number of unique judgments seen.

  Args:
    data: a ResolverData object.

  Returns:
    The observed size of the judgment (and thus answer) space; that is, the
    number of unique judgments seen amongst all the responses.
  """
  # Calculate the answer space size using the judgments given:
  answer_space = set()
  for responses, _ in data.itervalues():
    for _, judgment, _ in responses:
      answer_space.add(judgment)
  return len(answer_space)


############################
# ProbabilityCorrect class #
############################


class ProbabilityCorrect(model.StatisticalModel):
  """Models rater behaviour using one parameter for each contributor."""

  def __init__(self):
    self.probability_correct = {}
    self._answer_space_size = 0

  def SetMLEParameters(self, data, question_weights=None):
    """Given resolutions and judgments, updates self with MLE parameter values.

    Args:
      data: a ResolverData object.
      question_weights: Used by decision_tree.py.  An optional dict from
                        question to a float.  Questions will be weighted
                        accordingly in computing the parameters, with a
                        default weight of 1.0.
    """
    self.probability_correct = MLEProbabilityCorrect(
        data, question_weights=question_weights)
    self._UpdatePrivateMembers(data)

  def SetSampleParameters(self, data, question_weights=None):
    """Given resolutions and judgments, updates self with sampled parameters.

    Args:
      data: a ResolverData object.
      question_weights: Used by decision_tree.py.  An optional dict from
                        question to a float.  Questions will be weighted
                        accordingly in computing the parameters, with a
                        default weight of 1.0.
    """
    self.probability_correct = SampleProbabilityCorrect(
        ProbabilityCorrectBetaParameters(data,
                                         question_weights=question_weights))
    self._UpdatePrivateMembers(data)

  def SetVariationalParameters(self, data, question_weights=None):
    """Given resolutions and judgments, updates self with variational params.

    Args:
      data: a ResolverData object.
      question_weights: Used by decision_tree.py.  An optional dict from
                        question to a float.  Questions will be weighted
                        accordingly in computing the parameters, with a
                        default weight of 1.0.
    """
    self.probability_correct = VariationalProbabilityCorrect(
        ProbabilityCorrectBetaParameters(data,
                                         question_weights=question_weights))
    self._UpdatePrivateMembers(data)

  def SetAnswerSpaceSize(self, answer_space_size):
    self._answer_space_size = answer_space_size

  def UnsetAnswerSpaceSize(self):
    self._answer_space_size = 0

  def _UpdatePrivateMembers(self, data):
    """Updates this object's members used by ResolveQuestion.

    This method should be called (typically by a statistical-inference class)
    AFTER setting parameters (for example via this class's SetMLEParameters or
    SetSampleParameters methods) and BEFORE calling ResolveQuestion.

    What it does:
      - Sets self._answer_space_size if it's not already set.
      - Calculates the helper variable self._probability_factor for each
        contributor.

    Args:
      data:  a ResolverData object.
    """
    if not self._answer_space_size:
      # Calculate the answer space size using the judgments given:
      self.SetAnswerSpaceSize(AnswerSpaceSize(data))

    # Next we'll precompute a helper value probability_factor for each
    # contributor.  Here's why and how:
    # In equation (TODO(tpw)) in RCJ, for each possible answer, we have a
    # factor "pi" for each judgment.  For probability-correct type models:
    #   - If the judgment is equal to the putative answer, the factor is equal
    #     to the probability-correct value.  Call this the "correct" factor.
    #   - If the judgment is not equal to the putative answer, the factor is
    #     equal to (1.0 minus the probability-correct value) / (s minus 1.0),
    #     where s is the size of the answer space.  Call this the "non-correct"
    #     factor.
    # For each judgment, we multiply every answer's probability by the
    # appropriate pi.  After all this, we normalize the distribution of
    # probabilities to obtain the final resolution for the question.
    # But there's a problem:
    #   We don't want to multiply every element of the answer space by pi for
    #   each judgment, because this is slow and because the answer space might
    #   be huge, and we don't know all the elements of the answer space anyway,
    #   only the ones that we see judgments for!
    # However:
    #   If, instead of multiplying every answer not matching the judgment by
    #   the non-correct factor, we divide the matching answer by the
    #   non-correct factor, this has the effect of multiplying the whole
    #   distribution by one non-correct factor for each judgment provided.
    # For example, let c_h be the "correct" factor and i_h be the
    # "non-correct" factor for contributor h.  If contributors 1 and 2 give
    # judgment A and contributor 3 gives judgment B, then the unnormalized
    # posterior for the question is
    #   P(A) = c_1 c_2 i_3,
    #   P(B) = i_1 i_2 c_3,
    #   P(*) = i_1 i_2 i_3 for each other answer *.
    # Using the alternative arithmetic above, we instead calculate
    #   P(A) = (c_1 / i_1) (c_2 / i_2),
    #   P(B) = (c_3 / i_3),
    #   P(*) = 1,
    # which is simply the values above divided by i_1 i_2 i_3, so that the
    # normalized posterior is unchanged.  We'll call c_h / i_h the "probability
    # factor" for contributor h.
    # However:
    #   If any contributor has probability-correct equal to 1.0, in other words
    #   considered by the model parameters to be perfect, then i_h for that
    #   contributor is 0.0, so this scheme doesn't work because it involves
    #   division by zero.
    # So:
    #   In these cases, we'll just set probability_factor to infinity (as a
    #   placeholder) and special-case the contributor in the ResolveQuestion
    #   method.

    # Calculate the probability factor c_h / i_h for each contributor h:
    self._probability_factor = {}
    for contributor in self.probability_correct:
      correct_factor = self.probability_correct[contributor]
      if correct_factor < 1.0 and self._answer_space_size > 1:
        s = float(self._answer_space_size)
        incorrect_factor = (1.0 - correct_factor) / (s - 1.0)
        self._probability_factor[contributor] = (correct_factor /
                                                 incorrect_factor)
      else:
        # Store infinity, although this value will not be used for calculations:
        self._probability_factor[contributor] = INFINITY

  def QuestionEntropy(self, radix=2):
    """Calculates and returns per-question entropy, given self.priors.

    Args:
      radix:  The units in which to measure information (2 for bits, etc.).

    Returns:
      Entropy in units of base equal to the parameter radix (bits by default).
    """
    # Since we assume a uniform prior over the answer space, the entropy of a
    # question is just the logarithm of the answer space size.
    assert self._answer_space_size > 0
    return math.log(self._answer_space_size, radix)

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
      <answer>.  Note that if previous_responses is empty or None, this
      expression is equal to log(p(answer|judgment) * self._answer_space_size).
    """
    if previous_responses is None:
      previous_responses = []

    assert self._answer_space_size > 1
    s = float(self._answer_space_size)

    # Compute the answer probabilities before the contributor gave the judgment:
    resolution_before = self.ResolveQuestion(previous_responses)
    if answer in resolution_before:
      # Read the probability of answer directly out of the resolution:
      probability_before = resolution_before[answer]
    else:
      # answer is some element of the answer space that is not in
      # resolution_before, so it divides the residual probability
      # with any other such members of the answer space:
      assert s > len(resolution_before)
      probability_before = ((1.0 - sum(resolution_before.values())) /
                            (s - len(resolution_before)))

    # And after:
    resolution_after = self.ResolveQuestion(previous_responses +
                                            [(contributor, judgment, {})])
    # TODO(tpw):  As elsewhere in this class, we're ignoring the metadata part
    #             of the response for now, which in Matchmaker 2 tasks can
    #             include part of the judgment.  We must fix this, probably by
    #             adding a new model class for MM2 tasks.

    if answer in resolution_after:
      # Read the probability of answer directly out of the resolution:
      probability_after = resolution_after[answer]
    else:
      # answer is not in resolution_after, so it divides the residual
      # probability with any other such members of the answer space:
      assert s > len(resolution_after)  # else answer isn't in the answer space
      probability_after = ((1.0 - sum(resolution_after.values())) /
                           (s - len(resolution_after)))

    # Compute the reduction in information resulting from this judgment:
    if probability_before and probability_after:
      return math.log(probability_after / probability_before, radix)
    # else answer is impossible, which means in theory that we should return
    # negative infinity, but we'll simply return zero instead:
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

    assert self._answer_space_size > 1
    s = float(self._answer_space_size)

    resolution_via_previous_responses = self.ResolveQuestion(previous_responses)

    judgments_seen = set([judgment for _, judgment, _ in previous_responses])

    # Create two unique answers not seen in previous_responses, by incrementing
    # from zero until we have a unique value; we'll use these below as "none of
    # the above" answers and judgments:
    unseen_one = 0
    while unseen_one in judgments_seen:
      unseen_one += 1
    unseen_two = unseen_one + 1
    while unseen_two in judgments_seen:
      unseen_two += 1

    # Now comes the calculation.  What we're doing is summing up terms
    #   P(a) * P(j|a) * log(P(a|j,J) / P(a|J)),
    # where a is the answer, j is the judgment (1) given by worker2, and J is
    # previous_responses, a set of judgments already recorded for this question.
    # This sum is the expected value of the logarithm (which is the change in
    # information) over all possible values of a and j, weighted by their
    # joint probability given previous_responses.
    # Since this model allows the answer space to be arbitrarily large and for
    # us not to know all the answers by name, we must break up the sum into
    # five cases:
    #   1) a and j are both in previous_responses
    #   2) a is in previous_responses, but j is not
    #   3) a is not in previous_responses, but j is
    #   4) a and j are not in previous_responses, and a == j
    #   5) a and j are not in previous_responses, and a != j
    mutual_information = 0.0
    p = self.probability_correct[contributor]
    n_unseen_judgments = s - len(judgments_seen)
    # First add up the cases where the answer and the judgment are both in
    # previous_responses:
    for answer in judgments_seen:
      answer_probability = resolution_via_previous_responses[answer]
      for judgment in judgments_seen:
        p_judgment = p if judgment == answer else (1.0 - p) / (s - 1.0)
        mutual_information += (answer_probability * p_judgment *
                               self.PointwiseMutualInformation(
                                   contributor, answer, judgment,
                                   previous_responses=previous_responses,
                                   radix=radix))
      # And now the case where the answer is in the previous responses but the
      # judgment isn't:
      judgment = unseen_two
      p_judgment = (1.0 - p) / (s - 1.0)  # because judgment != answer
      mutual_information += (n_unseen_judgments *
                             answer_probability * p_judgment *
                             self.PointwiseMutualInformation(
                                 contributor, answer, judgment,
                                 previous_responses=previous_responses,
                                 radix=radix))
    if n_unseen_judgments:
      # Now the cases where the answer is not in the previous responses but the
      # judgment is:
      answer = unseen_one
      answer_probability = (
          (1.0 - sum(resolution_via_previous_responses.values())) /
          n_unseen_judgments)
      for judgment in judgments_seen:
        p_judgment = (1.0 - p) / (s - 1.0)  # because judgment != answer
        mutual_information += (n_unseen_judgments *
                               answer_probability * p_judgment *
                               self.PointwiseMutualInformation(
                                   contributor, answer, judgment,
                                   previous_responses=previous_responses,
                                   radix=radix))
      # Next, the case where the judgment and the answer are the same value
      # not in the previous responses:
      answer = unseen_one
      judgment = unseen_one
      p_judgment = p  # because judgment == answer
      mutual_information += (n_unseen_judgments *
                             answer_probability * p_judgment *
                             self.PointwiseMutualInformation(
                                 contributor, answer, judgment,
                                 previous_responses=previous_responses,
                                 radix=radix))
    if n_unseen_judgments >= 2:
      # Finally, the case where the judgment and the answer are distinct values
      # not in the previous responses:
      answer = unseen_one
      judgment = unseen_two
      p_judgment = (1.0 - p) / (s - 1.0)  # because judgment != answer
      mutual_information += (n_unseen_judgments * (n_unseen_judgments - 1.0) *
                             answer_probability * p_judgment *
                             self.PointwiseMutualInformation(
                                 contributor, answer, judgment,
                                 previous_responses=previous_responses,
                                 radix=radix))

    # Return the accumulated sum:
    return mutual_information

  def ResolveQuestion(self, responses):
    """Returns one resolution map based on judgments and current parameters.

    Note well:  This dict's values will not sum to 1.0 if the answer space is
                larger than the set of observed judgments for this question!
                Answers in the answer space not seen in the judgments will not
                be present in the returned dict.  For example, if the answer
                space is
                  ['A', 'B', 'C', 'D']
                (size 4) and the judgments are 'A', 'A', and 'C', the returned
                dict may look something like
                  {'A': 0.6, 'C': 0.2}
                with the unseen answers 'B' and 'D', each with probability 0.1,
                omitted.  The reason for this is that this class doesn't store
                the answer space itself, only its size; the class is meant to
                handle arbitrarily large answer spaces without needing to know
                or to enumerate all the possible answers.

    Args:
      responses:  A list of (contributor, judgment, metadata) tuples for this
                  question.

    Returns:
      A map from answers to probabilities.
    """
    # Each answer posterior is a product of factors, so we can use a
    # defaultdict that defaults to 1.0:
    unnormalized_posterior = collections.defaultdict(lambda: 1.0)
    perfect_posterior = {}
    for contributor, judgment, _ in responses:
      # Multiply the answer matching the judgment by the probability factor
      # described in _UpdatePrivateMembers() above.
      if self._probability_factor[contributor] == INFINITY:
        # This contributor is assumed perfect, so store its judgment in
        # perfect_posterior:
        perfect_posterior[judgment] = 1.0
      else:
        unnormalized_posterior[judgment] *= (
            self._probability_factor[contributor])
    if len(perfect_posterior) > 1:
      # Multiple assumed-perfect contributors gave conflicting judgments, so
      # the resolution is undefined; we'll return the empty resolution:
      return {}
    elif len(perfect_posterior) == 1:
      # One or more assumed-perfect contributors gave the same judgment, so it
      # must be the answer to this question:
      return perfect_posterior
    # else we normalize and return unnormalized_posterior:
    resolution_map = {}
    norm = (sum(unnormalized_posterior.itervalues()) +
            float(self._answer_space_size - len(unnormalized_posterior)))
    if norm:
      for answer in unnormalized_posterior:
        resolution_map[answer] = unnormalized_posterior[answer] / norm
    return resolution_map
