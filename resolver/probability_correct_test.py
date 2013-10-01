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

"""Tests for probability_correct.py."""

__author__ = 'tpw@google.com (Tamsyn Waterhouse)'

import math

import numpy
import scipy.special

import unittest
from resolver import probability_correct
from resolver import test_util


# Macros for functions used to compute test data:
EXP = math.exp
DIGAMMA = scipy.special.psi

# MLE estimates and beta distribution parameters from the Ipeirotis example:
IPEIROTIS_MLE_PROBABILITY_CORRECT = {'worker1': 0.4,
                                     'worker2': 0.8,
                                     'worker3': 1.0,
                                     'worker4': 1.0,
                                     'worker5': 0.0}
IPEIROTIS_ALPHA = {'worker1': 3.0,
                   'worker2': 5.0,
                   'worker3': 6.0,
                   'worker4': 6.0,
                   'worker5': 1.0}
IPEIROTIS_BETA = {'worker1': 4.0,
                  'worker2': 2.0,
                  'worker3': 1.0,
                  'worker4': 1.0,
                  'worker5': 6.0}
IPEIROTIS_VARIATIONAL_PROBABILITY_CORRECT = {
    'worker1': EXP(DIGAMMA(3.0)) / (EXP(DIGAMMA(3.0)) + EXP(DIGAMMA(4.0))),
    'worker2': EXP(DIGAMMA(5.0)) / (EXP(DIGAMMA(5.0)) + EXP(DIGAMMA(2.0))),
    'worker3': EXP(DIGAMMA(6.0)) / (EXP(DIGAMMA(6.0)) + EXP(DIGAMMA(1.0))),
    'worker4': EXP(DIGAMMA(6.0)) / (EXP(DIGAMMA(6.0)) + EXP(DIGAMMA(1.0))),
    'worker5': EXP(DIGAMMA(1.0)) / (EXP(DIGAMMA(1.0)) + EXP(DIGAMMA(6.0)))}

# and from the Dawid & Skene example (based on the resolutions obtained via EM):
DS_MLE_PROBABILITY_CORRECT = {1: 0.8288,
                              2: 0.7789,
                              3: 0.7766,
                              4: 0.8669,
                              5: 0.8424}
DS_ALPHA = {1: 112.883, 2: 36.049, 3: 35.949, 4: 40.013, 5: 38.907}
DS_BETA = {1: 24.117, 2: 10.951, 3: 11.051, 4: 6.987, 5: 8.093}
DS_VARIATIONAL_PROBABILITY_CORRECT = {
    1: EXP(DIGAMMA(112.883)) / (EXP(DIGAMMA(112.883)) + EXP(DIGAMMA(24.117))),
    2: EXP(DIGAMMA(36.049)) / (EXP(DIGAMMA(36.049)) + EXP(DIGAMMA(10.951))),
    3: EXP(DIGAMMA(35.949)) / (EXP(DIGAMMA(35.949)) + EXP(DIGAMMA(11.051))),
    4: EXP(DIGAMMA(40.013)) / (EXP(DIGAMMA(40.013)) + EXP(DIGAMMA(6.987))),
    5: EXP(DIGAMMA(38.907)) / (EXP(DIGAMMA(38.907)) + EXP(DIGAMMA(8.093)))}


class ProbabilityCorrectTest(unittest.TestCase):
  longMessage = True
  numpy.seterr(all='raise')

  def testMLEProbabilityCorrect(self):
    # First with the complete Ipeirotis example:
    result = probability_correct.MLEProbabilityCorrect(
        test_util.IPEIROTIS_DATA_FINAL)
    test_util.AssertMapsAlmostEqual(self,
                                    IPEIROTIS_MLE_PROBABILITY_CORRECT,
                                    result,
                                    label='contributor')

    # Now with the Dawid and Skene example (judgments plus resolutions):
    result = probability_correct.MLEProbabilityCorrect(
        test_util.DS_DATA_FINAL)
    test_util.AssertMapsAlmostEqual(self,
                                    DS_MLE_PROBABILITY_CORRECT,
                                    result,
                                    label='contributor')

  def testProbabilityCorrectBetaParameters(self):
    # Test for the Dawid & Skene data:
    result_alpha, result_beta = (
        probability_correct.ProbabilityCorrectBetaParameters(
            test_util.DS_DATA_FINAL))
    test_util.AssertMapsAlmostEqual(self, DS_ALPHA, result_alpha,
                                    label='alpha, contributor')
    test_util.AssertMapsAlmostEqual(self, DS_BETA, result_beta,
                                    label='beta, contributor')
    # Test for the Ipeirotis data:
    result_alpha, result_beta = (
        probability_correct.ProbabilityCorrectBetaParameters(
            test_util.IPEIROTIS_DATA_FINAL))
    test_util.AssertMapsAlmostEqual(self, IPEIROTIS_ALPHA, result_alpha,
                                    label='alpha, contributor')
    test_util.AssertMapsAlmostEqual(self, IPEIROTIS_BETA, result_beta,
                                    label='beta, contributor')

  def testSampleProbabilityCorrect(self):
    # Seed the random number generator to produce deterministic test results:
    numpy.random.seed(0)

    # Check that the mean of 50000 samples is close to the actual mean of the
    # beta distribution for a set of probability-correct parameters, for a case
    # with a narrow distribution (the Dawid and Skene example):
    samples = [
        probability_correct.SampleProbabilityCorrect((DS_ALPHA, DS_BETA))
        for _ in range(50000)]
    expected = {1: 112.883 / 137.0,
                2: 36.049 / 47.0,
                3: 35.949 / 47.0,
                4: 40.013 / 47.0,
                5: 38.907 / 47.0}
    mean = {}
    for contributor in expected:
      mean[contributor] = numpy.mean([sample[contributor]
                                      for sample in samples])
    test_util.AssertMapsAlmostEqual(self, expected, mean, label='contributor')

    # And again for a broad distribution (the Ipeirotis example):
    samples = [probability_correct.SampleProbabilityCorrect((IPEIROTIS_ALPHA,
                                                             IPEIROTIS_BETA))
               for _ in range(50000)]
    # In this case we'll check both the mean...
    expected = {}
    mean = {}
    for contributor in IPEIROTIS_ALPHA:
      expected[contributor] = IPEIROTIS_ALPHA[contributor] / (
          IPEIROTIS_ALPHA[contributor] + IPEIROTIS_BETA[contributor])
      mean[contributor] = numpy.mean([sample[contributor]
                                      for sample in samples])
    test_util.AssertMapsAlmostEqual(self, expected, mean,
                                    label='contributor', places=2)
    # ...and the variance:
    variance = {}
    expected = {}
    for contributor in IPEIROTIS_ALPHA:
      expected[contributor] = (
          IPEIROTIS_ALPHA[contributor] * IPEIROTIS_BETA[contributor] /
          ((IPEIROTIS_ALPHA[contributor] + IPEIROTIS_BETA[contributor]) ** 2 *
           (IPEIROTIS_ALPHA[contributor] + IPEIROTIS_BETA[contributor] + 1.0)))
      variance[contributor] = numpy.var([sample[contributor]
                                         for sample in samples])
    test_util.AssertMapsAlmostEqual(self, expected, variance,
                                    label='contributor', places=2)

  def testVariationalProbabilityCorrect(self):
    # Test with the complete Ipeirotis example:
    result = probability_correct.VariationalProbabilityCorrect((IPEIROTIS_ALPHA,
                                                                IPEIROTIS_BETA))
    test_util.AssertMapsAlmostEqual(self,
                                    IPEIROTIS_VARIATIONAL_PROBABILITY_CORRECT,
                                    result,
                                    label='contributor')

    # Now with the Dawid and Skene example (judgments plus resolutions):
    result = probability_correct.VariationalProbabilityCorrect((DS_ALPHA,
                                                                DS_BETA))
    test_util.AssertMapsAlmostEqual(self,
                                    DS_VARIATIONAL_PROBABILITY_CORRECT,
                                    result,
                                    label='contributor')

  def testSetMLEParameters(self):
    pc = probability_correct.ProbabilityCorrect()
    pc.SetMLEParameters(test_util.IPEIROTIS_DATA_FINAL)
    # Check that pc.probability_correct was set correctly:
    test_util.AssertMapsAlmostEqual(self,
                                    IPEIROTIS_MLE_PROBABILITY_CORRECT,
                                    pc.probability_correct,
                                    label='contributor')

  def testSetSampleParameters(self):
    pc = probability_correct.ProbabilityCorrect()
    pc.SetSampleParameters(test_util.IPEIROTIS_DATA_FINAL)
    # Check that pc.probability_correct was set to something sane:
    self.assertEqual(len(IPEIROTIS_MLE_PROBABILITY_CORRECT),
                     len(pc.probability_correct))

  def testSetVariationalParameters(self):
    pc = probability_correct.ProbabilityCorrect()
    pc.SetVariationalParameters(test_util.IPEIROTIS_DATA_FINAL)
    # Check that pc.probability_correct was set to something sane:
    self.assertEqual(len(IPEIROTIS_MLE_PROBABILITY_CORRECT),
                     len(pc.probability_correct))

  def testUpdatePrivateMembers(self):
    # Test that calling SetMLEParameters calls _UpdatePrivateMembers() to set
    # the members answer_space_size and probability_factor correctly, using the
    # data-with-answers for the Ipeirotis example:
    pc = probability_correct.ProbabilityCorrect()
    pc.SetMLEParameters(test_util.IPEIROTIS_DATA_FINAL)
    self.assertEqual(2, pc._answer_space_size)
    expected = {'worker1': 0.4 / 0.6,
                'worker2': 0.8 / 0.2,
                'worker3': float('inf'),  # thank you googletest
                'worker4': float('inf'),
                'worker5': 0.0}
    test_util.AssertMapsAlmostEqual(self,
                                    expected,
                                    pc._probability_factor,
                                    label='contributor')

  def testQuestionEntropy(self):
    pc = probability_correct.ProbabilityCorrect()
    pc.SetAnswerSpaceSize(10)
    # Test in binary:
    self.assertAlmostEqual(math.log(10, 2), pc.QuestionEntropy())
    # And in digits:
    self.assertAlmostEqual(1.0, pc.QuestionEntropy(radix=10))

  def testPointwiseMutualInformation(self):
    # Use data from the solution to the Ipeirotis example:
    pc = probability_correct.ProbabilityCorrect()
    pc.probability_correct = IPEIROTIS_MLE_PROBABILITY_CORRECT
    pc.SetAnswerSpaceSize(2)
    pc._UpdatePrivateMembers(None)

    # First we'll test the information of a first judgment.
    # Although worker1 is a spammer, the model understands only that it gives
    # the correct answer with probability 0.4 and the incorrect answer with
    # probability 0.6, so we think it gives a small amount of information.
    # worker2 gives partial information, and worker3, worker4, and worker5 all
    # give complete information, even though worker5 always lies.
    # Entries for impossible contingencies in the matrix below are filled in
    # with 0.0.
    expected = {'worker1': {'notporn': {'notporn': math.log(0.4 / 0.5, 2),
                                        'porn': math.log(0.6 / 0.5, 2)},
                            'porn': {'notporn': math.log(0.6 / 0.5, 2),
                                     'porn': math.log(0.4 / 0.5, 2)}},
                'worker2': {'notporn': {'notporn': math.log(0.8 / 0.5, 2),
                                        'porn': math.log(0.2 / 0.5, 2)},
                            'porn': {'notporn': math.log(0.2 / 0.5, 2),
                                     'porn': math.log(0.8 / 0.5, 2)}},
                'worker3': {'notporn': {'notporn': 1.0, 'porn': 0.0},
                            'porn': {'notporn': 0.0, 'porn': 1.0}},
                'worker4': {'notporn': {'notporn': 1.0, 'porn': 0.0},
                            'porn': {'notporn': 0.0, 'porn': 1.0}},
                'worker5': {'notporn': {'notporn': 0.0, 'porn': 1.0},
                            'porn': {'notporn': 1.0, 'porn': 0.0}}}
    # Pack the results into a three-dimensional structure (like confusion
    # matrices):
    result = {}
    for contributor in expected:
      result[contributor] = {}
      for answer in expected[contributor]:
        result[contributor][answer] = {}
        for judgment in expected[contributor][answer]:
          result[contributor][answer][judgment] = pc.PointwiseMutualInformation(
              contributor, answer, judgment)
    # Thus, we can use the method test_util.AssertConfusionMatricesAlmostEqual:
    test_util.AssertConfusionMatricesAlmostEqual(self, expected, result)

    # Now we'll test the information of a second judgment.
    # Start by supposing that worker2 gave judgment 'porn':
    previous_responses = [('worker2', 'porn', {})]

    # Suppose the correct answer is 'notporn', and the next judgment is
    # worker2 giving another 'porn' judgment.  After the first judgment, the
    # probability of the correct answer is 1/5 (odds 1:4).  After the second
    # judgment, the probability of the correct answer is 1/17 (odds 1:16).  So
    # the change in information is log(5/17), or about -1.76553 bits:
    self.assertAlmostEqual(math.log(5.0 / 17.0, 2),
                           pc.PointwiseMutualInformation(
                               'worker2', 'notporn', 'porn',
                               previous_responses=previous_responses))

    # Now suppose the correct answer is 'porn', and the next judgment is
    # worker2 giving another 'porn' judgment.  After the first judgment, the
    # probability of the correct answer is 4/5 (odds 4:1).  After the second
    # judgment, the probability of the correct answer is 16/17 (odds 16:1).  So
    # the change in information is log(20/17).
    # (In this test we'll use base 3 for the logarithm.)
    self.assertAlmostEqual(math.log(20.0 / 17.0, 3),
                           pc.PointwiseMutualInformation(
                               'worker2', 'porn', 'porn',
                               previous_responses=previous_responses,
                               radix=3))

    # Now suppose the correct answer is 'notporn', and the next judgment is
    # worker2 giving a 'notporn' judgment.  After the first judgment, the
    # probability of the correct answer is 1/5.  After the second judgment, the
    # probability of the correct answer is 1/2.  So the change in information
    # is log(5/2):
    self.assertAlmostEqual(math.log(2.5, 2), pc.PointwiseMutualInformation(
        'worker2', 'notporn', 'notporn', previous_responses=previous_responses))

    # Finally, suppose the correct answer is 'porn', and the next judgment is
    # worker5 giving a 'notporn' judgment.  After the first judgment, the
    # probability of the correct answer is 4/5.  After the second judgment, the
    # probability of the correct answer is 1.  So the change in information is
    # log(5/4):
    self.assertAlmostEqual(math.log(1.25, 2), pc.PointwiseMutualInformation(
        'worker5', 'porn', 'notporn', previous_responses=previous_responses))

  def testPointwiseMutualInformationLargeAnswerSpace(self):
    # This is just like the previous test, except that we'll set the answer
    # space size to 5 (instead of 2).  Only 2 answers are ever observed, and
    # the other 3 remain anonymous.
    pc = probability_correct.ProbabilityCorrect()
    pc.probability_correct = IPEIROTIS_MLE_PROBABILITY_CORRECT
    pc.SetAnswerSpaceSize(5)
    pc._UpdatePrivateMembers(None)

    # Information of a first judgment.  Now the prior probability of any answer
    # is 1/5, so the perfect workers give log(5) units of information.
    # Note that with the larger answer space, worker1's 40% accuracy is now
    # advantageous rather than adversarial, and worker5 (who always lies) is no
    # longer perfect.
    expected = {'worker1': {'notporn': {'notporn': math.log(2.0, 2),
                                        'porn': math.log(0.75, 2)},
                            'porn': {'notporn': math.log(0.75, 2),
                                     'porn': math.log(2.0, 2)}},
                'worker2': {'notporn': {'notporn': math.log(4.0, 2),
                                        'porn': math.log(0.25, 2)},
                            'porn': {'notporn': math.log(0.25, 2),
                                     'porn': math.log(4.0, 2)}},
                'worker3': {'notporn': {'notporn': math.log(5.0, 2),
                                        'porn': 0.0},
                            'porn': {'notporn': 0.0,
                                     'porn': math.log(5.0, 2)}},
                'worker4': {'notporn': {'notporn': math.log(5.0, 2),
                                        'porn': 0.0},
                            'porn': {'notporn': 0.0,
                                     'porn': math.log(5.0, 2)}},
                'worker5': {'notporn': {'notporn': 0.0,
                                        'porn': math.log(1.25, 2)},
                            'porn': {'notporn': math.log(1.25, 2),
                                     'porn': 0.0}}}
    # Pack the results into a three-dimensional structure (like confusion
    # matrices):
    result = {}
    for contributor in expected:
      result[contributor] = {}
      for answer in expected[contributor]:
        result[contributor][answer] = {}
        for judgment in expected[contributor][answer]:
          result[contributor][answer][judgment] = pc.PointwiseMutualInformation(
              contributor, answer, judgment)
    # Thus, we can use the method test_util.AssertConfusionMatricesAlmostEqual:
    test_util.AssertConfusionMatricesAlmostEqual(self, expected, result)

    # Information of a second judgment.  This time we'll start with worker1,
    # who is no longer a spammer in this regime!
    previous_responses = [('worker1', 'porn', {})]

    # Suppose the correct answer is 'notporn', and the next judgment is
    # worker1 giving another 'porn' judgment.  After the first judgment, the
    # probability of the correct answer is 0.15 (odds 3:8:3:3:3).  After
    # the second judgment, the probability of the correct answer is 0.09 (odds
    # 9:64:9:9:9).  So the change in information is log(3/5), or about -0.7370
    # bits:
    self.assertAlmostEqual(math.log(0.6, 2), pc.PointwiseMutualInformation(
        'worker1', 'notporn', 'porn', previous_responses=previous_responses))

    # Now suppose the correct answer is 'porn', and the next judgment is
    # worker1 giving another 'porn' judgment.  After the first judgment, the
    # probability of the correct answer is 0.40 (odds 8:3:3:3:3).  After
    # the second judgment, the probability of the correct answer is 0.64 (odds
    # 64:9:9:9:9).  So the change in information is log(8/5), or about 0.6781
    # bits:
    self.assertAlmostEqual(math.log(1.6, 2), pc.PointwiseMutualInformation(
        'worker1', 'porn', 'porn', previous_responses=previous_responses))

    # Now suppose the correct answer is 'notporn', and the next judgment is
    # worker1 giving a 'notporn' judgment.  After the first judgment, the
    # probability of the correct answer is 3/20 (odds 3:8:3:3:3).  After the
    # second judgment, the probability of the correct answer is 24/75 (odds
    # 24:24:9:9:9).  So the change in information is log(32/15):
    self.assertAlmostEqual(math.log(32.0 / 15.0, 2),
                           pc.PointwiseMutualInformation(
                               'worker1', 'notporn', 'notporn',
                               previous_responses=previous_responses))

    # Finally, suppose the correct answer is 'porn', and the next judgment is
    # worker4 giving a 'porn' judgment.  After the first judgment, the
    # probability of the correct answer is 0.4.  After the second judgment, the
    # probability of the correct answer is 1.  So the change in information is
    # log(2.5):
    self.assertAlmostEqual(math.log(2.5, 2), pc.PointwiseMutualInformation(
        'worker4', 'porn', 'porn', previous_responses=previous_responses))

  def testMutualInformation(self):
    # Use data from the solution to the Ipeirotis example:
    pc = probability_correct.ProbabilityCorrect()
    pc.probability_correct = IPEIROTIS_MLE_PROBABILITY_CORRECT
    s = 2.0
    pc.SetAnswerSpaceSize(s)  # to make sure it still works fine with s a float
    pc._UpdatePrivateMembers(None)

    # First we'll test for a first judgment:
    expected = {'worker1': (0.4 * math.log(2.0 * 0.4, 2) +
                            0.6 * math.log(2.0 * 0.6, 2)),
                'worker2': (0.8 * math.log(2.0 * 0.8, 2) +
                            0.2 * math.log(2.0 * 0.2, 2)),
                'worker3': 1.0,
                'worker4': 1.0,
                'worker5': 1.0}
    result = {}
    for contributor in IPEIROTIS_MLE_PROBABILITY_CORRECT:
      result[contributor] = pc.MutualInformation(contributor)
    test_util.AssertMapsAlmostEqual(self, expected, result, label='contributor')

    # Now we'll test for a second judgment.

    # We don't need to keep talking about porn, since the model understands
    # only the probability that a contributor gives a correct answer.  So let's
    # suppose that we start with worker2 giving the judgment '1', which sets
    # odds 4:1 in favor of the answer being '1'.
    # For convenience, we'll call the other member of the answer space '2',
    # although we never refer to it directly.
    previous_responses = [('worker2', 1, {})]  # Test that judgments can be ints

    # Each line in the calculations below takes the format
    #   P(a) * P(j_1|a) * log(P(a|j_1,j_2) / P(a|j_1)),
    # where a is the answer, j_1 is the judgment (1) given by worker2, and
    # j_2 is the judgment given by the next contributor.  What we're doing is
    # calculating the expected value of the logarithm (which is the change in
    # information) over all possible values of a and j_2, for each contributor.
    # The calculations of the arguments to the logarithm are pretty
    # straightforward if you think of them in terms of odds (as we did above).
    expected = {
        'worker1': (
            0.8 * 0.4 * math.log((8.0 / 11.0) / 0.8, 2) +   # a == 1, j == 1
            0.8 * 0.6 * math.log((6.0 / 7.0) / 0.8, 2) +    # a == 1, j == 2
            0.2 * 0.4 * math.log((1.0 / 7.0) / 0.2, 2) +    # a == 2, j == 2
            0.2 * 0.6 * math.log((3.0 / 11.0) / 0.2, 2)),   # a == 2, j == 1
        'worker2': (
            0.8 * 0.8 * math.log((16.0 / 17.0) / 0.8, 2) +  # a == 1, j == 1
            0.8 * 0.2 * math.log(0.5 / 0.8, 2) +            # a == 1, j == 2
            0.2 * 0.8 * math.log(0.5 / 0.2, 2) +            # a == 2, j == 2
            0.2 * 0.2 * math.log((1.0 / 17.0) / 0.2, 2)),   # a == 2, j == 1
        'worker3': (
            0.8 * math.log(1.0 / 0.8, 2) +                  # a == 1, j == 1
            0.2 * math.log(1.0 / 0.2, 2)),                  # a == 2, j == 2
        'worker4': (
            0.8 * math.log(1.0 / 0.8, 2) +                  # a == 1, j == 1
            0.2 * math.log(1.0 / 0.2, 2)),                  # a == 2, j == 2
        'worker5': (
            0.8 * math.log(1.0 / 0.8, 2) +                  # a == 1, j == 2
            0.2 * math.log(1.0 / 0.2, 2))}                  # a == 2, j == 1
    result = {}
    for contributor in expected:
      result[contributor] = pc.MutualInformation(
          contributor, previous_responses=previous_responses)
    test_util.AssertMapsAlmostEqual(self, expected, result, label='contributor')

    # However, if the first judgment was given by a perfect contributor (for
    # example, worker3), then no second judgment can give any more information:
    previous_responses = [('worker3', 1, {})]
    self.assertAlmostEqual(0.0, pc.MutualInformation(
        'worker2', previous_responses=previous_responses))

    # Finally, we'll test for a third judgment.  This is necessary in order to
    # check the case in which judgment and answer are distinct but are both
    # found in previous_responses.  In this example, the odds from
    # previous_responses are 1:6 for answers 1 and 2 respectively.
    previous_responses = [('worker1', 1, {}), ('worker2', 2, {})]
    expected = {
        'worker1': (
            (1.0 / 7.0) * 0.4 * math.log(0.1 / (1.0 / 7.0),   # a == 1, j == 1
                                         2) +
            (1.0 / 7.0) * 0.6 * math.log(0.2 / (1.0 / 7.0),   # a == 1, j == 2
                                         2) +
            (6.0 / 7.0) * 0.4 * math.log(0.8 / (6.0 / 7.0),   # a == 2, j == 2
                                         2) +
            (6.0 / 7.0) * 0.6 * math.log(0.9 / (6.0 / 7.0),   # a == 2, j == 1
                                         2)),
        'worker2': (
            (1.0 / 7.0) * 0.8 * math.log(0.4 / (1.0 / 7.0),   # a == 1, j == 1
                                         2) +
            (1.0 / 7.0) * 0.2 * math.log(0.04 / (1.0 / 7.0),  # a == 1, j == 2
                                         2) +
            (6.0 / 7.0) * 0.8 * math.log(0.96 / (6.0 / 7.0),  # a == 2, j == 2
                                         2) +
            (6.0 / 7.0) * 0.2 * math.log(0.6 / (6.0 / 7.0),   # a == 2, j == 1
                                         2)),
        'worker3': (
            (1.0 / 7.0) * math.log(1.0 / (1.0 / 7.0), 2) +    # a == 1, j == 1
            (6.0 / 7.0) * math.log(1.0 / (6.0 / 7.0), 2)),    # a == 2, j == 2
        'worker4': (
            (1.0 / 7.0) * math.log(1.0 / (1.0 / 7.0), 2) +    # a == 1, j == 1
            (6.0 / 7.0) * math.log(1.0 / (6.0 / 7.0), 2)),    # a == 2, j == 2
        'worker5': (
            (1.0 / 7.0) * math.log(1.0 / (1.0 / 7.0), 2) +    # a == 1, j == 2
            (6.0 / 7.0) * math.log(1.0 / (6.0 / 7.0), 2))}    # a == 2, j == 1
    result = {}
    for contributor in expected:
      result[contributor] = pc.MutualInformation(
          contributor, previous_responses=previous_responses)
    test_util.AssertMapsAlmostEqual(self, expected, result, label='contributor')

  def testMutualInformationLargeAnswerSpace(self):
    # This is just like the previous test, except that we'll set the answer
    # space size to 5 (instead of 2).
    pc = probability_correct.ProbabilityCorrect()
    pc.probability_correct = IPEIROTIS_MLE_PROBABILITY_CORRECT
    pc.SetAnswerSpaceSize(5)
    pc._UpdatePrivateMembers(None)

    # Suppose that we start with worker1 giving the judgment '1', which sets
    # odds 8:3:3:3:3 (for '1':*:*:*:*, where * is an anonymous answer).
    previous_responses = [('worker1', 1, {})]

    # As above, each line in the calculations below takes the format:
    #   P(a) * P(j_1|a) * log(P(a|j_1,j_2) / P(a|j_1))
    expected = {
        'worker1': (
            0.4 * 0.4 * math.log(0.64 / 0.4, 2) +  # a == 1, j == 1
            0.4 * 0.6 * math.log(0.32 / 0.4, 2) +  # a == 1, j != 1
            0.6 * 0.4 * math.log(0.32 / 0.15, 2) +  # a != 1, j == a
            0.6 * 0.15 * math.log(0.09 / 0.15, 2) +  # a != 1, j == 1
            0.6 * 0.45 * math.log(0.12 / 0.15, 2)),  # a != 1, j != a, j != 1
        'worker2': (
            0.4 * 0.8 * math.log((128.0 / 140.0) / 0.4, 2) +  # a == 1, j == 1
            0.4 * 0.2 * math.log((8.0 / 65.0) / 0.4, 2) +  # a == 1, j != 1
            0.6 * 0.8 * math.log((48.0 / 65.0) / 0.15, 2) +  # a != 1, j == a
            0.6 * 0.05 * math.log((3.0 / 140.0) / 0.15, 2) +  # a != 1, j == 1
            0.6 * 0.15 * math.log((3.0 / 65.0) / 0.15,
                                  2)),  # a != 1, j != a, j != 1
        'worker3': (
            0.4 * math.log(1.0 / 0.4, 2) +  # a == 1, j == a
            0.6 * math.log(1.0 / (3.0 / 20.0), 2)),  # a != 1, j == a
        'worker4': (
            0.4 * math.log(1.0 / 0.4, 2) +  # a == 1, j == a
            0.6 * math.log(1.0 / (3.0 / 20.0), 2)),  # a != 1, j == a
        'worker5': (
            0.4 * math.log((8.0 / 17.0) / (8.0 / 20.0), 2) +  # a == 1, j != a
            0.6 * 0.25 * math.log(0.25 / (3.0 / 20.0), 2) +  # a != 1, j == 1
            0.6 * 0.75 * math.log((3.0 / 17.0) / (3.0 / 20.0),
                                  2))}  # a != 1, j != a, j != 1
    result = {}
    for contributor in expected:
      result[contributor] = pc.MutualInformation(
          contributor, previous_responses=previous_responses)
    test_util.AssertMapsAlmostEqual(self, expected, result, label='contributor')

  def testResolveQuestion(self):
    # Test using data from the solution to the Ipeirotis example:
    pc = probability_correct.ProbabilityCorrect()
    pc.probability_correct = IPEIROTIS_MLE_PROBABILITY_CORRECT
    pc._UpdatePrivateMembers(test_util.IPEIROTIS_DATA)
    for i in range(len(test_util.IPEIROTIS_DATA)):
      resolution_map = pc.ResolveQuestion(test_util.IPEIROTIS_RESPONSES[i])
      test_util.AssertMapsAlmostEqual(self,
                                      test_util.IPEIROTIS_ALL_ANSWERS[i],
                                      resolution_map,
                                      label='question ' + str(i) + ', answer')
    # Test using conflicting answers from assumed-perfect contributors:
    result = pc.ResolveQuestion([('worker3', 'notporn', {}),
                                 ('worker4', 'porn', {})])
    self.assertEqual({}, result)
    # TODO(tpw):  Test with other values of pc._answer_space_size.


if __name__ == '__main__':
  unittest.main()
