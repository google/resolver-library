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

"""Tests for confusion_matrices.py."""

__author__ = 'tpw@google.com (Tamsyn Waterhouse)'

import collections
import math

import numpy
import scipy.special

import unittest
from resolver import confusion_matrices
from resolver import test_util


# Maximum-likelihood estimates from the Ipeirotis example:
IPEIROTIS_MLE_PRIORS = {'notporn': 0.6, 'porn': 0.4}
IPEIROTIS_MLE_CM = {
    'worker1': {'notporn': {'porn': 1.0},
                'porn': {'porn': 1.0}},
    'worker2': {'notporn': {'notporn': 2.0 / 3.0, 'porn': 1.0 / 3.0},
                'porn': {'porn': 1.0}},
    'worker3': {'notporn': {'notporn': 1.0},
                'porn': {'porn': 1.0}},
    'worker4': {'notporn': {'notporn': 1.0},
                'porn': {'porn': 1.0}},
    'worker5': {'notporn': {'porn': 1.0},
                'porn': {'notporn': 1.0}}}

# And for the Dawid & Skene example:
DS_MLE_PRIORS = {1: 0.400, 2: 0.422, 3: 0.112, 4: 0.067}
DS_MLE_CM = {
    1: {1: {1: 0.8895, 2: 0.1105},
        2: {1: 0.0706, 2: 0.8764, 3: 0.0530},
        3: {2: 0.3388, 3: 0.6612},
        4: {3: 0.5556, 4: 0.4444}},
    2: {1: {1: 0.8342, 2: 0.1658},
        2: {1: 0.0527, 2: 0.6329, 3: 0.3143},
        3: {3: 1.0},
        4: {4: 1.0}},
    3: {1: {1: 1.0},
        2: {1: 0.1064, 2: 0.7883, 3: 0.1053},
        3: {2: 0.4037, 3: 0.1988, 4: 0.3975},
        4: {3: 0.6667, 4: 0.3333}},
    4: {1: {1: 0.9444, 2: 0.0556},
        2: {1: 0.0537, 2: 0.8426, 3: 0.1037},
        3: {3: 0.8012, 4: 0.1988},
        4: {3: 0.3333, 4: 0.6667}},
    5: {1: {1: 1.0},
        2: {1: 0.1590, 2: 0.7345, 3: 0.1064},
        3: {2: 0.2091, 3: 0.7909},
        4: {3: 0.3333, 4: 0.6667}}}

# Dirichlet parameters for the Ipeirotis data:
IPEIROTIS_PRIORS_DIRICHLET = {'notporn': 4.0, 'porn': 3.0}
IPEIROTIS_CM_DIRICHLET = {'worker1': {'notporn': {'notporn': 1.0, 'porn': 4.0},
                                      'porn': {'notporn': 1.0, 'porn': 3.0}},
                          'worker2': {'notporn': {'notporn': 3.0, 'porn': 2.0},
                                      'porn': {'notporn': 1.0, 'porn': 3.0}},
                          'worker3': {'notporn': {'notporn': 4.0, 'porn': 1.0},
                                      'porn': {'notporn': 1.0, 'porn': 3.0}},
                          'worker4': {'notporn': {'notporn': 4.0, 'porn': 1.0},
                                      'porn': {'notporn': 1.0, 'porn': 3.0}},
                          'worker5': {'notporn': {'notporn': 1.0, 'porn': 4.0},
                                      'porn': {'notporn': 3.0, 'porn': 1.0}}}

# And for the Dawid & Skene data:
DS_PRIORS_DIRICHLET = {1: 18.98, 2: 19.989, 3: 6.031, 4: 4.0}
DS_CM_DIRICHLET = {1: {1: {1: 48.979, 2: 6.961, 3: 1.0, 4: 1.0},
                       2: {1: 5.021, 2: 50.925, 3: 4.021, 4: 1.0},
                       3: {1: 1.0, 2: 6.114, 3: 10.979, 4: 1.0},
                       4: {1: 1.0, 2: 1.0, 3: 6.0, 4: 5.0}},
                   2: {1: {1: 15.999, 2: 3.981, 3: 1.0, 4: 1.0},
                       2: {1: 2.001, 2: 13.019, 3: 6.969, 4: 1.0},
                       3: {1: 1.0, 2: 1.0, 3: 6.031, 4: 1.0},
                       4: {1: 1.0, 2: 1.0, 3: 1.0, 4: 4.0}},
                   3: {1: {1: 18.98, 2: 1.0, 3: 1.0, 4: 1.0},
                       2: {1: 3.02, 2: 15.969, 3: 3.0, 4: 1.0},
                       3: {1: 1.0, 2: 3.031, 3: 2.0, 4: 3.0},
                       4: {1: 1.0, 2: 1.0, 3: 3.0, 4: 2.0}},
                   4: {1: {1: 17.981, 2: 1.999, 3: 1.0, 4: 1.0},
                       2: {1: 2.019, 2: 17.001, 3: 2.969, 4: 1.0},
                       3: {1: 1.0, 2: 1.0, 3: 5.031, 4: 2.0},
                       4: {1: 1.0, 2: 1.0, 3: 2.0, 4: 3.0}},
                   5: {1: {1: 18.98, 2: 1.0, 3: 1.0, 4: 1.0},
                       2: {1: 4.02, 2: 14.948, 3: 3.021, 4: 1.0},
                       3: {1: 1.0, 2: 2.052, 3: 4.979, 4: 1.0},
                       4: {1: 1.0, 2: 1.0, 3: 2.0, 4: 3.0}}}

# Variational parameters from the Ipeirotis example (non-normalized!):
EXP = math.exp
DIGAMMA = scipy.special.psi
IPEIROTIS_VARIATIONAL_PRIORS = {'notporn': EXP(DIGAMMA(4.0)),
                                'porn': EXP(DIGAMMA(3.0))}
IPEIROTIS_VARIATIONAL_CM = {
    'worker1': {'notporn': {'notporn': EXP(DIGAMMA(1.0) - DIGAMMA(5.0)),
                            'porn': EXP(DIGAMMA(4.0) - DIGAMMA(5.0))},
                'porn': {'notporn': EXP(DIGAMMA(1.0) - DIGAMMA(4.0)),
                         'porn': EXP(DIGAMMA(3.0) - DIGAMMA(4.0))}},
    'worker2': {'notporn': {'notporn': EXP(DIGAMMA(3.0) - DIGAMMA(5.0)),
                            'porn': EXP(DIGAMMA(2.0) - DIGAMMA(5.0))},
                'porn': {'notporn': EXP(DIGAMMA(1.0) - DIGAMMA(4.0)),
                         'porn': EXP(DIGAMMA(3.0) - DIGAMMA(4.0))}},
    'worker3': {'notporn': {'notporn': EXP(DIGAMMA(4.0) - DIGAMMA(5.0)),
                            'porn': EXP(DIGAMMA(1.0) - DIGAMMA(5.0))},
                'porn': {'notporn': EXP(DIGAMMA(1.0) - DIGAMMA(4.0)),
                         'porn': EXP(DIGAMMA(3.0) - DIGAMMA(4.0))}},
    'worker4': {'notporn': {'notporn': EXP(DIGAMMA(4.0) - DIGAMMA(5.0)),
                            'porn': EXP(DIGAMMA(1.0) - DIGAMMA(5.0))},
                'porn': {'notporn': EXP(DIGAMMA(1.0) - DIGAMMA(4.0)),
                         'porn': EXP(DIGAMMA(3.0) - DIGAMMA(4.0))}},
    'worker5': {'notporn': {'notporn': EXP(DIGAMMA(1.0) - DIGAMMA(5.0)),
                            'porn': EXP(DIGAMMA(4.0) - DIGAMMA(5.0))},
                'porn': {'notporn': EXP(DIGAMMA(3.0) - DIGAMMA(4.0)),
                         'porn': EXP(DIGAMMA(1.0) - DIGAMMA(4.0))}}}

# And for the Dawid & Skene example:
DS_VARIATIONAL_PRIORS = {1: EXP(DIGAMMA(18.98)),
                         2: EXP(DIGAMMA(19.989)),
                         3: EXP(DIGAMMA(6.031)),
                         4: EXP(DIGAMMA(4.0))}
DS_VARIATIONAL_CM = {
    1: {1: {1: EXP(DIGAMMA(48.979) - DIGAMMA(57.94)),
            2: EXP(DIGAMMA(6.961) - DIGAMMA(57.94)),
            3: EXP(DIGAMMA(1.0) - DIGAMMA(57.94)),
            4: EXP(DIGAMMA(1.0) - DIGAMMA(57.94))},
        2: {1: EXP(DIGAMMA(5.021) - DIGAMMA(60.967)),
            2: EXP(DIGAMMA(50.925) - DIGAMMA(60.967)),
            3: EXP(DIGAMMA(4.021) - DIGAMMA(60.967)),
            4: EXP(DIGAMMA(1.0) - DIGAMMA(60.967))},
        3: {1: EXP(DIGAMMA(1.0) - DIGAMMA(19.093)),
            2: EXP(DIGAMMA(6.114) - DIGAMMA(19.093)),
            3: EXP(DIGAMMA(10.979) - DIGAMMA(19.093)),
            4: EXP(DIGAMMA(1.0) - DIGAMMA(19.093))},
        4: {1: EXP(DIGAMMA(1.0) - DIGAMMA(13.0)),
            2: EXP(DIGAMMA(1.0) - DIGAMMA(13.0)),
            3: EXP(DIGAMMA(6.0) - DIGAMMA(13.0)),
            4: EXP(DIGAMMA(5.0) - DIGAMMA(13.0))}},
    2: {1: {1: EXP(DIGAMMA(15.999) - DIGAMMA(21.98)),
            2: EXP(DIGAMMA(3.981) - DIGAMMA(21.98)),
            3: EXP(DIGAMMA(1.0) - DIGAMMA(21.98)),
            4: EXP(DIGAMMA(1.0) - DIGAMMA(21.98))},
        2: {1: EXP(DIGAMMA(2.001) - DIGAMMA(22.989)),
            2: EXP(DIGAMMA(13.019) - DIGAMMA(22.989)),
            3: EXP(DIGAMMA(6.969) - DIGAMMA(22.989)),
            4: EXP(DIGAMMA(1.0) - DIGAMMA(22.989))},
        3: {1: EXP(DIGAMMA(1.0) - DIGAMMA(9.031)),
            2: EXP(DIGAMMA(1.0) - DIGAMMA(9.031)),
            3: EXP(DIGAMMA(6.031) - DIGAMMA(9.031)),
            4: EXP(DIGAMMA(1.0) - DIGAMMA(9.031))},
        4: {1: EXP(DIGAMMA(1.0) - DIGAMMA(7.0)),
            2: EXP(DIGAMMA(1.0) - DIGAMMA(7.0)),
            3: EXP(DIGAMMA(1.0) - DIGAMMA(7.0)),
            4: EXP(DIGAMMA(4.0) - DIGAMMA(7.0))}},
    3: {1: {1: EXP(DIGAMMA(18.98) - DIGAMMA(21.98)),
            2: EXP(DIGAMMA(1.0) - DIGAMMA(21.98)),
            3: EXP(DIGAMMA(1.0) - DIGAMMA(21.98)),
            4: EXP(DIGAMMA(1.0) - DIGAMMA(21.98))},
        2: {1: EXP(DIGAMMA(3.02) - DIGAMMA(22.989)),
            2: EXP(DIGAMMA(15.969) - DIGAMMA(22.989)),
            3: EXP(DIGAMMA(3.0) - DIGAMMA(22.989)),
            4: EXP(DIGAMMA(1.0) - DIGAMMA(22.989))},
        3: {1: EXP(DIGAMMA(1.0) - DIGAMMA(9.031)),
            2: EXP(DIGAMMA(3.031) - DIGAMMA(9.031)),
            3: EXP(DIGAMMA(2.0) - DIGAMMA(9.031)),
            4: EXP(DIGAMMA(3.0) - DIGAMMA(9.031))},
        4: {1: EXP(DIGAMMA(1.0) - DIGAMMA(7.0)),
            2: EXP(DIGAMMA(1.0) - DIGAMMA(7.0)),
            3: EXP(DIGAMMA(3.0) - DIGAMMA(7.0)),
            4: EXP(DIGAMMA(2.0) - DIGAMMA(7.0))}},
    4: {1: {1: EXP(DIGAMMA(17.981) - DIGAMMA(21.98)),
            2: EXP(DIGAMMA(1.999) - DIGAMMA(21.98)),
            3: EXP(DIGAMMA(1.0) - DIGAMMA(21.98)),
            4: EXP(DIGAMMA(1.0) - DIGAMMA(21.98))},
        2: {1: EXP(DIGAMMA(2.019) - DIGAMMA(22.989)),
            2: EXP(DIGAMMA(17.001) - DIGAMMA(22.989)),
            3: EXP(DIGAMMA(2.969) - DIGAMMA(22.989)),
            4: EXP(DIGAMMA(1.0) - DIGAMMA(22.989))},
        3: {1: EXP(DIGAMMA(1.0) - DIGAMMA(9.031)),
            2: EXP(DIGAMMA(1.0) - DIGAMMA(9.031)),
            3: EXP(DIGAMMA(5.031) - DIGAMMA(9.031)),
            4: EXP(DIGAMMA(2.0) - DIGAMMA(9.031))},
        4: {1: EXP(DIGAMMA(1.0) - DIGAMMA(7.0)),
            2: EXP(DIGAMMA(1.0) - DIGAMMA(7.0)),
            3: EXP(DIGAMMA(2.0) - DIGAMMA(7.0)),
            4: EXP(DIGAMMA(3.0) - DIGAMMA(7.0))}},
    5: {1: {1: EXP(DIGAMMA(18.98) - DIGAMMA(21.98)),
            2: EXP(DIGAMMA(1.0) - DIGAMMA(21.98)),
            3: EXP(DIGAMMA(1.0) - DIGAMMA(21.98)),
            4: EXP(DIGAMMA(1.0) - DIGAMMA(21.98))},
        2: {1: EXP(DIGAMMA(4.02) - DIGAMMA(22.989)),
            2: EXP(DIGAMMA(14.948) - DIGAMMA(22.989)),
            3: EXP(DIGAMMA(3.021) - DIGAMMA(22.989)),
            4: EXP(DIGAMMA(1.0) - DIGAMMA(22.989))},
        3: {1: EXP(DIGAMMA(1.0) - DIGAMMA(9.031)),
            2: EXP(DIGAMMA(2.052) - DIGAMMA(9.031)),
            3: EXP(DIGAMMA(4.979) - DIGAMMA(9.031)),
            4: EXP(DIGAMMA(1.0) - DIGAMMA(9.031))},
        4: {1: EXP(DIGAMMA(1.0) - DIGAMMA(7.0)),
            2: EXP(DIGAMMA(1.0) - DIGAMMA(7.0)),
            3: EXP(DIGAMMA(2.0) - DIGAMMA(7.0)),
            4: EXP(DIGAMMA(3.0) - DIGAMMA(7.0))}}}


class ConfusionMatricesTest(unittest.TestCase):
  longMessage = True
  numpy.seterr(all='raise')

  def testMLEResolutionPriors(self):
    # First check that we get an agnostic prior when we have no resolutions:
    test_util.AssertMapsAlmostEqual(
        self,
        {},
        confusion_matrices.MLEResolutionPriors(test_util.DS_DATA),
        label='answer')
    # Now check that we get an agnostic prior when we have one resolution of
    # each of two answers:
    test_util.AssertMapsAlmostEqual(
        self,
        {'notporn': 0.5, 'porn': 0.5},
        confusion_matrices.MLEResolutionPriors(test_util.IPEIROTIS_DATA),
        label='answer')
    # Now check that we get the correct prior when we do have resolutions:
    test_util.AssertMapsAlmostEqual(
        self,
        IPEIROTIS_MLE_PRIORS,
        confusion_matrices.MLEResolutionPriors(test_util.IPEIROTIS_DATA_FINAL),
        label='answer')
    # And again for the Dawid & Skene example:
    test_util.AssertMapsAlmostEqual(
        self,
        DS_MLE_PRIORS,
        confusion_matrices.MLEResolutionPriors(test_util.DS_DATA_FINAL),
        label='answer')
    # Check that the weighted test data gives the same results as the original:
    test_util.AssertMapsAlmostEqual(
        self,
        DS_MLE_PRIORS,
        confusion_matrices.MLEResolutionPriors(
            test_util.DS_DATA_EXTRA,
            question_weights=test_util.DS_EXTRA_WEIGHTS),
        label='answer')

  def testMLEConfusionMatrices(self):
    # Check that MLEConfusionMatrices returns agnostic matrices when there are
    # no resolutions:
    result = confusion_matrices.MLEConfusionMatrices(test_util.DS_DATA)
    test_util.AssertConfusionMatricesAlmostEqual(self, {}, result)
    # Check that MLEConfusionMatrices returns the correct matrices for the
    # Ipeirotis example:
    result = confusion_matrices.MLEConfusionMatrices(
        test_util.IPEIROTIS_DATA_FINAL)
    test_util.AssertConfusionMatricesAlmostEqual(self, IPEIROTIS_MLE_CM, result)
    # And for the Dawid & Skene example:
    result = confusion_matrices.MLEConfusionMatrices(
        test_util.DS_DATA_FINAL)
    test_util.AssertConfusionMatricesAlmostEqual(self, DS_MLE_CM, result)
    # Check that the weighted test data gives the same results as the original:
    result = confusion_matrices.MLEConfusionMatrices(
        test_util.DS_DATA_EXTRA, question_weights=test_util.DS_EXTRA_WEIGHTS)
    test_util.AssertConfusionMatricesAlmostEqual(self, DS_MLE_CM, result)

  def testResolutionPriorsDirichletParameters(self):
    # Check the Dirichlet alpha vector for the Ipeirotis data:
    result = confusion_matrices.ResolutionPriorsDirichletParameters(
        test_util.IPEIROTIS_DATA_FINAL)
    test_util.AssertMapsAlmostEqual(
        self, IPEIROTIS_PRIORS_DIRICHLET, result, label='answer')
    # And for the Dawid & Skene data:
    result = confusion_matrices.ResolutionPriorsDirichletParameters(
        test_util.DS_DATA_FINAL)
    test_util.AssertMapsAlmostEqual(
        self, DS_PRIORS_DIRICHLET, result, label='answer')
    # Check that the weighted test data gives the same results as the original:
    result = confusion_matrices.ResolutionPriorsDirichletParameters(
        test_util.DS_DATA_EXTRA, question_weights=test_util.DS_EXTRA_WEIGHTS)
    test_util.AssertMapsAlmostEqual(self, DS_PRIORS_DIRICHLET, result,
                                    label='answer')

  def testVariationalResolutionPriors(self):
    # Test with the Ipeirotis data:
    test_util.AssertMapsAlmostEqual(
        self,
        IPEIROTIS_VARIATIONAL_PRIORS,
        confusion_matrices.VariationalResolutionPriors(
            IPEIROTIS_PRIORS_DIRICHLET),
        label='answer')
    # And again with the Dawid & Skene data:
    test_util.AssertMapsAlmostEqual(
        self,
        DS_VARIATIONAL_PRIORS,
        confusion_matrices.VariationalResolutionPriors(DS_PRIORS_DIRICHLET),
        label='answer')

  def testSampleResolutionPriors(self):
    # Seed the random number generator to produce deterministic test results:
    numpy.random.seed(0)

    # We also need the Dirichlet parameter vectors to have fixed order:
    ipeirotis_priors_dirichlet_ordered = collections.OrderedDict(
        sorted(IPEIROTIS_PRIORS_DIRICHLET.iteritems()))
    ds_priors_dirichlet_ordered = collections.OrderedDict(
        sorted(DS_PRIORS_DIRICHLET.iteritems()))

    # Check that a set of randomly-sampled priors sums to unity:
    self.assertAlmostEqual(
        1.0,
        sum(confusion_matrices.SampleResolutionPriors(
            ipeirotis_priors_dirichlet_ordered).itervalues()))
    self.assertAlmostEqual(
        1.0,
        sum(confusion_matrices.SampleResolutionPriors(
            ds_priors_dirichlet_ordered).itervalues()))

    # Check that the mean of 50000 samples is close to the actual mean of the
    # Dirichlet distribution, which is the normalized alpha vector:
    alpha = DS_PRIORS_DIRICHLET
    norm = sum(alpha.itervalues())
    expected_mean = dict([(answer, alpha[answer] / norm) for answer in alpha])
    samples = [confusion_matrices.SampleResolutionPriors(alpha)
               for _ in range(50000)]
    mean = {}
    for answer in alpha:
      mean[answer] = numpy.mean([sample[answer] for sample in samples])
    test_util.AssertMapsAlmostEqual(self, expected_mean, mean, label='answer')
    # Also check the variance, which is given by the formula below:
    expected_variance = {}
    variance = {}
    for answer in alpha:
      expected_variance[answer] = (
          alpha[answer] * (norm - alpha[answer]) / (norm * norm * (norm + 1.0)))
      variance[answer] = numpy.var([sample[answer] for sample in samples])
    test_util.AssertMapsAlmostEqual(self, expected_variance, variance,
                                    label='answer', places=4)

  def testConfusionMatricesDirichletParameters(self):
    # Check the Dirichlet parameters for the Ipeirotis data:
    result = confusion_matrices.ConfusionMatricesDirichletParameters(
        test_util.IPEIROTIS_DATA_FINAL)
    test_util.AssertConfusionMatricesAlmostEqual(self,
                                                 IPEIROTIS_CM_DIRICHLET, result,
                                                 places=2)
    # And for the Dawid & Skene data:
    result = confusion_matrices.ConfusionMatricesDirichletParameters(
        test_util.DS_DATA_FINAL)
    test_util.AssertConfusionMatricesAlmostEqual(self,
                                                 DS_CM_DIRICHLET, result,
                                                 places=2)
    # Check that the weighted test data gives the same results as the original:
    result = confusion_matrices.ConfusionMatricesDirichletParameters(
        test_util.DS_DATA_EXTRA, question_weights=test_util.DS_EXTRA_WEIGHTS)
    test_util.AssertConfusionMatricesAlmostEqual(self,
                                                 DS_CM_DIRICHLET, result,
                                                 places=2)
    # Test that the Dirichlet vectors include judgments for questions with
    # empty resolutions (this was once not true due to a whitespace bug):
    result = confusion_matrices.ConfusionMatricesDirichletParameters(
        {'q1': ([('c1', 'B', None)], {'A': 1.0}),
         'q2': ([('c1', 'C', None)], {})})
    test_util.AssertConfusionMatricesAlmostEqual(
        self,
        {'c1': {'A': {'B': 2.0, 'C': 1.0}}},
        result)

  def testVariationalConfusionMatrices(self):
    # Test with the Ipeirotis data:
    test_util.AssertConfusionMatricesAlmostEqual(
        self,
        IPEIROTIS_VARIATIONAL_CM,
        confusion_matrices.VariationalConfusionMatrices(IPEIROTIS_CM_DIRICHLET))
    # And again with the Dawid & Skene data:
    test_util.AssertConfusionMatricesAlmostEqual(
        self,
        DS_VARIATIONAL_CM,
        confusion_matrices.VariationalConfusionMatrices(DS_CM_DIRICHLET))

  def testSampleConfusionMatrices(self):
    # Seed the random number generator to produce deterministic test results:
    numpy.random.seed(0)

    # We also need the Dirichlet parameter dicts to have fixed iteration order:
    ipeirotis_cm_dirichlet_ordered = collections.OrderedDict()
    for contributor in sorted(IPEIROTIS_CM_DIRICHLET):
      matrix = collections.OrderedDict()
      for answer in sorted(IPEIROTIS_CM_DIRICHLET[contributor]):
        row = collections.OrderedDict(
            sorted(IPEIROTIS_CM_DIRICHLET[contributor][answer].iteritems()))
        matrix[answer] = row
      ipeirotis_cm_dirichlet_ordered[contributor] = matrix

    # And also for the Dawid & Skene data:
    ds_cm_dirichlet_ordered = collections.OrderedDict()
    for contributor in sorted(DS_CM_DIRICHLET):
      matrix = collections.OrderedDict()
      for answer in sorted(DS_CM_DIRICHLET[contributor]):
        row = collections.OrderedDict(
            sorted(DS_CM_DIRICHLET[contributor][answer].iteritems()))
        matrix[answer] = row
      ds_cm_dirichlet_ordered[contributor] = matrix

    # Check that each row of a set of randomly-sampled matrices sums to unity:
    result = confusion_matrices.SampleConfusionMatrices(
        ipeirotis_cm_dirichlet_ordered)
    for contributor in result:
      for answer in result[contributor]:
        self.assertAlmostEqual(1.0,
                               sum(result[contributor][answer].itervalues()))

    # Check that the mean of 10000 samples is close to the actual mean of the
    # Dirichlet distribution for a set of confusion matrices, for a case with a
    # narrow distribution (the Dawid and Skene example):
    samples = [
        confusion_matrices.SampleConfusionMatrices(ds_cm_dirichlet_ordered)
        for _ in range(10000)]
    expected = {1: {1: {1: 0.845, 2: 0.120, 3: 0.017, 4: 0.017},
                    2: {1: 0.082, 2: 0.835, 3: 0.066, 4: 0.016},
                    3: {1: 0.052, 2: 0.320, 3: 0.575, 4: 0.052},
                    4: {1: 0.077, 2: 0.077, 3: 0.462, 4: 0.385}},
                2: {1: {1: 0.728, 2: 0.181, 3: 0.045, 4: 0.045},
                    2: {1: 0.087, 2: 0.566, 3: 0.303, 4: 0.043},
                    3: {1: 0.111, 2: 0.111, 3: 0.668, 4: 0.111},
                    4: {1: 0.143, 2: 0.143, 3: 0.143, 4: 0.571}},
                3: {1: {1: 0.864, 2: 0.045, 3: 0.045, 4: 0.045},
                    2: {1: 0.131, 2: 0.694, 3: 0.130, 4: 0.043},
                    3: {1: 0.111, 2: 0.336, 3: 0.221, 4: 0.332},
                    4: {1: 0.143, 2: 0.143, 3: 0.429, 4: 0.286}},
                4: {1: {1: 0.818, 2: 0.091, 3: 0.045, 4: 0.045},
                    2: {1: 0.088, 2: 0.740, 3: 0.129, 4: 0.043},
                    3: {1: 0.111, 2: 0.111, 3: 0.557, 4: 0.221},
                    4: {1: 0.143, 2: 0.143, 3: 0.286, 4: 0.429}},
                5: {1: {1: 0.864, 2: 0.045, 3: 0.045, 4: 0.045},
                    2: {1: 0.175, 2: 0.650, 3: 0.131, 4: 0.043},
                    3: {1: 0.111, 2: 0.227, 3: 0.551, 4: 0.111},
                    4: {1: 0.143, 2: 0.143, 3: 0.286, 4: 0.429}}}
    mean = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    for contributor in expected:
      for answer in expected[contributor]:
        for judgment in expected[contributor][answer]:
          mean[contributor][answer][judgment] = numpy.mean(
              [sample[contributor][answer].get(judgment, 0.0)
               for sample in samples])
    test_util.AssertConfusionMatricesAlmostEqual(self, expected, mean, places=2)

    # And again for a broad distribution (the Ipeirotis example), although now
    # we need 20000 samples to get a similar precision:
    samples = [
        confusion_matrices.SampleConfusionMatrices(
            ipeirotis_cm_dirichlet_ordered) for _ in range(20000)]
    expected = {'worker1': {'notporn': {'notporn': 0.2, 'porn': 0.8},
                            'porn': {'notporn': 0.25, 'porn': 0.75}},
                'worker2': {'notporn': {'notporn': 0.6, 'porn': 0.4},
                            'porn': {'notporn': 0.25, 'porn': 0.75}},
                'worker3': {'notporn': {'notporn': 0.8, 'porn': 0.2},
                            'porn': {'notporn': 0.25, 'porn': 0.75}},
                'worker4': {'notporn': {'notporn': 0.8, 'porn': 0.2},
                            'porn': {'notporn': 0.25, 'porn': 0.75}},
                'worker5': {'notporn': {'notporn': 0.2, 'porn': 0.8},
                            'porn': {'notporn': 0.75, 'porn': 0.25}}}
    mean = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    for contributor in expected:
      for answer in expected[contributor]:
        for judgment in expected[contributor][answer]:
          mean[contributor][answer][judgment] = numpy.mean(
              [sample[contributor][answer].get(judgment, 0.0)
               for sample in samples])
    test_util.AssertConfusionMatricesAlmostEqual(self, expected, mean, places=2)

    # We'll also check the sample variance in this case:
    # Define three quantities for convenience in the matrix below:
    x = 4.0 / 150.0
    y = 6.0 / 150.0
    z = 3.0 / 80.0
    expected = {'worker1': {'notporn': {'notporn': x, 'porn': x},
                            'porn': {'notporn': z, 'porn': z}},
                'worker2': {'notporn': {'notporn': y, 'porn': y},
                            'porn': {'notporn': z, 'porn': z}},
                'worker3': {'notporn': {'notporn': x, 'porn': x},
                            'porn': {'notporn': z, 'porn': z}},
                'worker4': {'notporn': {'notporn': x, 'porn': x},
                            'porn': {'notporn': z, 'porn': z}},
                'worker5': {'notporn': {'notporn': x, 'porn': x},
                            'porn': {'notporn': z, 'porn': z}}}
    variance = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    for contributor in expected:
      for answer in expected[contributor]:
        for judgment in expected[contributor][answer]:
          variance[contributor][answer][judgment] = numpy.var(
              [sample[contributor][answer].get(judgment, 0.0)
               for sample in samples])
    test_util.AssertConfusionMatricesAlmostEqual(self, expected, variance,
                                                 places=2)

  def testSetMLEParameters(self):
    # Use data from the solution to the Ipeirotis example:
    cm = confusion_matrices.ConfusionMatrices()
    cm.SetMLEParameters(test_util.IPEIROTIS_DATA_FINAL)
    # Check that cm.priors was set correctly:
    test_util.AssertMapsAlmostEqual(self, IPEIROTIS_MLE_PRIORS, cm.priors,
                                    label='answer')
    # Check that cm.confusion_matrices was set correctly:
    test_util.AssertConfusionMatricesAlmostEqual(self,
                                                 IPEIROTIS_MLE_CM,
                                                 cm.confusion_matrices)

  def testSetVariationalParameters(self):
    # Use data from the solution to the Ipeirotis example:
    cm = confusion_matrices.ConfusionMatrices()
    cm.SetVariationalParameters(test_util.IPEIROTIS_DATA_FINAL)
    # Check that cm.priors was set correctly:
    test_util.AssertMapsAlmostEqual(self,
                                    IPEIROTIS_VARIATIONAL_PRIORS, cm.priors,
                                    label='answer')
    # Check that cm.confusion_matrices was set correctly:
    test_util.AssertConfusionMatricesAlmostEqual(self,
                                                 IPEIROTIS_VARIATIONAL_CM,
                                                 cm.confusion_matrices)

  def testQuestionEntropy(self):
    # Use data from the solution to the Ipeirotis example:
    cm = confusion_matrices.ConfusionMatrices()
    cm.priors = IPEIROTIS_MLE_PRIORS
    # Test in binary:
    self.assertAlmostEqual(0.9709506, cm.QuestionEntropy())
    # And in digits:
    self.assertAlmostEqual(0.2922853, cm.QuestionEntropy(radix=10))
    # Ensure that we correctly normalize non-normalized priors (those produced
    # by VariationalParameters, for example):
    cm.priors = {k: v * 3.0 for k, v in IPEIROTIS_MLE_PRIORS.iteritems()}
    self.assertAlmostEqual(0.9709506, cm.QuestionEntropy())
    self.assertAlmostEqual(0.2922853, cm.QuestionEntropy(radix=10))

  def testPointwiseMutualInformation(self):
    # Use data from the solution to the Ipeirotis example:
    cm = confusion_matrices.ConfusionMatrices()
    cm.priors = IPEIROTIS_MLE_PRIORS
    cm.confusion_matrices = IPEIROTIS_MLE_CM

    # First we'll test the information of a first judgment.
    # We're going to define four helper variables:
    # Two for the information conent of the true resolutions:
    notporn_inf = -math.log(cm.priors['notporn'], 2)
    porn_inf = -math.log(cm.priors['porn'], 2)
    # Two for the information given by contributor 'worker2', judgment 'porn':
    norm = (cm.priors['notporn'] *
            cm.confusion_matrices['worker2']['notporn']['porn'] +
            cm.priors['porn'] *
            cm.confusion_matrices['worker2']['porn']['porn'])
    inf_01 = math.log(
        cm.confusion_matrices['worker2']['notporn']['porn'] / norm, 2)
    inf_11 = math.log(
        cm.confusion_matrices['worker2']['porn']['porn'] / norm, 2)
    # Now, worker1 is a spammer which always gives us zero information.
    # worker2 gives us complete information when it gives judgment 'notporn',
    # but partial information when it gives judgment 'porn'.
    # The other three contributors give us complete information for all of
    # their judgments, even though worker5 always lies.
    # Entries for impossible contingencies in the matrix below are filled in
    # with 0.0.
    expected = {'worker1': {'notporn': {'notporn': 0.0, 'porn': 0.0},
                            'porn': {'notporn': 0.0, 'porn': 0.0}},
                'worker2': {'notporn': {'notporn': notporn_inf, 'porn': inf_01},
                            'porn': {'notporn': 0.0, 'porn': inf_11}},
                'worker3': {'notporn': {'notporn': notporn_inf, 'porn': 0.0},
                            'porn': {'notporn': 0.0, 'porn': porn_inf}},
                'worker4': {'notporn': {'notporn': notporn_inf, 'porn': 0.0},
                            'porn': {'notporn': 0.0, 'porn': porn_inf}},
                'worker5': {'notporn': {'notporn': 0.0, 'porn': notporn_inf},
                            'porn': {'notporn': porn_inf, 'porn': 0.0}}}
    # Pack the results into a structure having the same form as
    # cm.confusion_matrices:
    result = {}
    for contributor in expected:
      result[contributor] = {}
      for answer in expected[contributor]:
        result[contributor][answer] = {}
        for judgment in expected[contributor][answer]:
          result[contributor][answer][judgment] = cm.PointwiseMutualInformation(
              contributor, answer, judgment)
    # Thus, we can use the method test_util.AssertConfusionMatricesAlmostEqual:
    test_util.AssertConfusionMatricesAlmostEqual(self, expected, result)

    # Now we'll test the information of a second judgment.
    # Start by supposing that worker2 gave judgment 'porn':
    previous_responses = [('worker2', 'porn', {})]

    # Suppose the correct answer is 'notporn', and the next judgment is
    # worker2 giving another 'porn' judgment.  After the first judgment, the
    # probability of the correct answer is 1/3.  After the second judgment, the
    # probability of the correct answer is 1/7.  So the change in information is
    # log(3/7), or about -1.222392 bits:
    self.assertAlmostEqual(math.log(3.0/7.0, 2), cm.PointwiseMutualInformation(
        'worker2', 'notporn', 'porn', previous_responses=previous_responses))

    # Now suppose the correct answer is 'porn', and the next judgment is
    # worker2 giving another 'porn' judgment.  After the first judgment, the
    # probability of the correct answer is 2/3.  After the second judgment, the
    # probability of the correct answer is 6/7.  So the change in information is
    # log(9/7):
    self.assertAlmostEqual(math.log(9.0/7.0, 2), cm.PointwiseMutualInformation(
        'worker2', 'porn', 'porn', previous_responses=previous_responses))

    # Now suppose the correct answer is 'notporn', and the next judgment is
    # worker2 giving a 'notporn' judgment.  After the first judgment, the
    # probability of the correct answer is 1/3.  After the second judgment, the
    # probability of the correct answer is 1.  So the change in information is
    # log(3):
    self.assertAlmostEqual(math.log(3.0, 2), cm.PointwiseMutualInformation(
        'worker2', 'notporn', 'notporn', previous_responses=previous_responses))

    # Finally, suppose the correct answer is 'porn', and the next judgment is
    # worker5 giving a 'notporn' judgment.  After the first judgment, the
    # probability of the correct answer is 2/3.  After the second judgment, the
    # probability of the correct answer is 1.  So the change in information is
    # log(3/2):
    self.assertAlmostEqual(math.log(1.5, 2), cm.PointwiseMutualInformation(
        'worker5', 'porn', 'notporn', previous_responses=previous_responses))

  def testMutualInformation(self):
    # Use data from the solution to the Ipeirotis example:
    cm = confusion_matrices.ConfusionMatrices()
    cm.priors = IPEIROTIS_MLE_PRIORS
    cm.confusion_matrices = IPEIROTIS_MLE_CM

    # First we'll test for a first judgment:
    expected = {'worker1': 0.0,
                'worker2': 0.419973,
                'worker3': 0.970951,
                'worker4': 0.970951,
                'worker5': 0.970951}
    result = {}
    for contributor in expected:
      result[contributor] = cm.MutualInformation(contributor)
    test_util.AssertMapsAlmostEqual(self, expected, result, label='contributor')

    # Now we'll test for a second judgment:
    previous_responses = [('worker2', 'porn', {})]
    expected = {'worker1': 0.0,
                'worker2': 0.4581059,
                'worker3': 0.9182958,
                'worker4': 0.9182958,
                'worker5': 0.9182958}
    result = {}
    for contributor in expected:
      result[contributor] = cm.MutualInformation(
          contributor, previous_responses=previous_responses)
    test_util.AssertMapsAlmostEqual(self, expected, result, label='contributor')

    # However, if the first judgment was given by a perfect contributor (for
    # example, worker3), then no second judgment can give any more information:
    previous_responses = [('worker3', 'notporn', {})]
    self.assertAlmostEqual(0.0, cm.MutualInformation(
        'worker2', previous_responses=previous_responses))

  def testResolveQuestion(self):
    # First check that we get nothing back when we test with an empty model:
    cm = confusion_matrices.ConfusionMatrices()
    cm.priors = {}
    cm.confusion_matrices = {}
    resolution_map = cm.ResolveQuestion(test_util.IPEIROTIS_RESPONSES[0])
    test_util.AssertMapsAlmostEqual(self, {}, resolution_map)

    # Use data from the solution to the Ipeirotis example:
    cm = confusion_matrices.ConfusionMatrices()
    cm.priors = IPEIROTIS_MLE_PRIORS
    cm.confusion_matrices = IPEIROTIS_MLE_CM
    for i in range(len(test_util.IPEIROTIS_DATA)):
      resolution_map = cm.ResolveQuestion(test_util.IPEIROTIS_RESPONSES[i])
      test_util.AssertMapsAlmostEqual(self,
                                      test_util.IPEIROTIS_ALL_ANSWERS[i],
                                      resolution_map,
                                      label='question ' + str(i) + ', answer')
    # And again for the Dawid & Skene example:
    cm.priors = DS_MLE_PRIORS
    cm.confusion_matrices = DS_MLE_CM
    for i in range(len(test_util.DS_DATA)):
      resolution_map = cm.ResolveQuestion(test_util.DS_RESPONSES[i])
      test_util.AssertMapsAlmostEqual(self,
                                      test_util.DS_EM_CM_RESOLUTIONS[i],
                                      resolution_map,
                                      label='question ' + str(i) + ', answer')


if __name__ == '__main__':
  unittest.main()
