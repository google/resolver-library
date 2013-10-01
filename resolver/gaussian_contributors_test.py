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


import copy

import numpy

import unittest
from resolver import gaussian_contributors
from resolver import model
from resolver import test_util


TEST_DATA = {'q1': ([('c1', 6.0, {}), ('c2', 6.5, {}), ('c3', 5.5, {})],
                    {gaussian_contributors.MEAN: 5.0,
                     gaussian_contributors.VARIANCE: 0.0}),
             'q2': ([('c1', 7.0, {}), ('c2', 6.5, {}), ('c3', 5.5, {})],
                    {gaussian_contributors.MEAN: 6.0,
                     gaussian_contributors.VARIANCE: 0.0}),
             'q3': ([('c1', 8.0, {}), ('c2', 6.5, {}), ('c3', 5.5, {})],
                    {gaussian_contributors.MEAN: 7.0,
                     gaussian_contributors.VARIANCE: 0.0}),
             'q4': ([('c1', 9.0, {}), ('c2', 6.5, {}), ('c3', 5.5, {})],
                    {gaussian_contributors.MEAN: 8.0,
                     gaussian_contributors.VARIANCE: 0.0})}

TEST_JUDGMENT_COUNT = {'c1': 4.0,
                       'c2': 4.0,
                       'c3': 4.0}
TEST_MLE_BIAS = {'c1': 1.0 * (1.0 - gaussian_contributors.EPSILON),
                 'c2': 0.0,
                 'c3': -1.0 * (1.0 - gaussian_contributors.EPSILON)}
TEST_SUM_SQUARED_DEVIATION = {'c1': 0.0,
                              'c2': 5.0,
                              'c3': 5.0}
TEST_MLE_PRECISION = {'c1': gaussian_contributors.INFINITY,
                      'c2': 0.8,
                      'c3': 0.8}
TEST_VARIATIONAL_PRECISION = {'c1': gaussian_contributors.INFINITY,
                              'c2': 1.0,
                              'c3': 1.0}
TEST_STATISTICS = TEST_JUDGMENT_COUNT, TEST_MLE_BIAS, TEST_SUM_SQUARED_DEVIATION


class GaussianContributorsTest(unittest.TestCase):
  longMessage = True
  numpy.seterr(all='raise')

  def testGaussianStatistics(self):
    result_judgment_count, result_mle_bias, result_sum_squared_deviation = (
        gaussian_contributors.GaussianStatistics(TEST_DATA))
    test_util.AssertMapsAlmostEqual(self,
                                    TEST_JUDGMENT_COUNT,
                                    result_judgment_count,
                                    label='judgment count, contributor')
    test_util.AssertMapsAlmostEqual(self,
                                    TEST_MLE_BIAS,
                                    result_mle_bias,
                                    label='MLE bias, contributor')
    test_util.AssertMapsAlmostEqual(
        self, TEST_SUM_SQUARED_DEVIATION, result_sum_squared_deviation,
        label='sum of squared deviations, contributor')
    # Check that data without resolutions (or with the
    # wrong resolution format) cause a KeyError:
    self.assertRaises(KeyError, gaussian_contributors.GaussianStatistics,
                      test_util.DS_DATA)
    self.assertRaises(KeyError, gaussian_contributors.GaussianStatistics,
                      test_util.IPEIROTIS_DATA)
    # Check that non-numeric judgments cause a TypeError:
    self.assertRaises(TypeError, gaussian_contributors.GaussianStatistics,
                      {'q1': ([('c1', 'WTF', {})],
                              {gaussian_contributors.MEAN: 5.0,
                               gaussian_contributors.VARIANCE: 0.0})})

  def testMLEGaussianParameters(self):
    result_mle_bias, result_mle_precision = (
        gaussian_contributors.MLEGaussianParameters(TEST_STATISTICS))
    test_util.AssertMapsAlmostEqual(self,
                                    TEST_MLE_BIAS,
                                    result_mle_bias,
                                    label='MLE bias, contributor')
    test_util.AssertMapsAlmostEqual(self,
                                    TEST_MLE_PRECISION,
                                    result_mle_precision,
                                    label='MLE precision, contributor')

  def testSampleGaussianParameters(self):
    numpy.random.seed(0)
    # We'll check that both the mean and the precision for a large sample of
    # parameters are as expected.
    samples = [
        gaussian_contributors.SampleGaussianParameters(TEST_STATISTICS)
        for _ in range(100000)]
    for contributor in TEST_JUDGMENT_COUNT:
      judgment_count = TEST_JUDGMENT_COUNT[contributor]
      sum_squared_deviation = TEST_SUM_SQUARED_DEVIATION[contributor]

      alpha = 0.5 * (judgment_count + 1.0)
      beta = 0.5 * sum_squared_deviation

      expected_bias_mean = TEST_MLE_BIAS[contributor]
      if alpha > 1.0:
        expected_bias_variance = beta / (judgment_count * (alpha - 1.0))
      else:
        expected_bias_variance = 0.0
      if beta > 0.0:
        expected_precision_mean = alpha / beta
        expected_precision_variance = alpha / beta ** 2
      else:
        expected_precision_mean = gaussian_contributors.INFINITY
        expected_precision_variance = numpy.nan

      result_bias_mean = numpy.mean(
          [sample[0][contributor] for sample in samples])
      result_bias_variance = numpy.var(
          [sample[0][contributor] for sample in samples])
      result_precision_mean = numpy.mean(
          [sample[1][contributor] for sample in samples])
      if numpy.isfinite(result_precision_mean):
        result_precision_variance = numpy.var(
            [sample[1][contributor] for sample in samples])
      else:
        result_precision_variance = numpy.nan

      self.assertAlmostEqual(
          expected_bias_mean, result_bias_mean,
          msg='mean bias, contributor "' + str(contributor) + '"',
          places=2)
      self.assertAlmostEqual(
          expected_bias_variance, result_bias_variance,
          msg='variance of bias, contributor "' + str(contributor) + '"',
          places=2)
      self.assertAlmostEqual(
          expected_precision_mean, result_precision_mean,
          msg='mean precision, contributor "' + str(contributor) + '"',
          places=2)
      if numpy.isfinite(expected_precision_variance):
        self.assertAlmostEqual(
            expected_precision_variance, result_precision_variance,
            msg='variance of precision, contributor "' + str(contributor) + '"',
            places=2)

  def testVariationalGaussianParameters(self):
    result_mle_bias, result_variational_precision = (
        gaussian_contributors.VariationalGaussianParameters(TEST_STATISTICS))
    test_util.AssertMapsAlmostEqual(self,
                                    TEST_MLE_BIAS,
                                    result_mle_bias,
                                    label='MLE bias, contributor')
    test_util.AssertMapsAlmostEqual(self,
                                    TEST_VARIATIONAL_PRECISION,
                                    result_variational_precision,
                                    label='variational precision, contributor')

  def testInitializeResolutions(self):
    data = copy.deepcopy(TEST_DATA)
    expected = {'q1': {gaussian_contributors.MEAN: 6.0,
                       gaussian_contributors.VARIANCE: 1.0 / 6.0},
                'q2': {gaussian_contributors.MEAN: 6.0 + 1.0 / 3.0,
                       gaussian_contributors.VARIANCE: 42.0 / 108.0},
                'q3': {gaussian_contributors.MEAN: 6.0 + 2.0 / 3.0,
                       gaussian_contributors.VARIANCE: 114.0 / 108.0},
                'q4': {gaussian_contributors.MEAN: 7.0,
                       gaussian_contributors.VARIANCE: 13.0 / 6.0}}
    gaussian_contributors.GaussianContributors.InitializeResolutions(
        data, overwrite_all_resolutions=True)
    result = model.ExtractResolutions(data)
    test_util.AssertResolutionsAlmostEqual(self, expected, result)
    # Check that non-numeric judgments cause a TypeError:
    self.assertRaises(
        TypeError,
        gaussian_contributors.GaussianContributors.InitializeResolutions,
        {'q1': ([('c1', 'WTF', {})], {})})

  def testSetMLEParameters(self):
    gc = gaussian_contributors.GaussianContributors()
    gc.SetMLEParameters(TEST_DATA)
    # Check that gc was set to something sane:
    contributors = set(['c1', 'c2', 'c3'])
    self.assertSetEqual(contributors, set(gc.bias.keys()))
    self.assertSetEqual(contributors, set(gc.precision.keys()))

  def testSetSampleParameters(self):
    gc = gaussian_contributors.GaussianContributors()
    gc.SetSampleParameters(TEST_DATA)
    # Check that gc was set to something sane:
    contributors = set(['c1', 'c2', 'c3'])
    self.assertSetEqual(contributors, set(gc.bias.keys()))
    self.assertSetEqual(contributors, set(gc.precision.keys()))

  def testSetVariationalParameters(self):
    gc = gaussian_contributors.GaussianContributors()
    gc.SetVariationalParameters(TEST_DATA)
    # Check that gc was set to something sane:
    contributors = set(['c1', 'c2', 'c3'])
    self.assertSetEqual(contributors, set(gc.bias.keys()))
    self.assertSetEqual(contributors, set(gc.precision.keys()))

  def testResolveQuestion(self):
    gc = gaussian_contributors.GaussianContributors()
    gc.bias = {'worker1': -0.5,
               'worker2': 0.8}
    gc.precision = {'worker1': 1.0 / 0.3,
                    'worker2': 1.0 / 0.2}
    expected = {gaussian_contributors.MEAN: 2.32,
                gaussian_contributors.VARIANCE: 0.12}
    resolution_map = gc.ResolveQuestion([('worker1', 2.0, {}),
                                         ('worker2', 3.0, {})])
    test_util.AssertMapsAlmostEqual(self,
                                    expected, resolution_map, label='variable')
    # Now try with infinite-precision contributors:
    gc.bias = {'worker1': 1.0,
               'worker2': 2.0,
               'worker3': 3.0}
    gc.precision = {'worker1': 1.0,
                    'worker2': gaussian_contributors.INFINITY,
                    'worker3': gaussian_contributors.INFINITY}
    expected = {gaussian_contributors.MEAN: -2.5,
                gaussian_contributors.VARIANCE: 0.25}
    resolution_map = gc.ResolveQuestion([('worker1', 0.0, {}),
                                         ('worker2', 0.0, {}),
                                         ('worker3', 0.0, {})])
    test_util.AssertMapsAlmostEqual(self,
                                    expected, resolution_map, label='variable')
    # Check that non-numeric judgments cause a TypeError:
    self.assertRaises(TypeError, gc.ResolveQuestion, [('worker1', 'WTF', {})])


if __name__ == '__main__':
  unittest.main()
