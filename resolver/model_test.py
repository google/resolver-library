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



import collections
import copy
import random

import numpy

import unittest
from resolver import model
from resolver import test_util


class ModelTest(unittest.TestCase):
  longMessage = True
  numpy.seterr(all='raise')

  def testInitializeResolutions(self):
    # Test that the method behaves as expected, setting initial guesses:
    data = copy.deepcopy(test_util.DS_DATA)
    expected = {1: {1: 1.0},
                2: {3: 5.0 / 7.0, 4: 2.0 / 7.0},
                3: {1: 3.0 / 7.0, 2: 4.0 / 7.0},
                4: {1: 2.0 / 7.0, 2: 4.0 / 7.0, 3: 1.0 / 7.0},
                5: {2: 6.0 / 7.0, 3: 1.0 / 7.0},
                6: {2: 5.0 / 7.0, 3: 2.0 / 7.0},
                7: {1: 4.0 / 7.0, 2: 3.0 / 7.0},
                8: {3: 6.0 / 7.0, 4: 1.0 / 7.0},
                9: {2: 6.0 / 7.0, 3: 1.0 / 7.0},
                10: {2: 5.0 / 7.0, 3: 2.0 / 7.0},
                11: {4: 1.0},
                12: {2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 1.0 / 7.0},
                13: {1: 1.0},
                14: {1: 1.0 / 7.0, 2: 5.0 / 7.0, 3: 1.0 / 7.0},
                15: {1: 6.0 / 7.0, 2: 1.0 / 7.0},
                16: {1: 6.0 / 7.0, 2: 1.0 / 7.0},
                17: {1: 1.0},
                18: {1: 1.0},
                19: {1: 1.0 / 7.0, 2: 6.0 / 7.0},
                20: {1: 1.0 / 7.0, 2: 5.0 / 7.0, 3: 1.0 / 7.0},
                21: {2: 1.0},
                22: {1: 1.0 / 7.0, 2: 6.0 / 7.0},
                23: {2: 6.0 / 7.0, 3: 1.0 / 7.0},
                24: {1: 1.0 / 7.0, 2: 6.0 / 7.0},
                25: {1: 1.0},
                26: {1: 1.0},
                27: {2: 6.0 / 7.0, 3: 1.0 / 7.0},
                28: {1: 1.0},
                29: {1: 1.0},
                30: {1: 5.0 / 7.0, 2: 2.0 / 7.0},
                31: {1: 1.0},
                32: {2: 1.0 / 7.0, 3: 6.0 / 7.0},
                33: {1: 1.0},
                34: {2: 1.0},
                35: {2: 5.0 / 7.0, 3: 2.0 / 7.0},
                36: {3: 4.0 / 7.0, 4: 3.0 / 7.0},
                37: {1: 1.0 / 7.0, 2: 5.0 / 7.0, 3: 1.0 / 7.0},
                38: {2: 3.0 / 7.0, 3: 4.0 / 7.0},
                39: {2: 1.0 / 7.0, 3: 5.0 / 7.0, 4: 1.0 / 7.0},
                40: {1: 1.0},
                41: {1: 1.0},
                42: {1: 5.0 / 7.0, 2: 2.0 / 7.0},
                43: {2: 6.0 / 7.0, 3: 1.0 / 7.0},
                44: {1: 6.0 / 7.0, 2: 1.0 / 7.0},
                45: {2: 1.0}}
    model.StatisticalModel.InitializeResolutions(data)
    results = model.ExtractResolutions(data)
    test_util.AssertResolutionsAlmostEqual(self, expected, results)
    # Now write garbage resolutions and check that the constructor overwrites
    # them:
    for _, resolution_map in data.itervalues():
      resolution_map['None of the above'] = 1.0
    model.StatisticalModel.InitializeResolutions(data,
                                                 overwrite_all_resolutions=True)
    results = model.ExtractResolutions(data)
    test_util.AssertResolutionsAlmostEqual(self, expected, results)
    # Finally, test that the judgments_to_answers mapping works:
    judgments_to_answers = {1: 'Good', 2: None, 3: 'Bad', 4: 'Bad'}
    model.StatisticalModel.InitializeResolutions(
        data,
        overwrite_all_resolutions=True,
        judgments_to_answers=judgments_to_answers)
    expected = {1: {'Good': 1.0},
                2: {'Bad': 1.0},
                3: {'Good': 1.0},
                4: {'Good': 2.0 / 3.0, 'Bad': 1.0 / 3.0},
                5: {'Bad': 1.0},
                6: {'Bad': 1.0},
                7: {'Good': 1.0},
                8: {'Bad': 1.0},
                9: {'Bad': 1.0},
                10: {'Bad': 1.0},
                11: {'Bad': 1.0},
                12: {'Bad': 1.0},
                13: {'Good': 1.0},
                14: {'Good': 1.0 / 2.0, 'Bad': 1.0 / 2.0},
                15: {'Good': 1.0},
                16: {'Good': 1.0},
                17: {'Good': 1.0},
                18: {'Good': 1.0},
                19: {'Good': 1.0},
                20: {'Good': 1.0 / 2.0, 'Bad': 1.0 / 2.0},
                21: {},
                22: {'Good': 1.0},
                23: {'Bad': 1.0},
                24: {'Good': 1.0},
                25: {'Good': 1.0},
                26: {'Good': 1.0},
                27: {'Bad': 1.0},
                28: {'Good': 1.0},
                29: {'Good': 1.0},
                30: {'Good': 1.0},
                31: {'Good': 1.0},
                32: {'Bad': 1.0},
                33: {'Good': 1.0},
                34: {},
                35: {'Bad': 1.0},
                36: {'Bad': 1.0},
                37: {'Good': 1.0 / 2.0, 'Bad': 1.0 / 2.0},
                38: {'Bad': 1.0},
                39: {'Bad': 1.0},
                40: {'Good': 1.0},
                41: {'Good': 1.0},
                42: {'Good': 1.0},
                43: {'Bad': 1.0},
                44: {'Good': 1.0},
                45: {}}
    results = model.ExtractResolutions(data)
    test_util.AssertResolutionsAlmostEqual(self, expected, results)

  def testRandomAnswer(self):
    # To keep this test deterministic, we fix NumPy's random seed and used
    # OrderedDicts below.
    numpy.random.seed(1)
    # Check that RandomAnswer can return both answers from a normalized
    # resolution map containing two answers:
    resolution_map = collections.OrderedDict(sorted([('A', 0.6), ('B', 0.4)]))
    results = set([model.StatisticalModel.RandomAnswer(resolution_map)
                   for _ in range(10)])
    self.assertEqual(set(['A', 'B']), results)
    # Check that RandomAnswer can return None with a non-normalized resolution
    # map:
    resolution_map = collections.OrderedDict(sorted([('A', 0.5), ('B', 0.2)]))
    results = set([model.StatisticalModel.RandomAnswer(resolution_map)
                   for _ in range(10)])
    self.assertEqual(set(['A', 'B', None]), results)

  def testMostLikelyResolution(self):
    # Check that in the event of a tied resolution, MostLikelyResolution is
    # capable of returning one or the other answers (breaking ties randomly):
    random.seed(1)
    resolution_map = {'A': 0.4, 'B': 0.2, 'C': 0.4}
    results = set([model.StatisticalModel.MostLikelyResolution(resolution_map)
                   for _ in range(2)])
    self.assertEqual(set(['A', 'C']), results)
    # Check that with an empty resolution, MostLikelyResolution returns None:
    self.assertEqual(None, model.StatisticalModel.MostLikelyResolution({}))

  def testMostLikelyResolutions(self):
    # Use data from the solution to the Dawid & Skene example:
    result = model.StatisticalModel.MostLikelyResolutions(
        test_util.DS_DATA_FINAL)
    expected = {1: 1, 2: 4, 3: 2, 4: 2, 5: 2,
                6: 2, 7: 1, 8: 3, 9: 2, 10: 2,
                11: 4, 12: 3, 13: 1, 14: 2, 15: 1,
                16: 1, 17: 1, 18: 1, 19: 2, 20: 2,
                21: 2, 22: 2, 23: 2, 24: 2, 25: 1,
                26: 1, 27: 2, 28: 1, 29: 1, 30: 1,
                31: 1, 32: 3, 33: 1, 34: 2, 35: 2,
                36: 4, 37: 2, 38: 3, 39: 3, 40: 1,
                41: 1, 42: 1, 43: 2, 44: 1, 45: 2}
    test_util.AssertMapsAlmostEqual(self, expected, result)

  def testResolutionDistanceSquared(self):
    # Test that ResolutionDistanceSquared returns the correct element-wise
    # distance between two resolutions:
    # (Note that it's fine for these resolutions to be non-normalized.)
    x = {'A': 1.0, 'B': 1.0, 'C': 1.0}
    y = {'A': 2.0, 'B': 4.0, 'D': 1.0}
    self.assertAlmostEqual(
        12.0, model.StatisticalModel.ResolutionDistanceSquared(x, y))

  def testNotImplementedMethods(self):
    base_model = model.StatisticalModel()
    self.assertRaises(NotImplementedError, base_model.SetMLEParameters, None)
    self.assertRaises(NotImplementedError, base_model.SetSampleParameters, None)
    self.assertRaises(NotImplementedError, base_model.QuestionEntropy, None)
    self.assertRaises(NotImplementedError, base_model.MutualInformation, None)
    self.assertRaises(NotImplementedError,
                      base_model.PointwiseMutualInformation, None)
    self.assertRaises(NotImplementedError, base_model.ResolveQuestion, None)


if __name__ == '__main__':
  unittest.main()
