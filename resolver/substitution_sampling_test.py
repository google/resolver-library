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

"""Tests for substitution_sampling.py."""

__author__ = 'tpw@google.com (Tamsyn Waterhouse)'

import numpy

import unittest
from resolver import model
from resolver import substitution_sampling
from resolver import test_util


MOCK_RESOLUTION = {'A': 0.7, 'B': 0.2}  # and 0.1 probability of something else
NUMBER_OF_SAMPLES = 10000


class MockModel(model.StatisticalModel):

  def __init__(self):
    self.times_called = 0

  def SetSampleParameters(self, unused_data):
    self.times_called += 1

  def ResolveQuestion(self, unused_question):
    # Return an arbitrary resolution probability matrix:
    return MOCK_RESOLUTION


class SubstitutionSamplingTest(unittest.TestCase):
  longMessage = True
  numpy.seterr(all='raise')

  def testDrawOneSample(self):
    # Set up mock data, a mock model object and a SubstitutionSampling object:
    resolution = {}
    data = {1: ([], resolution)}
    mock_model = MockModel()
    ss = substitution_sampling.SubstitutionSampling()

    # Call DrawOneSample a few times and check that it produces sensible
    # resolutions:
    for _ in range(100):
      ss.DrawOneSample(data, mock_model)
      self.assertTrue('A' in resolution or 'B' in resolution or not resolution)

  def testIntegrate(self):
    # Seed the random number generator to produce deterministic test results:
    numpy.random.seed(0)

    # Set up mock data, a mock model object, and a SubstitutionSampling object:
    data = {1: ([], {}),
            2: ([], {'C': 1.0})}
    mock_model = MockModel()
    ss = substitution_sampling.SubstitutionSampling()

    # Call ss.Integrate and check that the result is close to the mock model's
    # resolution:
    ss.Integrate(data, mock_model, golden_questions=[2],
                 number_of_samples=NUMBER_OF_SAMPLES)
    test_util.AssertResolutionsAlmostEqual(self,
                                           {1: MOCK_RESOLUTION, 2: {'C': 1.0}},
                                           mock_model.ExtractResolutions(data),
                                           places=2)
    # ss should have called SetSampleParameters NUMBER_OF_SAMPLES times:
    self.assertEqual(NUMBER_OF_SAMPLES, mock_model.times_called)


if __name__ == '__main__':
  unittest.main()
