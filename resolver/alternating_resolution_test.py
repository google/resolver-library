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

"""Tests for alternating_resolution.py."""

__author__ = 'tpw@google.com (Tamsyn Waterhouse)'

import unittest
from resolver import alternating_resolution
from resolver import model
from resolver import test_util


class MockModel(model.StatisticalModel):
  def __init__(self):
    self.dummy_parameter = 0.0

  def ChangeParameters(self, unused_data):
    self.dummy_parameter += 1.0

  def ResolveQuestion(self, unused_question):
    # Set a mock resolution that starts at {"A": 1.0, "B": 0.0} and converges
    # harmonically to {"A": 0.0, "B": 1.0} as this method is called repeatedly:
    resolution_map = {}
    resolution_map['A'] = 1.0 / self.dummy_parameter
    resolution_map['B'] = 1.0 - 1.0 / self.dummy_parameter
    return resolution_map


class MockResolution(alternating_resolution.AlternatingResolution):

  @staticmethod
  def UpdateModel(unused_data, mock_model):
    mock_model.ChangeParameters(None)


class AlternatingResolutionTest(unittest.TestCase):
  longMessage = True

  def testIterateOnce(self):
    data = {1: ([], {})}
    mock_model = MockModel()
    resolution = MockResolution()
    resolution.IterateOnce(data, mock_model)
    # resolution should have called ChangeParameters once and ResolveQuestion
    # once:
    self.assertEqual(1.0, mock_model.dummy_parameter)
    test_util.AssertResolutionsAlmostEqual(
        self,
        {1: {'A': 1.0, 'B': 0.0}},
        mock_model.ExtractResolutions(data))

  def testIterateUntilConvergence(self):
    data = {1: ([], {})}
    mock_model = MockModel()
    resolution = MockResolution()
    # Our silly model should take far more than MAX_ITERATIONS to converge:
    self.assertFalse(resolution.IterateUntilConvergence(data, mock_model))
    # resolution should have called IterateOnce exactly MAX_ITERATIONS times:
    expected_parameter = float(alternating_resolution.MAX_ITERATIONS)
    self.assertEqual(expected_parameter, mock_model.dummy_parameter)
    test_util.AssertResolutionsAlmostEqual(
        self,
        {1: {'A': 1.0 / expected_parameter,
             'B': 1.0 - 1.0 / expected_parameter}},
        mock_model.ExtractResolutions(data))
    # Now we'll force convergence by setting mock_model.dummy_parameter very
    # high...
    mock_model.dummy_parameter = 10.0 / alternating_resolution.EPSILON
    # and check that resolution understands that it has converged:
    self.assertTrue(resolution.IterateUntilConvergence(data, mock_model))


if __name__ == '__main__':
  unittest.main()
