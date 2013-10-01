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

"""Integration test for confusion_matrices with expectation-maximization."""

__author__ = 'tpw@google.com (Tamsyn Waterhouse)'

import unittest
from resolver import alternating_resolution
from resolver import confusion_matrices
from resolver import test_util


class EmConfusionTest(unittest.TestCase):
  longMessage = True

  def testIterateUntilConvergence(self):
    # Initialize a confusion matrices model and an EM object:
    cm = confusion_matrices.ConfusionMatrices()
    maximizer = alternating_resolution.ExpectationMaximization()

    # First with the Ipeirotis example:
    data = test_util.IPEIROTIS_DATA
    cm.InitializeResolutions(data)
    self.assertTrue(maximizer.IterateUntilConvergence(data, cm))
    expected = cm.ExtractResolutions(test_util.IPEIROTIS_DATA_FINAL)
    result = cm.ExtractResolutions(data)
    test_util.AssertResolutionsAlmostEqual(self, expected, result)

    # Now with the Dawid & Skene example:
    data = test_util.DS_DATA
    cm.InitializeResolutions(data)
    # The algorithm takes more than 10 steps to converge, so we expect
    # IterateUntilConvergence to return False:
    self.assertFalse(maximizer.IterateUntilConvergence(data, cm,
                                                       max_iterations=10))
    # Nevertheless, its results are accurate to 3 places:
    expected = cm.ExtractResolutions(test_util.DS_DATA_FINAL)
    result = cm.ExtractResolutions(data)
    test_util.AssertResolutionsAlmostEqual(self, expected, result)


if __name__ == '__main__':
  unittest.main()
