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

"""Integration test for probability_correct with expectation-maximization."""

__author__ = 'tpw@google.com (Tamsyn Waterhouse)'

import unittest
from resolver import alternating_resolution
from resolver import probability_correct
from resolver import test_util


# Estimated resolutions using EM plus the probability-correct model:
DS_EM_PC_RESOLUTIONS = {1: {1: 1.000},
                        2: {3: 1.000},
                        3: {1: 0.133, 2: 0.867},
                        4: {1: 0.001, 2: 0.999},
                        5: {2: 1.000},
                        6: {2: 1.000},
                        7: {1: 0.950, 2: 0.050},
                        8: {3: 1.000},
                        9: {2: 1.000},
                        10: {2: 1.000},
                        11: {4: 1.000},
                        12: {2: 0.899, 3: 0.099, 4: 0.002},
                        13: {1: 1.000},
                        14: {2: 1.000},
                        15: {1: 1.000},
                        16: {1: 1.000},
                        17: {1: 1.000},
                        18: {1: 1.000},
                        19: {2: 1.000},
                        20: {2: 1.000},
                        21: {2: 1.000},
                        22: {2: 1.000},
                        23: {2: 1.000},
                        24: {2: 1.000},
                        25: {1: 1.000},
                        26: {1: 1.000},
                        27: {2: 1.000},
                        28: {1: 1.000},
                        29: {1: 1.000},
                        30: {1: 0.999, 2: 0.001},
                        31: {1: 1.000},
                        32: {3: 1.000},
                        33: {1: 1.000},
                        34: {2: 1.000},
                        35: {2: 1.000},
                        36: {3: 0.965, 4: 0.035},
                        37: {2: 1.000},
                        38: {2: 0.133, 3: 0.867},
                        39: {3: 1.000},
                        40: {1: 1.000},
                        41: {1: 1.000},
                        42: {1: 1.000},
                        43: {2: 1.000},
                        44: {1: 1.000},
                        45: {2: 1.000}}


class EmProbabilityCorrectTest(unittest.TestCase):
  longMessage = True

  def testIterateUntilConvergence(self):
    maximizer = alternating_resolution.ExpectationMaximization()

    # First test with the Ipeirotis example (expecting the same resolution as
    # we get with confusion matrices):
    data = test_util.IPEIROTIS_DATA
    pc = probability_correct.ProbabilityCorrect()
    pc.InitializeResolutions(data)
    self.assertTrue(maximizer.IterateUntilConvergence(data, pc))
    expected = pc.ExtractResolutions(test_util.IPEIROTIS_DATA_FINAL)
    result = pc.ExtractResolutions(data)
    test_util.AssertResolutionsAlmostEqual(self, expected, result)
    self.assertEqual(2, pc._answer_space_size)

    # Now with the Dawid & Skene example (with a resolution, above, differing
    # slightly from the confusion matrix case):
    data = test_util.DS_DATA
    pc.UnsetAnswerSpaceSize()  # to reset the model
    pc.InitializeResolutions(data)
    # The algorithm takes more than 10 steps to converge, so we expect
    # IterateUntilConvergence to return False:
    self.assertFalse(maximizer.IterateUntilConvergence(data, pc,
                                                       max_iterations=10))
    # Nevertheless, its results are accurate to 3 places:
    result = pc.ExtractResolutions(data)
    test_util.AssertResolutionsAlmostEqual(self, DS_EM_PC_RESOLUTIONS, result)
    self.assertEqual(4, pc._answer_space_size)


if __name__ == '__main__':
  unittest.main()
