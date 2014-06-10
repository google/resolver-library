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

"""Integration test for probability_correct with variational inference."""

__author__ = 'tpw@google.com (Tamsyn Waterhouse)'

import unittest
from resolver import alternating_resolution
from resolver import probability_correct
from resolver import test_util


# Resolutions computed using VB plus the probability-correct model:
IPEIROTIS_VB_PC_RESOLUTIONS = {'url1': {'notporn': 1.0},
                               'url2': {'porn': 1.0},
                               'url3': {'notporn': 1.0},
                               'url4': {'porn': 0.999, 'notporn': 0.001},
                               'url5': {'porn': 0.002, 'notporn': 0.998}}

DS_VB_PC_RESOLUTIONS = {1: {1: 1.000},
                        2: {3: 1.000},
                        3: {1: 0.142, 2: 0.858},
                        4: {1: 0.001, 2: 0.999},
                        5: {2: 1.000},
                        6: {2: 1.000},
                        7: {1: 0.944, 2: 0.056},
                        8: {3: 1.000},
                        9: {2: 1.000},
                        10: {2: 1.000},
                        11: {4: 1.000},
                        12: {2: 0.903, 3: 0.096, 4: 0.002},
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
                        36: {3: 0.964, 4: 0.036},
                        37: {2: 1.000},
                        38: {2: 0.142, 3: 0.858},
                        39: {3: 1.000},
                        40: {1: 1.000},
                        41: {1: 1.000},
                        42: {1: 1.000},
                        43: {2: 1.000},
                        44: {1: 1.000},
                        45: {2: 1.000}}


class VbProbabilityCorrectTest(unittest.TestCase):
  longMessage = True

  def testIterateUntilConvergence(self):
    # Test first with the Ipeirotis example (expecting a resolution, above,
    # differing slightly from the expectation-maximization case):
    data = test_util.IPEIROTIS_DATA
    pc = probability_correct.ProbabilityCorrect()
    pc.InitializeResolutions(data)
    maximizer = alternating_resolution.VariationalBayes()
    # Run the variational inference algorithm:
    self.assertTrue(maximizer.IterateUntilConvergence(
        data, pc, golden_questions=['url1', 'url2']))
    # Check the results:
    result = pc.ExtractResolutions(data)
    test_util.AssertResolutionsAlmostEqual(self,
                                           IPEIROTIS_VB_PC_RESOLUTIONS, result)
    self.assertEqual(2, pc._answer_space_size)

    # Run the same experiment without gold data, and check that having gold
    # data gives us more faith in the contributors who agree with it:
    gold_contributor_accuracy = pc.probability_correct.copy()
    self.assertTrue(maximizer.IterateUntilConvergence(data, pc))
    non_gold_contributor_accuracy = pc.probability_correct.copy()
    self.assertGreater(gold_contributor_accuracy['worker2'],
                       non_gold_contributor_accuracy['worker2'])
    self.assertGreater(gold_contributor_accuracy['worker3'],
                       non_gold_contributor_accuracy['worker3'])
    self.assertGreater(gold_contributor_accuracy['worker4'],
                       non_gold_contributor_accuracy['worker4'])

    # Test with the Dawid & Skene example (expecting a resolution, above,
    # differing slightly from the expectation-maximization case):
    data = test_util.DS_DATA
    pc = probability_correct.ProbabilityCorrect()
    pc.InitializeResolutions(data)
    maximizer = alternating_resolution.VariationalBayes()
    # Run the variational inference algorithm:
    self.assertTrue(maximizer.IterateUntilConvergence(data, pc))
    # Check the results:
    result = pc.ExtractResolutions(data)
    test_util.AssertResolutionsAlmostEqual(self, DS_VB_PC_RESOLUTIONS, result)
    self.assertEqual(4, pc._answer_space_size)


if __name__ == '__main__':
  unittest.main()
