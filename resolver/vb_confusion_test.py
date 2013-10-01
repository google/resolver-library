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

"""Integration test for confusion_matrices with VariationalBayes."""

__author__ = 'tpw@google.com (Tamsyn Waterhouse)'

import unittest
from resolver import alternating_resolution
from resolver import confusion_matrices
from resolver import test_util


# These results were computed numerically:
DS_VB_CM_RESOLUTIONS = {1: {1: 1.0},
                        2: {3: 1.0},
                        3: {1: 0.721, 2: 0.279},
                        4: {1: 0.002, 2: 0.998},
                        5: {2: 1.0},
                        6: {2: 1.0},
                        7: {1: 0.990, 2: 0.010},
                        8: {3: 1.0},
                        9: {2: 1.0},
                        10: {2: 1.0},
                        11: {3: 0.003, 4: 0.997},
                        12: {2: 1.0},
                        13: {1: 1.0},
                        14: {2: 1.0},
                        15: {1: 1.0},
                        16: {1: 1.0},
                        17: {1: 1.0},
                        18: {1: 1.0},
                        19: {2: 1.0},
                        20: {2: 1.0},
                        21: {2: 1.0},
                        22: {2: 1.0},
                        23: {2: 1.0},
                        24: {2: 1.0},
                        25: {1: 1.0},
                        26: {1: 1.0},
                        27: {2: 1.0},
                        28: {1: 1.0},
                        29: {1: 1.0},
                        30: {1: 1.0},
                        31: {1: 1.0},
                        32: {2: 0.001, 3: 0.999},
                        33: {1: 1.0},
                        34: {2: 1.0},
                        35: {2: 1.0},
                        36: {3: 0.994, 4: 0.006},
                        37: {2: 1.0},
                        38: {2: 0.991, 3: 0.009},
                        39: {3: 1.0},
                        40: {1: 1.0},
                        41: {1: 1.0},
                        42: {1: 1.0},
                        43: {2: 1.0},
                        44: {1: 1.0},
                        45: {2: 1.0}}


class VariationalBayesConfusionTest(unittest.TestCase):
  longMessage = True

  def testIterateUntilConvergence(self):
    # Initialize a confusion matrices model and a VB object:
    cm = confusion_matrices.ConfusionMatrices()
    vb = alternating_resolution.VariationalBayes()

# TODO(tpw):  Once we have a way to mark input resolutions as 'golden', enable
#             this test (which requires golden data).
#   # First test with the Ipeirotis example:
#   data = test_util.IPEIROTIS_DATA
#   cm.InitializeResolutions(data)
#   self.assertTrue(vb.IterateUntilConvergence(data, cm))
#   result = cm.ExtractResolutions(data)
#   test_util.AssertResolutionsAlmostEqual(self,
#                                          IPEIROTIS_VB_CM_RESOLUTIONS, result)

    # Now test with the Dawid & Skene example:
    data = test_util.DS_DATA
    cm.InitializeResolutions(data)
    # VB is a little slower than EM, so we'll give it algorithm up to 50
    # iterations to converge:
    self.assertTrue(vb.IterateUntilConvergence(data, cm, max_iterations=50))
    result = cm.ExtractResolutions(data)
    test_util.AssertResolutionsAlmostEqual(self, DS_VB_CM_RESOLUTIONS, result)


if __name__ == '__main__':
  unittest.main()
