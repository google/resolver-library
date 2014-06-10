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

"""Integration test for substitution_sampling.py and confusion_matrices.py."""

__author__ = 'tpw@google.com (Tamsyn Waterhouse)'


import numpy

import unittest
from resolver import confusion_matrices
from resolver import substitution_sampling
from resolver import test_util


# Numerically-estimated resolution probabilities for the Ipeirotis exmaple:
IPEIROTIS_SS_CM_RESOLUTIONS = {'url1': {'notporn': 1.0},  # golden answer
                               'url2': {'porn': 1.0},  # golden answer
                               'url3': {'notporn': 0.95, 'porn': 0.05},
                               'url4': {'notporn': 0.05, 'porn': 0.95},
                               'url5': {'notporn': 0.9, 'porn': 0.1}}

# Convergence for the Dawid and Skene example is slow because of the small
# amount of data for categories 3 and 4.  Therefore we'll use only coarse
# estimates for this test.
DS_SS_CM_RESOLUTIONS = {1: {1: 1.0},
                        2: {3: 1.0},
                        3: {1: 0.550, 2: 0.450},
                        4: {1: 0.010, 2: 0.990},
                        5: {2: 1.0},
                        6: {2: 1.0},
                        7: {1: 0.958, 2: 0.042},
                        8: {2: 0.001, 3: 0.999},
                        9: {2: 1.0},
                        10: {2: 0.999, 3: 0.001},
                        11: {3: 0.999, 4: 0.001},
                        12: {2: 0.978, 3: 0.022},
                        13: {1: 1.0},
                        14: {2: 1.0},
                        15: {1: 1.0},
                        16: {1: 1.0},
                        17: {1: 1.0},
                        18: {1: 1.0},
                        19: {2: 1.0},
                        20: {1: 0.001, 2: 0.999},
                        21: {2: 1.0},
                        22: {2: 1.0},
                        23: {2: 1.0},
                        24: {2: 1.0},
                        25: {1: 1.0},
                        26: {1: 1.0},
                        27: {2: 1.0},
                        28: {1: 1.0},
                        29: {1: 1.0},
                        30: {1: 0.998, 2: 0.002},
                        31: {1: 1.0},
                        32: {2: 0.013, 3: 0.987},
                        33: {1: 1.0},
                        34: {2: 1.0},
                        35: {2: 1.0},
                        36: {3: 1.0},
                        37: {2: 1.0},
                        38: {2: 0.940, 3: 0.060},
                        39: {2: 0.003, 3: 0.997},
                        40: {1: 1.0},
                        41: {1: 1.0},
                        42: {1: 0.999, 2: 0.001},
                        43: {2: 1.0},
                        44: {1: 1.0},
                        45: {2: 1.0}}


class SubstitutionConfusionTest(unittest.TestCase):
  longMessage = True
  numpy.seterr(all='raise')

  def testIntegrate(self):
    # Seed the random number generator to produce deterministic test results:
    numpy.random.seed(0)
    # TODO(tpw):  This is necessary but not sufficient.  Python's arbitrary
    #             ordering of dict iteration makes some deeper calls
    #             non-deterministic, and thus this test may exhibit flakiness.

    # Initialize a confusion matrices model:
    cm = confusion_matrices.ConfusionMatrices()

    # First check the estimated answers for the Ipeirotis example, using the
    # EM algorithm's results as a starting point for the sampling chain:
    data = test_util.IPEIROTIS_DATA_FINAL
    sampler = substitution_sampling.SubstitutionSampling()
    sampler.Integrate(data, cm, golden_questions=['url1', 'url2'],
                      number_of_samples=20000)
    result = cm.ExtractResolutions(data)
    test_util.AssertResolutionsAlmostEqual(self,
                                           IPEIROTIS_SS_CM_RESOLUTIONS, result,
                                           places=1)

    # Now check the estimated answers for the Dawid & Skene example, again using
    # the EM algorithm's results as a starting point for the sampling chain:
    numpy.random.seed(0)
    data = test_util.DS_DATA_FINAL
    sampler = substitution_sampling.SubstitutionSampling()
    sampler.Integrate(data, cm, number_of_samples=20000)
    result = cm.ExtractResolutions(data)
    test_util.AssertResolutionsAlmostEqual(self, DS_SS_CM_RESOLUTIONS, result,
                                           places=2)


if __name__ == '__main__':
  unittest.main()
