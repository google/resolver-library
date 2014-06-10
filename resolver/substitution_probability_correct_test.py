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

"""Integration test combining substitution_sampling and probability_correct."""

__author__ = 'tpw@google.com (Tamsyn Waterhouse)'

import numpy

import unittest
from resolver import probability_correct
from resolver import substitution_sampling
from resolver import test_util


# Numerically-estimated resolution probabilities for the Ipeirotis exmaple:
IPEIROTIS_SS_PC_RESOLUTIONS = {'url1': {'notporn': 1.0},  # golden answer
                               'url2': {'porn': 1.0},  # golden answer
                               'url3': {'notporn': 0.99, 'porn': 0.01},
                               'url4': {'notporn': 0.01, 'porn': 0.99},
                               'url5': {'notporn': 0.95, 'porn': 0.05}}

# Estimated resolutions using substitution sampling with the
# probability-correct model:
DS_SS_PC_RESOLUTIONS = {1: {1: 1.000},
                        2: {3: 1.000},
                        3: {1: 0.174, 2: 0.826},
                        4: {1: 0.002, 2: 0.998},
                        5: {2: 1.000},
                        6: {2: 1.000},
                        7: {1: 0.927, 2: 0.073},
                        8: {3: 1.000},
                        9: {2: 1.000},
                        10: {2: 0.999, 3: 0.001},
                        11: {4: 1.000},
                        12: {2: 0.863, 3: 0.135, 4: 0.002},
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
                        36: {3: 0.954, 4: 0.046},
                        37: {2: 1.000},
                        38: {2: 0.178, 3: 0.822},
                        39: {3: 1.000},
                        40: {1: 1.000},
                        41: {1: 1.000},
                        42: {1: 1.000},
                        43: {2: 1.000},
                        44: {1: 1.000},
                        45: {2: 1.000}}


class SubstitutionProbabilityCorrectTest(unittest.TestCase):
  longMessage = True
  numpy.seterr(all='raise')

  def testIntegrate(self):
    # Seed the random number generator to produce deterministic test results:
    numpy.random.seed(0)
    # TODO(tpw):  This is necessary but not sufficient.  Python's arbitrary
    #             ordering of dict iteration makes some deeper calls
    #             non-deterministic, and thus this test may exhibit flakiness.

    # Initialize a probability-correct model:
    pc = probability_correct.ProbabilityCorrect()

    # First check the estimated answers for the Ipeirotis example, using the
    # EM algorithm's results as a starting point for the sampling chain:
    data = test_util.IPEIROTIS_DATA_FINAL
    sampler = substitution_sampling.SubstitutionSampling()
    sampler.Integrate(data, pc, golden_questions=['url1', 'url2'],
                      number_of_samples=20000)
    result = pc.ExtractResolutions(data)
    test_util.AssertResolutionsAlmostEqual(self,
                                           IPEIROTIS_SS_PC_RESOLUTIONS, result,
                                           places=1)
    # Reset the model:
    pc.UnsetAnswerSpaceSize()

    # Now check the estimated answers for the Dawid & Skene example, again using
    # the EM algorithm's results as a starting point for the sampling chain:
    data = test_util.DS_DATA_FINAL
    sampler = substitution_sampling.SubstitutionSampling()
    sampler.Integrate(data, pc, number_of_samples=20000)
    result = pc.ExtractResolutions(data)
    test_util.AssertResolutionsAlmostEqual(self, DS_SS_PC_RESOLUTIONS, result,
                                           places=2)


if __name__ == '__main__':
  unittest.main()
