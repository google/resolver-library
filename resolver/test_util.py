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

"""Data and helper methods for the unittests in this module."""

__author__ = 'tpw@google.com (Tamsyn Waterhouse)'


###############################################
# Example from Ipeirotis's qmturk.appspot.com #
###############################################

# Judgments:
IPEIROTIS_RESPONSES = [
    [('worker1', 'porn', {}),
     ('worker2', 'notporn', {}),
     ('worker3', 'notporn', {}),
     ('worker4', 'notporn', {}),
     ('worker5', 'porn', {})],
    [('worker1', 'porn', {}),
     ('worker2', 'porn', {}),
     ('worker3', 'porn', {}),
     ('worker4', 'porn', {}),
     ('worker5', 'notporn', {})],
    [('worker1', 'porn', {}),
     ('worker2', 'notporn', {}),
     ('worker3', 'notporn', {}),
     ('worker4', 'notporn', {}),
     ('worker5', 'porn', {})],
    [('worker1', 'porn', {}),
     ('worker2', 'porn', {}),
     ('worker3', 'porn', {}),
     ('worker4', 'porn', {}),
     ('worker5', 'notporn', {})],
    [('worker1', 'porn', {}),
     ('worker2', 'porn', {}),
     ('worker3', 'notporn', {}),
     ('worker4', 'notporn', {}),
     ('worker5', 'porn', {})]]

# The golden answers known a priori:
IPEIROTIS_GOLDEN_ANSWERS = [{'notporn': 1.0},
                            {'porn': 1.0},
                            {},
                            {},
                            {}]

# All the answers:
IPEIROTIS_ALL_ANSWERS = [{'notporn': 1.0},
                         {'porn': 1.0},
                         {'notporn': 1.0},
                         {'porn': 1.0},
                         {'notporn': 1.0}]

# Data object with only golden answers:
IPEIROTIS_DATA = dict(zip(['url1', 'url2', 'url3', 'url4', 'url5'],
                          zip(IPEIROTIS_RESPONSES, IPEIROTIS_GOLDEN_ANSWERS)))

# Data object with all the answers:
IPEIROTIS_DATA_FINAL = dict(zip(['url1', 'url2', 'url3', 'url4', 'url5'],
                                zip(IPEIROTIS_RESPONSES,
                                    IPEIROTIS_ALL_ANSWERS)))


#####################################
# Example from Dawid & Skene (1979) #
#####################################

# Judgments:
DS_RESPONSES = [
    [(1, 1, {}), (1, 1, {}), (1, 1, {}),
     (2, 1, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],
    [(1, 3, {}), (1, 3, {}), (1, 3, {}),
     (2, 4, {}), (3, 3, {}), (4, 3, {}), (5, 4, {})],
    [(1, 1, {}), (1, 1, {}), (1, 2, {}),
     (2, 2, {}), (3, 1, {}), (4, 2, {}), (5, 2, {})],
    [(1, 2, {}), (1, 2, {}), (1, 2, {}),
     (2, 3, {}), (3, 1, {}), (4, 2, {}), (5, 1, {})],
    [(1, 2, {}), (1, 2, {}), (1, 2, {}),
     (2, 3, {}), (3, 2, {}), (4, 2, {}), (5, 2, {})],

    [(1, 2, {}), (1, 2, {}), (1, 2, {}),
     (2, 3, {}), (3, 3, {}), (4, 2, {}), (5, 2, {})],
    [(1, 1, {}), (1, 2, {}), (1, 2, {}),
     (2, 2, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],
    [(1, 3, {}), (1, 3, {}), (1, 3, {}),
     (2, 3, {}), (3, 4, {}), (4, 3, {}), (5, 3, {})],
    [(1, 2, {}), (1, 2, {}), (1, 2, {}),
     (2, 2, {}), (3, 2, {}), (4, 2, {}), (5, 3, {})],
    [(1, 2, {}), (1, 3, {}), (1, 2, {}),
     (2, 2, {}), (3, 2, {}), (4, 2, {}), (5, 3, {})],

    [(1, 4, {}), (1, 4, {}), (1, 4, {}),
     (2, 4, {}), (3, 4, {}), (4, 4, {}), (5, 4, {})],
    [(1, 2, {}), (1, 2, {}), (1, 2, {}),
     (2, 3, {}), (3, 3, {}), (4, 4, {}), (5, 3, {})],
    [(1, 1, {}), (1, 1, {}), (1, 1, {}),
     (2, 1, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],
    [(1, 2, {}), (1, 2, {}), (1, 2, {}),
     (2, 3, {}), (3, 2, {}), (4, 1, {}), (5, 2, {})],
    [(1, 1, {}), (1, 2, {}), (1, 1, {}),
     (2, 1, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],

    [(1, 1, {}), (1, 1, {}), (1, 1, {}),
     (2, 2, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],
    [(1, 1, {}), (1, 1, {}), (1, 1, {}),
     (2, 1, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],
    [(1, 1, {}), (1, 1, {}), (1, 1, {}),
     (2, 1, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],
    [(1, 2, {}), (1, 2, {}), (1, 2, {}),
     (2, 2, {}), (3, 2, {}), (4, 2, {}), (5, 1, {})],
    [(1, 2, {}), (1, 2, {}), (1, 2, {}),
     (2, 1, {}), (3, 3, {}), (4, 2, {}), (5, 2, {})],

    [(1, 2, {}), (1, 2, {}), (1, 2, {}),
     (2, 2, {}), (3, 2, {}), (4, 2, {}), (5, 2, {})],
    [(1, 2, {}), (1, 2, {}), (1, 2, {}),
     (2, 2, {}), (3, 2, {}), (4, 2, {}), (5, 1, {})],
    [(1, 2, {}), (1, 2, {}), (1, 2, {}),
     (2, 3, {}), (3, 2, {}), (4, 2, {}), (5, 2, {})],
    [(1, 2, {}), (1, 2, {}), (1, 1, {}),
     (2, 2, {}), (3, 2, {}), (4, 2, {}), (5, 2, {})],
    [(1, 1, {}), (1, 1, {}), (1, 1, {}),
     (2, 1, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],

    [(1, 1, {}), (1, 1, {}), (1, 1, {}),
     (2, 1, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],
    [(1, 2, {}), (1, 3, {}), (1, 2, {}),
     (2, 2, {}), (3, 2, {}), (4, 2, {}), (5, 2, {})],
    [(1, 1, {}), (1, 1, {}), (1, 1, {}),
     (2, 1, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],
    [(1, 1, {}), (1, 1, {}), (1, 1, {}),
     (2, 1, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],
    [(1, 1, {}), (1, 1, {}), (1, 2, {}),
     (2, 1, {}), (3, 1, {}), (4, 2, {}), (5, 1, {})],

    [(1, 1, {}), (1, 1, {}), (1, 1, {}),
     (2, 1, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],
    [(1, 3, {}), (1, 3, {}), (1, 3, {}),
     (2, 3, {}), (3, 2, {}), (4, 3, {}), (5, 3, {})],
    [(1, 1, {}), (1, 1, {}), (1, 1, {}),
     (2, 1, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],
    [(1, 2, {}), (1, 2, {}), (1, 2, {}),
     (2, 2, {}), (3, 2, {}), (4, 2, {}), (5, 2, {})],
    [(1, 2, {}), (1, 2, {}), (1, 2, {}),
     (2, 3, {}), (3, 2, {}), (4, 3, {}), (5, 2, {})],

    [(1, 4, {}), (1, 3, {}), (1, 3, {}),
     (2, 4, {}), (3, 3, {}), (4, 4, {}), (5, 3, {})],
    [(1, 2, {}), (1, 2, {}), (1, 1, {}),
     (2, 2, {}), (3, 2, {}), (4, 3, {}), (5, 2, {})],
    [(1, 2, {}), (1, 3, {}), (1, 2, {}),
     (2, 3, {}), (3, 2, {}), (4, 3, {}), (5, 3, {})],
    [(1, 3, {}), (1, 3, {}), (1, 3, {}),
     (2, 3, {}), (3, 4, {}), (4, 3, {}), (5, 2, {})],
    [(1, 1, {}), (1, 1, {}), (1, 1, {}),
     (2, 1, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],

    [(1, 1, {}), (1, 1, {}), (1, 1, {}),
     (2, 1, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],
    [(1, 1, {}), (1, 2, {}), (1, 1, {}),
     (2, 2, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],
    [(1, 2, {}), (1, 3, {}), (1, 2, {}),
     (2, 2, {}), (3, 2, {}), (4, 2, {}), (5, 2, {})],
    [(1, 1, {}), (1, 2, {}), (1, 1, {}),
     (2, 1, {}), (3, 1, {}), (4, 1, {}), (5, 1, {})],
    [(1, 2, {}), (1, 2, {}), (1, 2, {}),
     (2, 2, {}), (3, 2, {}), (4, 2, {}), (5, 2, {})]]

# Answers estimated using expectation-maximization with confusion matrices:
DS_EM_CM_RESOLUTIONS = [
    # Note that the seventh entry here disagrees with Dawid & Skene, possibly
    # due to a numerical error in the original paper.
    {1: 1.0}, {4: 1.0}, {2: 1.0}, {2: 1.0}, {2: 1.0},
    {2: 1.0}, {1: 0.981, 2: 0.019}, {3: 1.0}, {2: 1.0}, {2: 1.0},
    {4: 1.0}, {3: 1.0}, {1: 1.0}, {2: 1.0}, {1: 1.0},
    {1: 1.0}, {1: 1.0}, {1: 1.0}, {2: 1.0}, {2: 1.0},
    {2: 1.0}, {2: 1.0}, {2: 1.0}, {2: 1.0}, {1: 1.0},
    {1: 1.0}, {2: 1.0}, {1: 1.0}, {1: 1.0}, {1: 0.999, 2: 0.001},
    {1: 1.0}, {3: 1.0}, {1: 1.0}, {2: 1.0}, {2: 0.948, 3: 0.052},
    {4: 1.0}, {2: 1.0}, {2: 0.021, 3: 0.979}, {3: 1.0}, {1: 1.0},
    {1: 1.0}, {1: 1.0}, {2: 1.0}, {1: 1.0}, {2: 1.0}]

# Data object with no resolutions:
DS_DATA = dict(zip(range(1, 46), [(response, {}) for response in DS_RESPONSES]))

# Data object with the EM-CM-estimated resolutions:
DS_DATA_FINAL = dict(zip(range(1, 46), zip(DS_RESPONSES, DS_EM_CM_RESOLUTIONS)))

# A modified data set with extra copies of some questions; we use this to test
# that question-weighting works correctly:
DS_DATA_EXTRA = DS_DATA_FINAL.copy()
DS_DATA_EXTRA['Extra question 1'] = DS_DATA_FINAL[DS_DATA_FINAL.keys()[0]]
DS_DATA_EXTRA['Extra question 2'] = DS_DATA_FINAL[DS_DATA_FINAL.keys()[1]]
DS_DATA_EXTRA['Extra question 3'] = DS_DATA_FINAL[DS_DATA_FINAL.keys()[2]]
DS_DATA_EXTRA['Extra question 4'] = DS_DATA_FINAL[DS_DATA_FINAL.keys()[3]]
# For convneience, set weights such that the results of all tested methods
# should be the same for the weighted data as for the original data:
DS_EXTRA_WEIGHTS = {DS_DATA_FINAL.keys()[0]: 0.1, 'Extra question 1': 0.9,
                    DS_DATA_FINAL.keys()[1]: 0.8, 'Extra question 2': 0.2,
                    DS_DATA_FINAL.keys()[2]: 0.3, 'Extra question 3': 0.7,
                    DS_DATA_FINAL.keys()[3]: 0.4, 'Extra question 4': 0.6}


#########################
# Helper test functions #
#########################


def AssertMapsAlmostEqual(test, expected, result, label='key', places=3):
  """Tests for equality of two dicts."""
  for key in set(expected.keys() + result.keys()):
    test.assertAlmostEqual(expected.get(key, 0.0),
                           result.get(key, 0.0),
                           places=places,
                           msg='Unequal on ' + label + ' "' + str(key) + '"')


def AssertResolutionsAlmostEqual(test, expected, result, places=3):
  """A helper function for comparing two dicts of resolutions."""
  for question in set(expected.keys() + result.keys()):
    test.assertTrue(question in expected, msg='question "' +
                    str(question) + '" in result but not expected')
    test.assertTrue(question in result, msg='question "' +
                    str(question) + '" expected but not in result')
    AssertMapsAlmostEqual(test,
                          expected[question],
                          result[question],
                          label=('question "' + str(question) + '", answer'),
                          places=places)


def AssertConfusionMatricesAlmostEqual(test, expected, result, places=3):
  """A helper function for comparing two sets of confusion matrices."""
  for contributor in set(expected.keys() + result.keys()):
    test.assertTrue(contributor in expected, msg='contributor "' +
                    str(contributor) + '" in result but not expected')
    test.assertTrue(contributor in result, msg='contributor "' +
                    str(contributor) + '" expected but not in result')
    for answer in set(expected[contributor].keys() +
                      result[contributor].keys()):
      test.assertTrue(contributor in expected, msg='contributor "' +
                      str(contributor) + '", answer "' + str(answer) +
                      '" in result but not expected')
      test.assertTrue(contributor in result, msg='contributor "' +
                      str(contributor) + '", answer "' + str(answer) +
                      '" expected but not in result')
      AssertMapsAlmostEqual(test,
                            expected[contributor][answer],
                            result[contributor][answer],
                            label=('contributor "' + str(contributor) +
                                   '", answer "' + str(answer) +
                                   '", judgment'),
                            places=places)
