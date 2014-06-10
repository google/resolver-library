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


import copy

import numpy

import unittest
from resolver import decision_tree
from resolver import test_util


# An example supposing that contributors are entering blood type
# classifications, where each judgment includes both a group and a type:
TEST_DATA = {'q1': ([('c1', ('ABO', 'A'), {}),
                     ('c2', ('Rh', '+'), {}),
                     ('c3', ('ABO', 'B'), {})],
                    {('ABO',): 0.8,
                     ('ABO', 'A'): 0.4,
                     ('ABO', 'B'): 0.4,
                     ('Rh',): 0.2,
                     ('Rh', '+'): 0.2}),
             'q2': ([('c1', ('Rh', '+'), {}),
                     ('c2', ('Rh', '-'), {}),
                     ('c3', ('ABO', 'O'), {})],
                    {('ABO',): 0.2,
                     ('ABO', 'O'): 0.2,
                     ('Rh',): 0.8,
                     ('Rh', '+'): 0.4,
                     ('Rh', '-'): 0.4}),
             'q3': ([('c1', ('Rh', '-'), {}),
                     ('c2', ('Other',), {}),
                     ('c3', ('Rh', '-'), {})],
                    {('Rh',): 0.4,
                     ('Rh', '-'): 0.4,
                     ('Other',): 0.6,
                     ('junk',): 0.0})}  # to test that we handle probability 0.0


class PathViewTest(unittest.TestCase):
  longMessage = True

  # This is how a PathView of TEST_DATA looks with various
  # paths (the keys of the FILTERED_DATA dict):
  FILTERED_DATA = {
      (): {'q1': ([('c1', 'ABO', {}), ('c2', 'Rh', {}), ('c3', 'ABO', {})],
                  {'Rh': 0.2, 'ABO': 0.8}),
           'q2': ([('c1', 'Rh', {}), ('c2', 'Rh', {}), ('c3', 'ABO', {})],
                  {'Rh': 0.8, 'ABO': 0.2}),
           'q3': ([('c1', 'Rh', {}), ('c2', 'Other', {}), ('c3', 'Rh', {})],
                  {'Rh': 0.4, 'Other': 0.6, 'junk': 0.0})},
      ('ABO',): {'q1': ([('c1', 'A', {}), ('c3', 'B', {})],
                        {'A': 0.5, 'B': 0.5}),
                 'q2': ([('c3', 'O', {})],
                        {'O': 1.0}),
                 'q3': ([],
                        {})},
      ('Rh',): {'q1': ([('c2', '+', {})],
                       {'+': 1.0}),
                'q2': ([('c1', '+', {}), ('c2', '-', {})],
                       {'+': 0.5, '-': 0.5}),
                'q3': ([('c1', '-', {}), ('c3', '-', {})],
                       {'-': 1.0})},
      ('Other',): {'q1': ([],
                          {}),
                   'q2': ([],
                          {}),
                   'q3': ([],
                          {})}}

  def testFilteredResponses(self):
    for question in TEST_DATA:
      responses = TEST_DATA[question][0]
      self.assertEqual(
          self.FILTERED_DATA[()][question][0],
          decision_tree.PathView.FilteredResponses(responses))
      self.assertEqual(
          self.FILTERED_DATA[('ABO',)][question][0],
          decision_tree.PathView.FilteredResponses(responses, ('ABO',)))

  def testFilteredResolution(self):
    for question in TEST_DATA:
      resolution_map = TEST_DATA[question][1]
      self.assertEqual(
          self.FILTERED_DATA[()][question][1],
          decision_tree.PathView.FilteredResolution(resolution_map))
      self.assertEqual(
          self.FILTERED_DATA[('ABO',)][question][1],
          decision_tree.PathView.FilteredResolution(resolution_map, ('ABO',)))

  def test__getitem__(self):  # pylint: disable=g-bad-name
    data_view = decision_tree.PathView(TEST_DATA)
    data_view.SetPrimaryKey()
    for question in TEST_DATA:
      self.assertEqual(self.FILTERED_DATA[()][question], data_view[question])
    data_view.SetPrimaryKey(('ABO',))
    for question in TEST_DATA:
      self.assertEqual(self.FILTERED_DATA[('ABO',)][question],
                       data_view[question])

  def test__iter__(self):  # pylint: disable=g-bad-name
    data_view = decision_tree.PathView(TEST_DATA)
    data_view.SetPrimaryKey(())
    self.assertEqual(set(k for k in data_view), set(TEST_DATA.keys()))
    data_view.SetPrimaryKey(('ABO',))
    self.assertEqual(set(k for k in data_view), set(TEST_DATA.keys()))
    data_view.SetPrimaryKey(('Rh',))
    self.assertEqual(set(k for k in data_view), set(TEST_DATA.keys()))

  def test__len__(self):  # pylint: disable=g-bad-name
    data_view = decision_tree.PathView(TEST_DATA)
    self.assertEqual(len(TEST_DATA), len(data_view))

  def testiteritems(self):  # pylint: disable=g-bad-name
    data_view = decision_tree.PathView(TEST_DATA)
    data_view.SetPrimaryKey(())
    for question, (responses, resolution_map) in data_view.iteritems():
      self.assertEqual(self.FILTERED_DATA[()][question],
                       (responses, resolution_map))
    data_view.SetPrimaryKey(('ABO',))
    for question, (responses, resolution_map) in data_view.iteritems():
      self.assertEqual(self.FILTERED_DATA[('ABO',)][question],
                       (responses, resolution_map))


class DecisionTreeTest(unittest.TestCase):
  longMessage = True

  def test_ExpandResolution(self):  # pylint: disable=g-bad-name
    resolution_map = {('apidae',): 0.5,  # this one should get overwritten!
                      ('apidae', 'apis', 'mellifera'): 0.9,
                      ('apidae', 'apis', 'cerana'): 0.05,
                      ('apidae', 'bombus'): 0.05}
    expected = {('apidae',): 1.0,
                ('apidae', 'apis'): 0.95,
                ('apidae', 'apis', 'mellifera'): 0.9,
                ('apidae', 'apis', 'cerana'): 0.05,
                ('apidae', 'bombus'): 0.05}
    decision_tree.DecisionTree._ExpandResolution(resolution_map)
    test_util.AssertMapsAlmostEqual(self, expected, resolution_map,
                                    label='answer')

  def testIsSubpath(self):
    resolution_map = {('apidae',): 1.0,
                      ('apidae', 'apis'): 0.95,
                      ('apidae', 'apis', 'mellifera'): 0.9,
                      ('apidae', 'apis', 'cerana'): 0.05,
                      ('apidae', 'bombus'): 0.05}
    self.assertTrue(decision_tree.DecisionTree.IsSubpath(
        (), resolution_map))
    self.assertTrue(decision_tree.DecisionTree.IsSubpath(
        ('apidae',), resolution_map))
    self.assertTrue(decision_tree.DecisionTree.IsSubpath(
        ('apidae', 'apis'), resolution_map))
    self.assertFalse(decision_tree.DecisionTree.IsSubpath(
        ('apidae', 'apis', 'mellifera'), resolution_map))
    self.assertFalse(decision_tree.DecisionTree.IsSubpath(
        ('apidae', 'bombus'), resolution_map))
    self.assertFalse(decision_tree.DecisionTree.IsSubpath(
        ('something', 'else', 'entirely'), resolution_map))
    # Check that it works even with a non-expanded resolution:
    resolution_map = {('apidae', 'apis', 'mellifera'): 0.9,
                      ('apidae', 'apis', 'cerana'): 0.05,
                      ('apidae', 'bombus'): 0.05}
    self.assertTrue(decision_tree.DecisionTree.IsSubpath(
        (), resolution_map))
    self.assertTrue(decision_tree.DecisionTree.IsSubpath(
        ('apidae',), resolution_map))
    self.assertTrue(decision_tree.DecisionTree.IsSubpath(
        ('apidae', 'apis'), resolution_map))
    self.assertFalse(decision_tree.DecisionTree.IsSubpath(
        ('apidae', 'apis', 'mellifera'), resolution_map))
    self.assertFalse(decision_tree.DecisionTree.IsSubpath(
        ('apidae', 'bombus'), resolution_map))
    self.assertFalse(decision_tree.DecisionTree.IsSubpath(
        ('something', 'else', 'entirely'), resolution_map))

  def testInitializeResolutions(self):
    data = copy.deepcopy(TEST_DATA)
    expected = {'q1': {('ABO',): 2.0 / 3.0,
                       ('ABO', 'A'): 1.0 / 3.0,
                       ('ABO', 'B'): 1.0 / 3.0,
                       ('Rh',): 1.0 / 3.0,
                       ('Rh', '+'): 1.0 / 3.0},
                'q2': {('ABO',): 1.0 / 3.0,
                       ('ABO', 'O'): 1.0 / 3.0,
                       ('Rh',): 2.0 / 3.0,
                       ('Rh', '+'): 1.0 / 3.0,
                       ('Rh', '-'): 1.0 / 3.0},
                'q3': {('Rh',): 2.0 / 3.0,
                       ('Rh', '-'): 2.0 / 3.0,
                       ('Other',): 1.0 / 3.0}}
    decision_tree.DecisionTree.InitializeResolutions(
        data, overwrite_all_resolutions=True)
    result = decision_tree.DecisionTree.ExtractResolutions(data)
    test_util.AssertResolutionsAlmostEqual(self, expected, result)

  def testRandomAnswer(self):
    numpy.random.seed(1)  # for determinism
    # TODO(tpw):  Add numpy to BUILD for this test
    # In the following resolution map, all Apis are either A. mellifera or A.
    # cerana.  However, the map includes Xylocopa that are not X. violacea, but
    # are instead anonymous at the species level:
    resolution_map = {('apidae',): 1.0,
                      ('apidae', 'apis'): 0.55,
                      ('apidae', 'apis', 'mellifera'): 0.25,
                      ('apidae', 'apis', 'cerana'): 0.3,
                      ('apidae', 'xylocopa',): 0.45,
                      ('apidae', 'xylocopa', 'violacea'): 0.4}
    # Therefore, RandomAnswer should be able to return A. mellifera, A. cerana,
    # X. violacea, and also just Xylocopa, but not Apis (because all Apis are
    # either A. mellifera or A. cerana):
    expected = set([('apidae', 'apis', 'mellifera'),
                    ('apidae', 'apis', 'cerana'),
                    ('apidae', 'xylocopa',),
                    ('apidae', 'xylocopa', 'violacea')])
    results = set([decision_tree.DecisionTree.RandomAnswer(resolution_map)
                   for _ in range(50)])
    self.assertEqual(expected, results)

  def testMostLikelyAnswer(self):
    resolution_map = {('apidae', 'apis'): 0.55,
                      ('apidae', 'apis', 'mellifera'): 0.25,
                      ('apidae', 'apis', 'cerana'): 0.3,
                      ('apidae', 'xylocopa',): 0.45,
                      ('apidae', 'xylocopa', 'violacea'): 0.4}
    # Note that we skipped the top-level ('apidae',): 1.0 entry in
    # resolution_map, because MostLikelyAnswer does not require fully-specified
    # resolution maps.
    self.assertEqual((('apidae', 'xylocopa', 'violacea'), 0.4),
                     decision_tree.DecisionTree.MostLikelyAnswer(
                         resolution_map))

  def test_SetParameters(self):  # pylint: disable=g-bad-name

    # Create a mock submodel that just notes the PathView settings it sees:
    class MockModel(object):

      def SetMockParameters(
          self, data, question_weights=None):  # pylint: disable=unused-argument
        self.called_with = data._path

    # Create a subclass that uses our mock model as its submodel:
    class MyDecisionTree(decision_tree.DecisionTree):

      @staticmethod
      def _CreateSubmodel(unused_path):
        return MockModel()

    my_decision_tree_model = MyDecisionTree()
    # Call _SetParameters on the subclass:
    my_decision_tree_model._SetParameters(TEST_DATA, 'SetMockParameters')
    # Expect that the model has created the correct submodels and called
    # SetMockParameters on each:
    expected_submodels = set(((), ('ABO',), ('Rh',)))
    self.assertEqual(expected_submodels,
                     set(my_decision_tree_model.model_tree.keys()))
    for submodel in expected_submodels:
      self.assertEqual(submodel,
                       my_decision_tree_model.model_tree[submodel].called_with)

  def testSetMLEParameters(self):
    # The previous test checked that we call submodels using the correct paths;
    # this test is more end-to-end, checking that the correct submodels are
    # created and that we set the correct parameters for one of them.
    decision_tree_model = decision_tree.DecisionTree()
    decision_tree_model.SetMLEParameters(TEST_DATA)

    # Test that the correct submodels were created:
    self.assertEqual(set(((), ('ABO',), ('Rh',))),
                     set(decision_tree_model.model_tree.keys()))

    # Test the root confusion matrix parameters:
    expected_priors = {'ABO': 1.0 / 3.0, 'Rh': 1.4 / 3.0, 'Other': 0.6 / 3.0}
    test_util.AssertMapsAlmostEqual(
        self,
        expected_priors,
        decision_tree_model.model_tree[()].priors)
    expected_cm = {'c1': {'ABO': {'ABO': 0.8 / 1.0, 'Rh': 0.2 / 1.0},
                          'Rh': {'ABO': 0.2 / 1.4, 'Rh': 1.2 / 1.4},
                          'Other': {'Rh': 1.0}},
                   'c2': {'ABO': {'Rh': 1.0},
                          'Rh': {'Rh': 1.0 / 1.4, 'Other': 0.4 / 1.4},
                          'Other': {'Other': 1.0}},
                   'c3': {'ABO': {'ABO': 1.0},
                          'Rh': {'ABO': 1.0 / 1.4, 'Rh': 0.4 / 1.4},
                          'Other': {'Rh': 1.0}}}
    test_util.AssertConfusionMatricesAlmostEqual(
        self,
        expected_cm,
        decision_tree_model.model_tree[()].confusion_matrices)

    # Test the ('ABO',) confusion matrix parameters:
    expected_priors = {'A': 0.4, 'B': 0.4, 'O': 0.2}
    test_util.AssertMapsAlmostEqual(
        self,
        expected_priors,
        decision_tree_model.model_tree[('ABO',)].priors)
    expected = {'c1': {'A': {'A': 1.0},
                       'B': {'A': 1.0}},
                # c2 never said 'ABO', so it has no entry here
                'c3': {'A': {'B': 1.0},
                       'B': {'B': 1.0},
                       'O': {'O': 1.0}}}
    test_util.AssertConfusionMatricesAlmostEqual(
        self,
        expected,
        decision_tree_model.model_tree[('ABO',)].confusion_matrices)

    # Test the ('Rh',) confusion matrix parameters:
    expected_priors = {'+': 0.6 / 1.4, '-': 0.8 / 1.4}
    test_util.AssertMapsAlmostEqual(
        self,
        expected_priors,
        decision_tree_model.model_tree[('Rh',)].priors)
    expected = {'c1': {'+': {'+': 1.0},
                       '-': {'+': 0.5, '-': 0.5}},
                'c2': {'+': {'+': 0.2 / 0.6, '-': 0.4 / 0.6},
                       '-': {'-': 1.0}},
                'c3': {'-': {'-': 1.0}}}
    test_util.AssertConfusionMatricesAlmostEqual(
        self,
        expected,
        decision_tree_model.model_tree[('Rh',)].confusion_matrices)

  def testSetSampleParameters(self):
    # The logic of _SetParameters is tested by the two methods above, so here
    # we'll just check that the correct submodels get created:
    decision_tree_model = decision_tree.DecisionTree()
    decision_tree_model.SetSampleParameters(TEST_DATA)
    self.assertEqual(set(((), ('ABO',), ('Rh',))),
                     set(decision_tree_model.model_tree.keys()))

  def testSetVariationalParameters(self):
    # The logic of _SetParameters is tested above, so here we'll just check
    # that the correct submodels get created:
    decision_tree_model = decision_tree.DecisionTree()
    decision_tree_model.SetVariationalParameters(TEST_DATA)
    self.assertEqual(set(((), ('ABO',), ('Rh',))),
                     set(decision_tree_model.model_tree.keys()))

  def testResolveQuestion(self):
    decision_tree_model = decision_tree.DecisionTree()
    decision_tree_model.SetMLEParameters(TEST_DATA)

    # Using the MLE model parameters we know from above, we worked out the
    # following resolutions by hand:
    responses = [('c1', ('ABO', 'A'), {})]
    expected = {('ABO',): 0.8, ('Rh',): 0.2}
    expected[('ABO', 'A')] = expected[('ABO',)] * 0.4
    expected[('ABO', 'B')] = expected[('ABO',)] * 0.4
    expected[('ABO', 'O')] = expected[('ABO',)] * 0.2
    expected[('Rh', '+')] = expected[('Rh',)] * 0.6 / 1.4
    expected[('Rh', '-')] = expected[('Rh',)] * 0.8 / 1.4
    result = decision_tree_model.ResolveQuestion(responses)
    test_util.AssertMapsAlmostEqual(self, expected, result)

    responses = [('c2', ('Rh', '-'), {})]
    expected = {('ABO',): 0.5, ('Rh',): 0.5}
    expected[('ABO', 'A')] = expected[('ABO',)] * 0.4
    expected[('ABO', 'B')] = expected[('ABO',)] * 0.4
    expected[('ABO', 'O')] = expected[('ABO',)] * 0.2
    expected[('Rh', '+')] = expected[('Rh',)] * 1.0 / 3.0
    expected[('Rh', '-')] = expected[('Rh',)] * 2.0 / 3.0
    result = decision_tree_model.ResolveQuestion(responses)
    test_util.AssertMapsAlmostEqual(self, expected, result)


if __name__ == '__main__':
  unittest.main()
