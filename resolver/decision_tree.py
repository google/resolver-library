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

"""A model comprising a tree of confusion matrices.

In this model, all judgments must be tuples.  Each judgment is regarded as a
path through a decision tree, expressed as a tuple whose elements are
successive choices in the tree.

For example, suppose that judgments express morphological classifications from
the GalaxyZoo project.  The top-level judgments are chosen from "not a galaxy",
"elliptical", and "disk".  Elliptical galaxies are then judged on their
ellipticity: "cigar shaped", "completely round", or "in between", while disk
galaxies have other further judgments.

In this example, a judgment path of "not a galaxy" would be encoded as the
1-tuple ("not a galaxy",) and a judgment path of "elliptical" followed by
"cigar shaped" would be encoded as the 2-tuple ('elliptical', 'cigar shaped').

Resolutions returned by this model include all possible paths.  Thus, if we
have 60% confidence that an object is a cigar-shaped elliptical galaxy, 5% that
it is a completely round elliptical galaxy, 15% that it is an in-between
elliptical galaxy, and 20% that it is not a galaxy, the resolution will include
the implied 80% confidence that the galaxy is elliptical.  Thus, the resolution
map will look like this:
  {('elliptical',): 0.8,
   ('elliptical', 'cigar shaped'): 0.6,
   ('elliptical', 'completely round'): 0.05,
   ('elliptical', 'in between'): 0.15,
   ('not a galaxy',): 0.2}

To facilitate slicing ResolverData objects by path, this module provides a new
class, PathView, that provides a tunable view of the judgments and resolutions.
Resolution probabilities as seen through a PathView are conditioned on the path
in use.

In the context of the above example, a PathView with path set to () would see
the above resolution as:
  {('elliptical',): 0.8,
   ('not a galaxy',): 0.2}
With path set to ('elliptical',), it would see the resolution as:
  {('elliptical', 'cigar shaped'): 0.75,
   ('elliptical', 'completely round'): 0.0625,
   ('elliptical', 'in between'): 0.1875}

PathViews ignore the root node of the tree as seen from their current path.
Thus, setting path to ('elliptical', 'cigar shaped') or to ('not a galaxy')
would make the above resolution empty.

The DecisionTree model makes use of the optional question_weights parameter in
its submodels.  This is to avoid double-counting errors.  In the above example,
a judgment of ('elliptical', 'cigar shaped') has an 80% probability of being
correct with regard to the 'elliptical' part, and conditioned on that, a 75%
probability of being correct with resepect to the 'cigar shaped' part.  We give
the 'cigar shaped' part of the judgment only 80% weight, however, because if
the object is actually 'not a galaxy', then the mistake is entirely counted by
the 'elliptical' judgment.
"""

import collections
import random

from resolver import confusion_matrices
from resolver import model


class PathView(collections.Mapping):
  """A dict view that slices ResolverData objects by judgment path."""

  @staticmethod
  def FilteredResponses(responses, path=None):
    """Slices responses by path.

    Args:
      responses:  An iterable of (contributor, judgment, metadata) tuples, in
                  which each judgment is a tuple representing a judgment path.
      path:  A tuple representing a judgment path.

    Returns:
      A list of (contributor, judgment, metadata) tuples, in which each
      judgment is an individual judgment.

      Each judgment returned is found by attempting to match path in the full
      judgment tuple, and if found, yielding the next (single) step in the
      path.  That is, we want the first judgment that hangs off a given path.

      For example, suppose:
        responses = [('c1', ('elliptical', 'in between'), {}),
                     ('c2', ('elliptical', 'in between', 'lensing'), {}),
                     ('c3', ('not a galaxy',), {}),
                     ('c4', ('elliptical',), {})]
      Then FilteredResponses(responses, path=('elliptical',)) returns:
        [('c1', 'in between', {}),
         ('c2', 'in between', {})]
      The judgment from 'c3' is excluded because it does not start with
      'elliptical'.  The judgment from 'c4' is excluded because it has nothing
      after 'elliptical'.
    """
    if path is None:
      path = ()
    depth = len(path)
    return [(contributor, judgment[depth], metadata)
            for contributor, judgment, metadata in responses
            if judgment[:depth] == path and len(judgment) > depth]

  @staticmethod
  def FilteredResolution(resolution_map, path=None):
    """Slices resolution maps by path, yielding conditional probabilities.

    Args:
      resolution_map:  A map from answer to probability, where answer is
                       a tuple representing an answer path.  The map must be
                       fully specified (in the sense of the _ExpandResolution
                       method).
      path:  A tuple representing an answer path.

    Returns:
      A map from answer to conditional probability, in which answer is an
      individual answer.

      Conditional probability means this:  In the returned map, each answer is
      assigned a probability conditioned on the 'state of the world' being an
      extension of path, e.g., if we know that a galaxy is 'elliptical', what
      is the distribution of options given that assumption?

      For example, suppose:
        resolution_map = {('elliptical',): 0.8,
                          ('elliptical', 'in between'): 0.6,
                          ('not a galaxy',): 0.2}
      Then FilteredResolution(resolution_map, path=('elliptical',))
      returns:
        {'in between': 0.75}
    """
    if path is None:
      path = ()
    depth = len(path)
    if depth:
      if path in resolution_map:
        # Get the marginal probability for this path:
        marginal_probability = resolution_map[path]
        if not marginal_probability:
          # This path has zero probability, so return the empty resolution:
          return {}
      else:
        return {}
    else:
      # depth is 0, so we're at the root level:
      marginal_probability = 1.0
    filtered_resolution_map = {}
    for answer_tuple, probability in resolution_map.iteritems():
      if answer_tuple[:depth] == path and len(answer_tuple) == depth + 1:
        answer_slice = answer_tuple[depth]
        # Divide joint probability by marginal probability to get conditional
        # probability:
        filtered_resolution_map[answer_slice] = (probability
                                                 / marginal_probability)
    return filtered_resolution_map

  def SetPrimaryKey(self, path=None):
    """Sets our view to a particular judgment/answer path."""
    if path is None:
      path = ()
    self._path = path

  def __init__(self, data):
    """Cretes a view of a ResolverData object."""
    self._data_dict = data
    self._path = ()

  def __getitem__(self, key):
    """Provies a dict getter that returns filtered responses and resolutions."""
    responses, resolution_map = self._data_dict.__getitem__(key)
    return (self.FilteredResponses(responses, self._path),
            self.FilteredResolution(resolution_map, self._path))

  def __iter__(self):
    """Provides self._data_dict's key iterator."""
    return self._data_dict.__iter__()

  def __len__(self):
    """Provides self._data_dict's length method."""
    return self._data_dict.__len__()


class DecisionTree(model.StatisticalModel):
  """A Resolver model for decision trees, with each node a confusion matrix."""

  def __init__(self):
    self.model_tree = {}

  @staticmethod
  def _CreateSubmodel(unused_path):
    """Returns a new ConfusionMatrix object.  Convenient for overriding."""
    return confusion_matrices.ConfusionMatrices()

  @staticmethod
  def _ExpandResolution(resolution_map):
    """Converts a resolution from terminal-only to fully specified.

    Args:
      resolution_map:  A map from answer tuple to probability.

    This method is called by InitializeResolutions to extend the initial
    resolutions set by the superclass method into the subpath-inclusive
    resolutions specified by this class.
    For example, if this method is called on a resolution
      {('apidae', 'apis', 'mellifera'): 0.9,
       ('apidae', 'apis', 'cerana'): 0.05,
       ('apidae', 'bombus'): 0.05}
    then we expand subpaths and compute marginals, turning that resolution into
      {('apidae',): 1.0,
       ('apidae', 'apis'): 0.95,
       ('apidae', 'apis', 'mellifera'): 0.9,
       ('apidae', 'apis', 'cerana'): 0.05,
       ('apidae', 'bombus'): 0.05}
    If subpaths are already included in the resolution, they will be
    overwritten.  Thus, this method is idempotent.
    """
    # First, clear out all non-terminal answers.  (This ensures that this
    # method is idempotent.)
    for answer, probability in resolution_map.items():  # iterate on a copy
      for depth in range(1, len(answer)):
        answer_slice = answer[:depth]
        if answer_slice in resolution_map:
          del resolution_map[answer_slice]
    # Next, add each leading subpath of each answer to the resolution map:
    for answer, probability in resolution_map.items():  # iterate on a copy
      for depth in range(1, len(answer)):
        answer_slice = answer[:depth]
        resolution_map[answer_slice] = (
            resolution_map.get(answer_slice, 0.0) + probability)

  @staticmethod
  def IsSubpath(answer, resolution_map):
    """Determines whether an answer is a subpath within a resolution map.

    Args:
      answer:  A tuple representing a judgment path.
      resolution_map:  A map from answer to probability, where answer is
                       a tuple representing an answer path.  The map does not
                       need to be fully specified (in the sense of the
                       _ExpandResolution method).

    Returns:
      True if answer is a strict subpath of any other element of resolution_map,
      False otherwise.
    """
    return any(len(answer) < len(x) and answer == x[:len(answer)]
               for x in resolution_map)

  @staticmethod
  def InitializeResolutions(data,
                            overwrite_all_resolutions=False,
                            judgments_to_answers=None):
    """Like the superclass's method but populates partial path resolutions.

    Args:
      data:  A ResolverData object.
      overwrite_all_resolutions:  As in StatisticalModel.InitializeResolutions.
      judgments_to_answers:  Must be None, since this model doesn't
                             support judgment remapping.

    For example, if data contains a question with judgments
    ('apidae', 'apis', 'mellifera') and ('apidae', 'apis', 'cerana'), we
    set the corresponding resolution to:
      {('apidae',): 1.0,
       ('apidae', 'apis'): 1.0,
       ('apidae', 'apis', 'mellifera'): 0.5,
       ('apidae', 'apis', 'cerana'): 0.5}
    """
    assert not judgments_to_answers  # TODO(tpw):  Implement support for this?

    # Use the superclass's method to get all the full judgment paths:
    model.StatisticalModel.InitializeResolutions(
        data,
        overwrite_all_resolutions=overwrite_all_resolutions)
    # But we also need the subpaths of those paths:
    for _, (_, resolution_map) in data.iteritems():
      DecisionTree._ExpandResolution(resolution_map)

  @staticmethod
  def RandomAnswer(resolution_map):
    """Returns an answer drawn randomly from resolution_map.

    Args:
      resolution_map:  A map from answer to probability, where answer is
                       a tuple representing an answer path.  The map must be
                       fully specified (in the sense of the _ExpandResolution
                       method).

    Returns:
      A key from resolution_map.
      TODO: explain more
    """
    path = ()
    filtered_resolution = PathView.FilteredResolution(resolution_map,
                                                      path=path)
    while filtered_resolution:
      filtered_answer = model.StatisticalModel.RandomAnswer(filtered_resolution)
      if filtered_answer:
        path += (filtered_answer,)
        filtered_resolution = PathView.FilteredResolution(resolution_map,
                                                          path=path)
      else:
        # RandomAnswer returned None, which we take to be a "none of the above"
        # answer at some point in the tree, so we stop here.
        break
    return path

  @staticmethod
  def MostLikelyAnswer(resolution_map):
    """Returns the most probable leaf-node answer from a resolution map.

    Args:
      resolution_map:  A map from answer to probability, where answer is
                       a tuple representing an answer path.  The map does not
                       need to be fully specified (in the sense of the
                       _ExpandResolution method).

    Returns:
      A tuple containing (the highest probability answer in resolution_map that
      is not a prefix of another answer in resolution_map, the confidence in
      that answer).

    Example:
      r = {('apis',): 0.6,
           ('apis', 'mellifera'): 0.3,
           ('apis', 'cerana'): 0.3,
           ('xylocopa',): 0.4,
           ('xylocopa', 'violacea'): 0.4}
      MostLikelyAnswer(r) == (('xylocopa', 'violacea'), 0.4)
      Note that the answer is not ('apis',), because all Apis are either
      A. mellifera or A. cerana.
    """
    # First we select just the leaf nodes of resolution_map, then sort (for
    # deterministic tests) and shuffle (to break ties randomly):
    resolution_items = sorted(
        (answer, confidence)
        for answer, confidence in resolution_map.iteritems()
        if not DecisionTree.IsSubpath(answer, resolution_map))
    if resolution_items:
      random.shuffle(resolution_items)
      # Return the most likely answer, along with its confidence:
      return max(resolution_items, key=lambda x: x[1])
    return None

  def _SetParameters(self, data, setter_name):
    """Calls submodels' parameter-setters for each path in the decision tree.

    Args:
      data:  A ResolverData object.
      setter_name:  The name of the submodel setter to call, for example
                    "SetMLEParameters" or "SetVariationalParameters".

    This method has the following postconditions:
      1) self._paths is a set containing all non-terminal paths contained in
         the judgments present in data.  For example, if the path
         ('apidae', 'apis', 'mellifera') appears in the judgments in data, then
         self._paths will contain ('apidae',) and ('apidae', 'apis').
      2) self.model_tree is a dict containing a key for each element of
         self._paths, mapping to an object of type Model.
      3) For each path-model pair in self.model_tree, the model's method with
         name equal to setter_name has been called with a PathView of data as
         its argument, and the PathView's path set to path.
    In short, we ensure that all necessary submodels have been created, and we
    call each one's parameter-setter method with the corresponding slice of
    data.
    """
    # First, collect all non-terminal paths seen among the judgments:
    self._paths = set()
    for responses, _ in data.itervalues():
      for _, judgment, _ in responses:
        # We want the path minus the last element.  For example, if the
        # judgment is ('disk', 'not edge-on', 'spiral'), we need a submodel
        # rooted at the ('disk', 'not edge-on') key.
        self._paths.add(judgment[:-1])
    # Now we make sure we have all the subpaths.  In the above example, once we
    # have ('disk', 'not edge-on'), we also need to have ('disk',).
    for path in self._paths.copy():
      self._paths.update((path[:index] for index in range(len(path))))

    # Next, create a PathView of the data and iterate over all the slices:
    data_view = PathView(data)
    for path in self._paths:
      data_view.SetPrimaryKey(path)
      # Create the submodel for this key if it doesn't already exist:
      if path not in self.model_tree:
        # Construct a new submodel at this node in the tree:
        self.model_tree[path] = self._CreateSubmodel(path)
      # Set the question weights according to the parent node of path:
      weights = {}
      if path:
        for question, (_, resolution_map) in data.iteritems():
          if path in resolution_map:
            weights[question] = resolution_map[path]
      # Call the appropriate SetXParameters method for the node in model_tree
      # corresponding to the path path:
      getattr(self.model_tree[path], setter_name)(data_view,
                                                  question_weights=weights)

  def SetMLEParameters(self, data):
    """Calls _SetParameters and submodels' SetMLEParameters methods."""
    self._SetParameters(data, 'SetMLEParameters')

  def SetSampleParameters(self, data):
    """Calls _SetParameters and submodels' SetSampleParameters methods."""
    self._SetParameters(data, 'SetSampleParameters')

  def SetVariationalParameters(self, data):
    """Calls _SetParameters and submodels' SetVariationalParameters methods."""
    self._SetParameters(data, 'SetVariationalParameters')

  def ResolveQuestion(self, responses):
    """Returns one resolution map based on judgments and current parameters.

    Args:
      responses:  A list of (contributor, judgment, metadata) tuples, in
                  which each judgment is a tuple representing a judgment path.

    Returns:
      A map from answers to probabilities, in which each answer is a tuple
      representing an answer path.  The map is fully-specified (in the sense of
      the _ExpandResolution method).
    """
    resolution_map = {}

    for path in sorted(self.model_tree, key=len):
      # First, get the submodel and filtered responses corresponding to the
      # current path, then call the submodel's ResolveQuestion method to get
      # conditional probabilities of answers conditioned on path:
      conditional_resolution_map = self.model_tree[path].ResolveQuestion(
          PathView.FilteredResponses(responses, path))

      # The submodel doesn't understand judgment paths, so now we prepend
      # path to the answer singletons returned and multiply the probability by
      # the marginal probability of that path, producing a resolution_map with
      # fully-specified judgment paths and absolute probabilities:
      prefix_probability = resolution_map.get(path, 0.0) if path else 1.0
      resolution_map.update((path + (answer,), probability * prefix_probability)
                            for answer, probability
                            in conditional_resolution_map.iteritems())

    return resolution_map
