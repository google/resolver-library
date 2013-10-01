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

"""The substitution-sampling algorithm applied to judgment resolution.

This is based on the substitution sampling algorithm of Tanner and Wong (1987).

In the context of judgment resolution, substitution sampling means drawing
successive sample resolutions from the distribution
  P(resolutions|judgments)
by alternating between drawing resolutions from the distribution
  P(resolutions|judgments, model parameters)
and drawing parameters from the distribution
  P(model parameters|judgments, resolutions).

The elementwise mean of the sampled resolutions numerically approximates
the set of actual resolution probabilities.

Note that this inference method is not compatible with the model defined in
gaussian_contributors.py.
"""

__author__ = 'tpw@google.com (Tamsyn Waterhouse)'

import itertools
import numpy


# The number of samples to use to compute the resolution inference integral:
# Assuming 1/sqrt(n) variance, this should give roughly percent-level accuracy.
NUMBER_OF_SAMPLES = 10000


##################################
# The SubstitutionSampling class #
##################################


class SubstitutionSampling(object):
  """Implements the substitution sampling algorithm for judgments."""

  @staticmethod
  def DrawOneSample(data, model, add_to=None):
    """Performs one step of the substitution sampling algorithm.

    Note:  This method updates the resolution maps in data with randomly
           sampled resolutions and updates model with randomly sampled
           parameters.

    Args:
      data:    A ResolverData structure.
      model:   A statistical model for the problem.  It must provide
               the methods SetSampleParameters and ResolveQuestion.
      add_to:  A list of length equal to data, with each element a
               resolution map (a Python dict).  If specified, the resolutions
               drawn will be added to this accumulator.  For example, if
               DrawOneSample is called with add_to equal to
                 [{'A': 1.0, 'B': 3.0},
                  {'A': 3.0, 'W': 2.0}]
               and we draw 'C' for question 1 and 'A' for question 2, then we
               update add_to to
                 [{'A': 1.0, 'B': 3.0, 'C': 1.0},
                  {'A': 4.0, 'W': 2.0}].
    """
    # Have the model draw sample parameters conditioned on the current
    # resolutions in the data object:
    model.SetSampleParameters(data)

    # Draw random resolutions based on the sampled parameters:
    for question, (responses, resolution_map) in data.iteritems():
      resolution_map.clear()
      # Get the model's estimated resolution distribution for this question:
      probabilistic_resolution_map = model.ResolveQuestion(responses)
      probability_values_list = probabilistic_resolution_map.values()
      # Now we terminate the list of probabilities with a "none of the above"
      # option.  This is because ResolveQuestion is allowed to return a map
      # that sums to less than 1.0, with the remainder representing "an
      # anonymous answer in the answer space but not seen in the judgments".
      # (ProbabilityCorrect is one example of a model that does this.)
      probability_values_list.append(0.0)
      # (numpy.random.multinomial will change this value to be 1.0 minus the
      #  sum of the other elments, so we don't need to do so ourselves.)
      # Now we draw a random index from the list:
      answer_index = next(itertools.compress(
          range(len(probability_values_list)),
          numpy.random.multinomial(1, probability_values_list)))
      if answer_index < len(probabilistic_resolution_map):
        # We drew a known answer; store it in resolution_map:
        answer = probabilistic_resolution_map.keys()[answer_index]
        resolution_map[answer] = 1.0
        if add_to:
          # Update add_to by adding the resolution to this question:
          assert question in add_to
          add_to[question][answer] = add_to[question].get(answer, 0.0) + 1.0
      # (else we leave resolution_map as the empty dict, representing choosing
      #  an anonymous answer.)

  @staticmethod
  def Integrate(data, model, number_of_samples=NUMBER_OF_SAMPLES,
                show_progress_bar=False):
    """Estimates answers by integration using number_of_samples samples.

    Args:
      data:               A ResolverData structure.
      model:              A statistical model for the problem.  It must provide
                          methods SetSampleParameters and ResolveQuestion.
      number_of_samples:  The number of samples to use for the integration.
      show_progress_bar:  Show a progress bar using pybar.
    """
    assert number_of_samples > 0

    # Draw number_of_samples samples of the resolution matrix:
    # Note that these samples come from a Markov chain, which is a sequential
    # process.  The "state" is stored in data as we go, and we accumulate the
    # integral's partial sums in resolution_sum.
    resolution_sum = dict([(question, {}) for question in data])
    for _ in range(number_of_samples):
      # Update parameters and draw one sample, clearing the resolution_map list
      # on the first iteration and adding to it on all subsequent iterations:
      SubstitutionSampling.DrawOneSample(data, model, add_to=resolution_sum)

    # The integral we seek is the mean of the sampled resolutions, so we must
    # divide by number_of_samples:
    for question, (_, resolution_map) in data.iteritems():
      resolution_map.clear()
      for key in resolution_sum[question]:
        # Any key in resolution_map should have a positive value:
        assert resolution_sum[question][key]
        # Normalize:
        resolution_map[key] = (
            resolution_sum[question][key] / float(number_of_samples))
