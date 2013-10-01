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

"""Gaussian contributor model for judgment resolution.

This model assumes that each contributor's judgments are drawn from a normal
distribution with bias and precision parameters, with an uninformative prior on
the answers themselves.

In this module, resolutions are Gaussian distributions specified by a mean
value and a variance.  Thus, the resolution map structure, instead of mapping
answers to probabilities, maps a key 'mean' to the value of the mean and a key
'variance' to the value of the variance.  So a well-specified resolution map
may look like this:
  resolution_map = {'mean': 9.2, 'variance': 0.7}

This module can raise the following exceptions:
  KeyError: if it is given resolutions not in the form above.
  TypeError: if it is given non-numeric judgments.

This module provides a SetSampleParameters() method for completeness only; it
is not compatible with substitution_sampling.py.
"""

import collections
import math

import numpy

from resolver import model


# Keys used in the resolution maps:
MEAN = 'mean'
VARIANCE = 'variance'

# A small value for symmetry-breaking in the model:
EPSILON = 1e-2

# A value for infinite precision (that is, zero variance) in a contributor:
INFINITY = numpy.inf


####################
# Module functions #
####################


def GaussianStatistics(data):
  """Given resolutions and judgments, returns MLE contributor parameters.

  These statistics, described in RCJ, summarize contributor behavior for the
  Gaussian model.
  - judgment_count is the number of judgments collected from a contributor.
  - mle_bias is the maximum-likelihood (also mean) bias estimate for a
    contributor.
  - sum_squared_deviation is, for each contributor, judgment_count times the
    maximum-likelihood precision estimate for that contributor.

  Args:
    data:  A ResolverData structure.

  Returns:
    3 maps from contributor to judgment count, MLE bias, and sum of squared
    deviations respectively.
  """

  # Iterate over data to count judgments:
  judgment_count = collections.defaultdict(float)
  difference = collections.defaultdict(float)
  for responses, resolution_map in data.itervalues():
    mean = resolution_map[MEAN]
    for contributor, judgment, _ in responses:
      judgment_count[contributor] += 1.0
      difference[contributor] += judgment - mean

  # Normalize and multiply by 1 - epsilon to get MLE contributor bias:
  mle_bias = {}  # eta_hat
  for contributor in difference:
    mle_bias[contributor] = (
        (1.0 - EPSILON) * difference[contributor] / judgment_count[contributor])

  # Iterate over data again to compute the sum of squares of deviations (the
  # numerator in the contributor's MLE precision):
  sum_squared_deviation = collections.defaultdict(float)
  for responses, resolution_map in data.itervalues():
    mean = resolution_map[MEAN]
    variance = resolution_map[VARIANCE]
    for contributor, judgment, _ in responses:
      delta = judgment - mean - mle_bias[contributor]
      sum_squared_deviation[contributor] += variance + delta * delta

  return judgment_count, mle_bias, sum_squared_deviation


def MLEGaussianParameters((judgment_count, mle_bias, sum_squared_deviation)):
  """Given the Gaussian statistics, returns MLE bias and precision.

  Args:
    (judgment_count, mle_bias, sum_squared_deviation):  Maps from contributor
      to number of judgments entered, maximum-likelihood bias, and sum of
      squared deviations, respectively.

  Returns:
    A map from contributor to maximum-likelihood bias and a map from
    contributor to maximum-likelihood precision.
  """
  assert (
      judgment_count.keys() == mle_bias.keys() == sum_squared_deviation.keys())
  # Compute precision from judgment_count and sum_squared_deviation:
  mle_precision = {}
  for contributor in judgment_count:
    if sum_squared_deviation[contributor] > 0.0:
      mle_precision[contributor] = (judgment_count[contributor] /
                                    sum_squared_deviation[contributor])
    else:
      mle_precision[contributor] = INFINITY

  return mle_bias, mle_precision


def SampleGaussianParameters((judgment_count, mle_bias, sum_squared_deviation)):
  """Given the Gaussian statistics, returns sampled bias and precision.

  Args:
    (judgment_count, mle_bias, sum_squared_deviation):  Maps from contributor
      to number of judgments entered, maximum-likelihood bias, and sum of
      squared deviations, respectively.

  Returns:
    A map from contributor to bias and a map from contributor to precision,
    drawn randomly from the normal-gamma distribution for these parameters.
  """
  assert (
      judgment_count.keys() == mle_bias.keys() == sum_squared_deviation.keys())
  bias = {}
  precision = {}

  for contributor in judgment_count:
    # Draw precision from a gamma distribution:
    k = 0.5 * (judgment_count[contributor] + 1.0)
    if sum_squared_deviation[contributor] > 0.0:
      theta = 2.0 / sum_squared_deviation[contributor]
      precision[contributor] = numpy.random.gamma(k, scale=theta)
      # Draw bias from a normal distribution using the just-drawn precision:
      scale = math.sqrt(1.0 /
                        (precision[contributor] * judgment_count[contributor]))
      bias[contributor] = numpy.random.normal(mle_bias[contributor],
                                              scale=scale)
    else:
      bias[contributor] = mle_bias[contributor]
      precision[contributor] = INFINITY

  return bias, precision


def VariationalGaussianParameters(
    (judgment_count, mle_bias, sum_squared_deviation)):
  """Given the Gaussian statistics, returns variational bias and precision.

  Args:
    (judgment_count, mle_bias, sum_squared_deviation):  Maps from contributor
      to number of judgments entered, maximum-likelihood bias, and sum of
      squared deviations, respectively.

  Returns:
    A map from contributor to maximum-likelihood bias and a map from
    contributor to variational precision.
  """
  assert (
      judgment_count.keys() == mle_bias.keys() == sum_squared_deviation.keys())

  variational_precision = {}
  for contributor in judgment_count:
    if sum_squared_deviation[contributor] > 0.0:
      variational_precision[contributor] = (
          (judgment_count[contributor] + 1.0) /
          sum_squared_deviation[contributor])
    else:
      variational_precision[contributor] = INFINITY

  return mle_bias, variational_precision


##############################
# GaussianContributors class #
##############################


class GaussianContributors(model.StatisticalModel):
  """Models rater behavior using one Gaussian distribution per contributor."""

  @staticmethod
  def InitializeResolutions(data,
                            overwrite_all_resolutions=False):
    """Stores initial resolution guesses in a ResolverData object.

    The initial resolution guesses are simply normal distributions fitted to
    the judgments for each question.

    Args:
      data:  A ResolverData object.
      overwrite_all_resolutions:  Same as in model.StatisticalModel.
        - If True, all questions get a resolution.
        - If False, existing resolutions in the data object are preserved.
    """
    for responses, resolution_map in data.itervalues():
      if overwrite_all_resolutions or not resolution_map:
        numeric_judgments = [judgment for _, judgment, _ in responses]
        resolution_map.clear()
        resolution_map[MEAN] = numpy.mean(numeric_judgments)
        resolution_map[VARIANCE] = numpy.var(numeric_judgments)

  def SetMLEParameters(self, data):
    """Given resolutions and judgments, updates self with parameter MLEs.

    Args:
      data:  A ResolverData object with Gaussian resolutions.
    """
    self.bias, self.precision = MLEGaussianParameters(GaussianStatistics(data))

  def SetSampleParameters(self, data):
    """Given resolutions and judgments, updates self with sampled parameters.

    Args:
      data:  A ResolverData object with Gaussian resolutions.
    """
    self.bias, self.precision = SampleGaussianParameters(
        GaussianStatistics(data))

  def SetVariationalParameters(self, data):
    """Given resolutions and judgments, updates self with variational eta/sigma.

    Args:
      data:  A ResolverData object with Gaussian resolutions.
    """
    self.bias, self.precision = VariationalGaussianParameters(
        GaussianStatistics(data))

  def ResolveQuestion(self, responses):
    """Returns one resolution map based on judgments and current parameters.

    Args:
      responses:  A list of (contributor, judgment, metadata) tuples for this
                  question.

    Returns:
      A map from answers to probabilities.
    """
    mean = 0.0
    variance = 1.0 / sum([self.precision[contributor]
                          for contributor, _, _ in responses])
    mean = variance * sum(
        [(judgment - self.bias[contributor]) * self.precision[contributor]
         for contributor, judgment, _ in responses])
    if numpy.isnan(mean):
      # At least one contributor in responses has infinite precision, so just
      # take the mean and variance of the set of judgments from such
      # contributors:
      judgments = [judgment - self.bias[contributor]
                   for contributor, judgment, _ in responses
                   if numpy.isinf(self.precision[contributor])]
      mean = numpy.mean(judgments)
      variance = numpy.var(judgments)

    return {MEAN: mean, VARIANCE: variance}
