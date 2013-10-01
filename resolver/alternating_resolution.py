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

"""Alternating-expectation resolution algorithms:  EM and variational Bayes."""

__author__ = 'tpw@google.com (Tamsyn Waterhouse)'


# The maximum number of steps to take, by default:
MAX_ITERATIONS = 30
# If the maximum elementwise difference between resolutions at two successive
# steps is less than EPSILON, then we say we have convergence:
EPSILON = 1e-15


###################################
# The AlternatingResolution class #
###################################


class AlternatingResolution(object):
  """Implements alternating-expectation algorithms for collective judgment."""

  @staticmethod
  def UpdateModel(unused_data, unused_model):
    """Placeholder for the method that performs the maximization step.

    Subclasses should define this method.
    """
    raise NotImplementedError

  @classmethod
  def IterateOnce(cls, data, model):
    """Performs one step of an alternating algorithm.

    Args:
      data:   A ResolverData structure.
      model:  A statistical model for the problem.  It must provide
              ResolveQuestion and any methods needed by UpdateModel.

    Returns:
      The sum-of-resolution-distance-squared between the old and new
      resolution maps.
    """
    # Set model parameters:
    cls.UpdateModel(data, model)
    # Use the parameters to estimate a resolution for each question:
    resolution_delta = 0.0
    for responses, resolution_map in data.itervalues():
      new_resolution_map = model.ResolveQuestion(responses)
      resolution_delta += model.ResolutionDistanceSquared(resolution_map,
                                                          new_resolution_map)
      resolution_map.clear()
      resolution_map.update(new_resolution_map)
    # Return the difference between the previous resolution and the new one:
    return resolution_delta

  @classmethod
  def IterateUntilConvergence(cls, data, model, max_iterations=MAX_ITERATIONS):
    """Calls IterateOnce until convergence or a fixed number of steps.

    Args:
      data:            A ResolverData structure.
      model:           A statistical model for the problem.  It must provide
                       ResolveQuestion and any methods needed by UpdateModel.
      max_iterations:  The maximum number of iterations to perform.

    Returns:
      True if the algorithm converged (with tolerance EPSILON) within
      max_iterations steps, and False otherwise.
    """
    for _ in range(max_iterations):
      if cls.IterateOnce(data, model) < EPSILON:
        return True
    return False


#####################################
# The ExpectationMaximization class #
#####################################


class ExpectationMaximization(AlternatingResolution):
  """Implements the EM algorithm for judgments, as in Dawid and Skene (1979)."""

  @staticmethod
  def UpdateModel(data, model):
    """Call's model's SetMLEParameters method.

    Args:
      data:   A ResolverData structure.
      model:  A statistical model for the problem.  It must provide the method
              SetMLEParameters.
    """
    model.SetMLEParameters(data)


##############################
# The VariationalBayes class #
##############################


class VariationalBayes(AlternatingResolution):
  """Implements variational Bayesian inference, as in Simpson et al. (2011)."""

  @staticmethod
  def UpdateModel(data, model):
    """Call's model's SetVariationalParameters method.

    Args:
      data:   A ResolverData structure.
      model:  A statistical model for the problem.  It must provide the method
              SetVariationalParameters.
    """
    model.SetVariationalParameters(data)
