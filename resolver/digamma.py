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

"""The digamma function (the logarithmic derivative of the gamma function).

For simplicity, we here implement it for only positive argument values.

This implementation is based on SciPy's and uses the series expansion
                            infty  B
                        1    ---    2n
  digamma(x) = ln(x) - --- -  >  ------
                       2 x   ---     2n
                             n=1 2n x
where B_i is the ith Bernoulli number.
"""

import math


# We'll need the first few even-numbered Bernoulli numbers:
B_2 = 1.0 / 6.0
B_4 = -1.0 / 30.0
B_6 = 1.0 / 42.0
B_8 = -1.0 / 30.0
B_10 = -5.0 / 66.0
B_12 = -691.0 / 2730.0
B_14 = 7.0 / 6.0

# We use them in the form B_2n / 2n in the series above:
A = [B_2 / 2.0,
     B_4 / 4.0,
     B_6 / 6.0,
     B_8 / 8.0,
     B_10 / 10.0,
     B_12 / 12.0,
     B_14 / 14.0]


def Digamma(x):
  """Returns the digamma function evaluated at x.

  Args:
    x:  A positive real value.

  Returns:
    digamma(x)
  """
  assert x > 0.0
  series = 0.0
  harmonic = 0.0
  if x < 1.0e17:  # (For large x, the series goes to 0.0 so we can skip this.)
    while x < 10.0:
      # To use the series expansion, we need x to be large enough that the
      # series converges within the six terms we limit ourselves to.  Thus, if
      # x < 10.0, we start by using the relation
      #    digamma(x + 1) = digamma(x) + 1/x
      # to increment x up to 10.0, summing the harmonic parts 1.0 / x as we go:
      harmonic += 1.0 / x
      x += 1.0
    # Now we'll sum the first few terms of the series B_2n / (2 * n * x^2n).
    # We'll do it using the form
    #   B_2 + (B_4 + (B_6 + (B_8 +
    #       (B_10 + (B_12 + (B_14 / x) / x) / x) / x) / x) / x) / x
    # and calculating from the inside out.  This is computationally optimal
    # because it minimizes the number of operations and numerically optimal
    # because it sums the small contributions first and then the large ones.
    x_squared = x * x
    for coefficient in reversed(A):
      series += coefficient
      series /= x_squared
  return math.log(x) - (0.5 / x) - series - harmonic
