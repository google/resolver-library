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


import math

import scipy.special

import unittest
from resolver import digamma


class DigammaTest(unittest.TestCase):
  longMessage = True

  def testDigamma(self):
    # First, we'll test for positive values of x up to 1:
    for i in range(1, 10000):
      x = 1.0 / float(i)
      self.assertAlmostEqual(scipy.special.psi(x), digamma.Digamma(x),
                             places=10, msg='Unequal for x=' + str(x))
    # Now we'll test for positive values of x of order 1 and greater:
    for i in range(1, 10000):
      x = 0.982725 * float(i)  # (totally arbitrary)
      self.assertAlmostEqual(scipy.special.psi(x), digamma.Digamma(x),
                             places=10, msg='Unequal for x=' + str(x))
    # Finally, we'll test a large range of magnitudes of positive x:
    for i in range(-100, 100):
      x = math.exp(i)
      self.assertAlmostEqual(scipy.special.psi(x), digamma.Digamma(x),
                             places=10, msg='Unequal for x=' + str(x))


if __name__ == '__main__':
  unittest.main()
