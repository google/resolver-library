A library for using Bayesian inference to turn judgments from (presumably human) contributors into estimated answers to questions, a process we call "resolution".

A resolution algorithm requires two parts:  A statistical model that uses some set of parameters to predict contributor behavior, and an inference method that makes use of such a model to perform statistical inference.

This project includes four models (confusion matrices, probability-correct, Gaussian contributors, and decision trees) and three inference methods (expectation-maximization, substitution sampling, and variational inference).

It additionally provides methods for computing the information-theoretic contributor performance quantities described in "Pay by the Bit: An Information-Theoretic Metric for Collective Human Judgment" (http://research.google.com/pubs/pub40700.html).

The library depends on NumPy, which must be installed on your system.  The unit and integration tests depend additionally on SciPy.

A paper on the theory behind the code is currently in preparation.

Note that this is a work in progress, and both data structures and interfaces may change in future versions.