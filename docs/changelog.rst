Changelog
=========

Version 2.0.0
-------------

- Added binary logistic regression (``combss.logistic``).
- Added multinomial logistic regression (``combss.multinomial``).
- Added Frank-Wolfe homotopy algorithm as the new default method for linear
  regression (``combss.linear.model.fit(method='glm')``).
- Added LOOCV-based ridge penalty selection (``combss.cv``).
- Original Adam + dynamic-lambda method remains available via
  ``combss.linear.model.fit(method='original')``.
- Added ``scikit-learn`` and ``pandas`` as dependencies.

Version 1.1.4
-------------

- Linear regression via Adam optimiser with dynamic lambda grid.
- Performance metrics for variable selection evaluation.
