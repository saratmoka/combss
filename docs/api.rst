API Reference
=============

.. contents:: Modules
   :local:

combss.linear
-------------

Best subset selection for linear regression.

Two methods are available:

- **GLM method** (``method='glm'``, default): Frank-Wolfe homotopy algorithm.
  Sparsity is controlled by ``k`` (model size): COMBSS returns selected
  features for each ``k = 1, ..., q``.  The ``lam_ridge`` parameter is an
  optional ridge regularisation on the coefficients in the inner solver.

- **Original method** (``method='original'``): Adam optimiser with a dynamic
  lambda grid, as proposed in Moka et al. (2024).  Sparsity is controlled by
  ``lambda``: a grid of lambda values is searched, and each lambda yields a
  different subset.  The best subset is selected by validation MSE.

.. autoclass:: combss.linear.model
   :members:
   :undoc-members:

combss.logistic
---------------

Best subset selection for binary logistic regression.

Uses the Frank-Wolfe homotopy algorithm with Danskin's envelope gradient and
a warm-started sklearn L-BFGS-B inner solver.

Labels ``y`` must be binary ``{0, 1}``.

.. autoclass:: combss.logistic.model
   :members:
   :undoc-members:

combss.multinomial
------------------

Best subset selection for multinomial logistic regression.

Uses the Frank-Wolfe homotopy algorithm with a baseline-category multinomial
model.  Labels ``y`` must be in ``{1, ..., C}``.

.. autoclass:: combss.multinomial.model
   :members:
   :undoc-members:

combss.cv
---------

Leave-one-out cross-validation for selecting the ridge penalty
``lam_ridge`` in the COMBSS Frank-Wolfe algorithm.

Note: the ``lambda_grid`` in this module contains **ridge penalty** values,
not the sparsity penalty lambda used in the original COMBSS method.

.. autofunction:: combss.cv.cv_select_lambda

.. autofunction:: combss.cv.loocv_mse_linear

.. autofunction:: combss.cv.loocv_accuracy

combss.metrics
--------------

Performance metrics for evaluating variable selection.

.. autofunction:: combss.metrics.performance_metrics

.. autofunction:: combss.metrics.binary_confusion_matrix
