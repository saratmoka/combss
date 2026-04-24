Algorithm Overview
==================

COMBSS (Continuous Optimisation for Best Subset Selection) reformulates the
discrete combinatorial problem of selecting the best :math:`k` predictors as a
continuous optimisation over the hypercube :math:`[0,1]^p` via a Boolean
relaxation.

Frank-Wolfe method (method='fw')
---------------------------------

The Frank-Wolfe homotopy algorithm (Algorithm 1 in the paper) operates as
follows:

1. **Initialise** :math:`t = (k/p, \ldots, k/p)` --- centroid of the
   :math:`k`-sparse simplex :math:`T_k`.

2. **For** :math:`i = 1, \ldots, N` (homotopy loop):

   - Set :math:`\delta_i` on a geometric schedule from :math:`\delta_{\min}`
     to :math:`\delta_{\max}`.
   - Solve the ridge-penalised GLM inner problem at current :math:`t`.
   - Compute the Danskin gradient --- no Hessian required.
   - Find the Frank-Wolfe vertex: :math:`s = \arg\min_{s \in T_k}
     \langle \nabla f(t), s \rangle` (the :math:`k` smallest gradient
     components).
   - Update: :math:`t \leftarrow (1 - \alpha) t + \alpha s`.

3. **Select** the :math:`k` features with the largest final :math:`t` values.

Key properties:

- **Gradient via Danskin's envelope theorem**: requires only a single
  ridge-penalised GLM solve per iteration --- no Hessian computation.
- **Auto-calibrated penalty schedule**: :math:`\delta_{\min}` and
  :math:`\delta_{\max}` are set from :math:`\lambda_{\max}(X^T X)` estimated
  by power iteration, so no manual tuning is needed.
- **Scalable to** :math:`p \gg n` via the Woodbury identity (linear) or
  warm-started L-BFGS-B (logistic, multinomial).

Inner solvers by model type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Linear regression**: closed-form solution with Woodbury identity when
  :math:`p > n` for :math:`O(n^2 p)` complexity instead of :math:`O(p^3)`.
- **Binary logistic regression**: sklearn ``LogisticRegression`` with L-BFGS-B
  via feature scaling to convert weighted ridge into standard ridge.
- **Multinomial logistic regression**: sklearn multinomial L-BFGS-B with
  baseline-category parameterisation.  Gradient uses all :math:`C` class
  coefficients to match the symmetric objective.

Original method: Adam with dynamic lambda grid
-----------------------------------------------

The original COMBSS method for linear regression (Moka et al. 2024) uses:

1. **Adam optimiser** to minimise the COMBSS objective function for a fixed
   penalty parameter :math:`\lambda`.
2. **Dynamic lambda grid**: starting from
   :math:`\lambda_{\max} = y^T y / n`, :math:`\lambda` is halved until a
   subset of size :math:`\geq q` is found.  A second pass refines the grid by
   bisecting between adjacent lambdas that produced different subset sizes.
3. **Validation**: the best subset is selected from the grid by minimising MSE
   on a held-out validation set.

Note on the two lambdas
^^^^^^^^^^^^^^^^^^^^^^^^

The parameter called "lambda" has different meanings in the two methods:

- **Original method**: :math:`\lambda` is the **sparsity penalty** in the
  COMBSS objective.  A grid of :math:`\lambda` values is searched, and each
  :math:`\lambda` yields a different subset.
- **Frank-Wolfe method**: ``lam_ridge`` is a **ridge regularisation penalty** on the
  coefficients in the inner solver.  Sparsity is controlled by :math:`k`
  (the model size), not by lambda.  This parameter is typically 0 or small.

Intercept handling
------------------

In the Frank-Wolfe method, the intercept is handled internally and is **not** subject
to selection.  An intercept column is prepended automatically by the model
classes.

In the original method, the intercept (if included) is treated the same as
other features.
