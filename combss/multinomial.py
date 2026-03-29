"""
combss.multinomial

Public module for best subset selection in multinomial logistic regression
via COMBSS.

Uses the Frank-Wolfe homotopy algorithm with Danskin's envelope gradient
and a warm-started sklearn L-BFGS-B inner solver for the baseline-category
multinomial model.
"""

import numpy as np
import combss._opt_glm as oglm


class model:
    """
    COMBSS model for best subset selection in multinomial logistic regression.

    Methods
    -------
    fit(X_train, y_train, ...)
        Run COMBSS to select the best subset of predictors.

    Attributes (available after fitting)
    ------------------------------------
    subset : ndarray
        Indices of the selected features (0-based).
    models : list
        List of selected subsets for k = 1, ..., q.
        Each entry is a sorted array of 1-indexed feature indices.
    run_time : float
        Execution time in seconds.
    lambda_ : float
        The ridge penalty parameter used.
    """

    def __init__(self):
        self.subset = None
        self.models = None
        self.run_time = None
        self.lambda_ = None

    def fit(self, X_train, y_train,
            q=None,
            C=None,
            Niter=25,
            lam_ridge=0,
            alpha=0.01,
            scale=True,
            verbose=True,
            mandatory_features=None,
            inner_tol=1e-4):
        """
        Fit the COMBSS model for multinomial logistic regression.

        Parameters
        ----------
        X_train : ndarray (n, p)
            Training design matrix (no intercept column).
        y_train : ndarray (n,)
            Class labels in {1, ..., C}.
        q : int, optional
            Maximum subset size. Defaults to min(n, p).
        C : int, optional
            Number of classes. If None, inferred from y_train.
        Niter : int
            Number of homotopy iterations (default 25).
        lam_ridge : float
            Ridge regularisation parameter for the inner solver (default 0).
        alpha : float
            Frank-Wolfe step size (default 0.01).
        scale : bool
            Column-normalise X before running (default True).
        verbose : bool
            Print progress (default True).
        mandatory_features : list or None
            1-indexed features to force into every model.
        inner_tol : float
            Inner solver convergence tolerance (default 1e-4).
        """
        import time

        n, p = X_train.shape
        if q is None:
            q = min(n, p)
        if C is None:
            C = len(np.unique(y_train))

        # Prepend intercept column
        X_fw = np.hstack([np.ones((n, 1)), X_train])

        tic = time.process_time()
        result = oglm.fw(
            X_fw, y_train,
            q=q,
            Niter=Niter,
            lam=lam_ridge,
            alpha=alpha,
            scale=scale,
            verbose=verbose,
            mandatory_features=mandatory_features,
            model_type='multinomial',
            C=C,
            inner_tol=inner_tol,
        )
        toc = time.process_time()

        self.models = result.models
        self.run_time = toc - tic
        self.lambda_ = lam_ridge

        # Set subset to the last (largest) model, 0-indexed
        if result.models:
            last_model = result.models[-1]
            self.subset = np.array(last_model) - 1

        return
