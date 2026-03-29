"""
combss.linear

Public module for best subset selection in linear regression via COMBSS.

Provides two methods:
- ``method='glm'`` (default): Frank-Wolfe homotopy algorithm from the
  COMBSS-GLM framework (Mathur, Liquet, Muller & Moka, 2026). Uses a closed-form inner
  solver with Danskin's envelope gradient and auto-calibrated penalty schedule.
- ``method='original'``: Adam optimiser with dynamic lambda grid, as proposed
  in Moka et al. (2024). This is the implementation from combss versions <= 1.1.

Both methods are accessed through the ``model`` class.
"""

import numpy as np
import combss._opt_lm as olm
import combss._opt_glm as oglm


class model:
    """
    COMBSS model for best subset selection in linear regression.

    Methods
    -------
    fit(X_train, y_train, ...)
        Run COMBSS to select the best subset of predictors.

    Attributes (available after fitting)
    ------------------------------------
    subset : ndarray
        Indices of the selected features.
    coef_ : ndarray
        Regression coefficients for the selected subset.
    mse : float
        Mean squared error on validation data (method='original') or None (method='glm').
    lambda_ : float
        Optimal lambda value (method='original') or the lam parameter used (method='glm').
    run_time : float
        Execution time in seconds.
    models : list
        List of selected subsets for k = 1, ..., q (method='glm' only).
    subset_list : list
        List of subsets over the lambda grid (method='original' only).
    lambda_list : list
        Grid of lambda values (method='original' only).
    """

    def __init__(self):
        self.subset = None
        self.mse = None
        self.coef_ = None
        self.lambda_ = None
        self.run_time = None
        self.subset_list = None
        self.lambda_list = None
        self.models = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            q=None,
            method='glm',
            # --- GLM method parameters ---
            Niter=25,
            lam_ridge=0,
            alpha=0.01,
            scale=True,
            verbose=True,
            mandatory_features=None,
            inner_tol=1e-4,
            # --- Original method parameters ---
            nlam=50,
            t_init=[],
            scaling=True,
            tau=0.5,
            delta_frac=1,
            eta=0.001,
            patience=10,
            gd_maxiter=1000,
            gd_tol=1e-5,
            cg_maxiter=None,
            cg_tol=1e-5):
        """
        Fit the COMBSS model for linear regression.

        Parameters
        ----------
        X_train : ndarray (n_train, p)
            Training design matrix.
        y_train : ndarray (n_train,)
            Training response vector.
        X_val : ndarray (n_val, p), optional
            Validation design matrix. Required when method='original'.
        y_val : ndarray (n_val,), optional
            Validation response. Required when method='original'.
        q : int, optional
            Maximum subset size. Defaults to min(n, p).
        method : str
            ``'glm'`` (default) for the Frank-Wolfe homotopy algorithm, or
            ``'original'`` for the Adam + dynamic lambda grid method.

        GLM method parameters (method='glm')
        -------------------------------------
        Niter : int
            Number of homotopy iterations (default 25).
        lam_ridge : float
            Ridge regularisation parameter for the inner solver (default 0).
            This is NOT the sparsity penalty lambda used in the original method.
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

        Original method parameters (method='original')
        -----------------------------------------------
        nlam : int
            Number of lambda values in the dynamic grid (default 50).
        t_init : array-like
            Initial t vector (default centre of hypercube).
        scaling : bool
            Enable feature scaling (default True).
        tau : float
            Threshold parameter (default 0.5).
        delta_frac : float
            n/delta in the objective function (default 1).
        eta : float
            Truncation parameter (default 0.001).
        patience : int
            Early stopping rounds (default 10).
        gd_maxiter : int
            Maximum Adam iterations (default 1000).
        gd_tol : float
            Adam convergence tolerance (default 1e-5).
        cg_maxiter : int or None
            Conjugate gradient max iterations (default n_train).
        cg_tol : float
            Conjugate gradient tolerance (default 1e-5).
        """
        import time

        if method == 'glm':
            n, p = X_train.shape
            if q is None:
                q = min(n, p)

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
                model_type='linear',
                inner_tol=inner_tol,
            )
            toc = time.process_time()

            self.models = result.models
            self.run_time = toc - tic
            self.lambda_ = lam_ridge

            # Set subset to the last (largest) model
            if result.models:
                last_model = result.models[-1]
                # Convert 1-indexed to 0-indexed
                self.subset = np.array(last_model) - 1

        elif method == 'original':
            if X_val is None or y_val is None:
                raise ValueError("X_val and y_val are required when method='original'.")

            print("Fitting the model ...")
            result = olm.bss(X_train, y_train, X_val, y_val,
                             t_init=t_init,
                             q=q,
                             scaling=scaling,
                             tau=tau,
                             delta_frac=delta_frac,
                             nlam=nlam,
                             eta=eta,
                             patience=patience,
                             gd_maxiter=gd_maxiter,
                             gd_tol=gd_tol,
                             cg_maxiter=cg_maxiter,
                             cg_tol=cg_tol)
            print("Fitting is complete")
            self.subset = result["subset"]
            self.mse = result["mse"]
            self.coef_ = result["coef"]
            self.lambda_ = result["lambda"]
            self.run_time = result["time"]
            self.subset_list = result["subset_list"]
            self.lambda_list = result["lambda_list"]

        else:
            raise ValueError(f"Unknown method '{method}'. Use 'glm' or 'original'.")

        return
