"""
combss.linear

Public module for best subset selection in linear regression via COMBSS.

Provides two methods:
- ``method='fw'`` (default, Frank-Wolfe): homotopy algorithm from the
  COMBSS-GLM framework (Mathur, Liquet, Muller & Moka, 2026). Uses a closed-form inner
  solver with Danskin's envelope gradient and auto-calibrated penalty schedule.
- ``method='original'``: Adam optimiser with dynamic lambda grid, as proposed
  in Moka et al. (2024). This is the implementation from combss versions <= 1.1.

Both methods are accessed through the ``model`` class.
"""

import numpy as np
from numpy.linalg import pinv
import combss._opt_lm as olm
import combss._opt_glm as oglm


class model:
    """
    COMBSS model for best subset selection in linear regression.

    Methods
    -------
    fit(X_train, y_train, ...)
        Run COMBSS to select the best subset of predictors.

    Attributes (after fitting with method='fw', the Frank-Wolfe method)
    ---------------------------------------------
    subset : ndarray or None
        Indices of the best subset (0-indexed). Requires validation data.
    mse : float or None
        Validation MSE for the best subset. Requires validation data.
    coef_ : ndarray or None
        Regression coefficients (length p, zeros for unselected).
        Requires validation data.
    subset_list : list
        Subsets for k = 1, ..., q (0-indexed). May be shorter than q
        if early stopping was triggered.
    k_list : list
        Subset sizes evaluated. May be shorter than [1, ..., q]
        if early stopping was triggered.
    lam_ridge : float
        Ridge penalty used in the inner solver.

    Attributes (after fitting with method='original')
    --------------------------------------------------
    subset : ndarray
        Indices of the best subset (0-indexed).
    mse : float
        Validation MSE for the best subset.
    coef_ : ndarray
        Regression coefficients (length p, zeros for unselected).
    subset_list : list
        Subsets across the lambda grid (0-indexed).
    lambda_list : list
        Lambda grid values.
    lambda_ : float
        Optimal lambda value.
    """

    def __init__(self):
        pass

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            q=None,
            method='fw',
            # --- Frank-Wolfe method parameters ---
            Niter=25,
            lam_ridge=0,
            alpha=0.01,
            scale=True,
            verbose=True,
            mandatory_features=None,
            inner_tol=1e-4,
            patience=20,
            min_k=20,
            # --- Original method parameters ---
            nlam=50,
            t_init=[],
            scaling=True,
            tau=0.5,
            delta_frac=1,
            eta=0.001,
            gd_patience=10,
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
            Validation design matrix. Required for method='original'.
            Optional for method='fw'; when provided, the best subset
            is selected by validation MSE and coef_ are computed.
        y_val : ndarray (n_val,), optional
            Validation response. Required for method='original'.
        q : int, optional
            Maximum subset size. Defaults to min(n, p).
        method : str
            ``'fw'`` (default) for the Frank-Wolfe homotopy algorithm, or
            ``'original'`` for the Adam + dynamic lambda grid method.

        Frank-Wolfe method parameters (method='fw')
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
        patience : int
            Early stopping patience (default 20). When validation data is
            provided, stop if validation MSE has not improved for this many
            consecutive k values. Only active when X_val/y_val are given.
            Set to None to disable early stopping.
        min_k : int
            Minimum number of k values to evaluate before early stopping
            can trigger (default 20). Together with patience, the total
            k values evaluated is at least min(min_k + patience, q),
            capped so that min_k + patience <= p.

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
        gd_patience : int
            Patience for Adam termination (default 10).
        gd_maxiter : int
            Maximum Adam iterations (default 1000).
        gd_tol : float
            Adam convergence tolerance (default 1e-5).
        cg_maxiter : int or None
            Conjugate gradient max iterations (default n_train).
        cg_tol : float
            Conjugate gradient tolerance (default 1e-5).
        """

        if method == 'fw':
            n, p = X_train.shape
            if q is None:
                q = min(n, p)

            # Disable early stopping if min_k + patience exceeds q
            if patience is not None and min_k + patience > q:
                patience = None

            # Determine whether to use early stopping
            use_early_stop = (patience is not None
                              and X_val is not None
                              and y_val is not None)

            # Prepend intercept column
            X_fw = np.hstack([np.ones((n, 1)), X_train])

            if not use_early_stop:
                # Run all k = 1, ..., q at once
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
                self.subset_list = [np.array(m) - 1 for m in result.models]
                self.k_list = list(range(1, q + 1))

            else:
                # Run with early stopping: evaluate one k at a time
                # First run all k up to q (fw returns all at once)
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
                all_subsets = [np.array(m) - 1 for m in result.models]

                # Evaluate validation MSE incrementally with early stopping
                best_mse = np.inf
                no_improve_count = 0
                stop_k = q

                for k_idx in range(q):
                    sub = all_subsets[k_idx]
                    if len(sub) == 0:
                        mse_k = np.inf
                    else:
                        X_hat = X_train[:, sub]
                        beta_hat = pinv(X_hat.T @ X_hat) @ (X_hat.T @ y_train)
                        y_pred = X_val[:, sub] @ beta_hat
                        mse_k = np.mean((y_val - y_pred) ** 2)

                    if mse_k < best_mse:
                        best_mse = mse_k
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    # Check early stopping after min_k
                    if (k_idx + 1) >= min_k and no_improve_count >= patience:
                        stop_k = k_idx + 1
                        if verbose:
                            print(f"  Early stopping at k={stop_k} "
                                  f"(no improvement for {patience} steps)")
                        break

                self.subset_list = all_subsets[:stop_k]
                self.k_list = list(range(1, stop_k + 1))

            self.lam_ridge = lam_ridge

            # If validation data provided, find best k* by MSE
            self.subset = None
            self.mse = None
            self.coef_ = None
            if X_val is not None and y_val is not None:
                mse_list = []
                beta_list = []
                for sub in self.subset_list:
                    if len(sub) == 0:
                        mse_list.append(np.inf)
                        beta_list.append(np.zeros(p))
                        continue
                    X_hat = X_train[:, sub]
                    beta_hat = pinv(X_hat.T @ X_hat) @ (X_hat.T @ y_train)
                    y_pred = X_val[:, sub] @ beta_hat
                    mse_list.append(np.mean((y_val - y_pred) ** 2))
                    beta_pred = np.zeros(p)
                    beta_pred[sub] = beta_hat
                    beta_list.append(beta_pred)

                ind_opt = np.argmin(mse_list)
                self.subset = self.subset_list[ind_opt]
                self.mse = mse_list[ind_opt]
                self.coef_ = beta_list[ind_opt]

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
                             patience=gd_patience,
                             gd_maxiter=gd_maxiter,
                             gd_tol=gd_tol,
                             cg_maxiter=cg_maxiter,
                             cg_tol=cg_tol)
            print("Fitting is complete")
            self.subset = result["subset"]
            self.mse = result["mse"]
            self.coef_ = result["coef"]
            self.subset_list = result["subset_list"]
            self.lambda_list = result["lambda_list"]
            self.lambda_ = result["lambda"]

        else:
            raise ValueError(f"Unknown method '{method}'. Use 'fw' or 'original'.")

        return
