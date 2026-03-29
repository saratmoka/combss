"""
combss.logistic

Public module for best subset selection in binary logistic regression via COMBSS.

Uses the Frank-Wolfe homotopy algorithm with Danskin's envelope gradient
and a warm-started sklearn L-BFGS-B inner solver.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import combss._opt_glm as oglm


class model:
    """
    COMBSS model for best subset selection in binary logistic regression.

    Methods
    -------
    fit(X_train, y_train, ...)
        Run COMBSS to select the best subset of predictors.

    Attributes (available after fitting)
    ------------------------------------
    subset : ndarray or None
        Indices of the best subset (0-indexed). Requires validation data.
    accuracy : float or None
        Validation accuracy for the best subset. Requires validation data.
    coef_ : ndarray or None
        Logistic regression coefficients (length p, zeros for unselected).
        Requires validation data.
    lam_ridge : float
        Ridge penalty used in the inner solver.
    subset_list : list
        Sequence of subsets for k = 1, ..., q (0-indexed).
    k_list : list
        [1, 2, ..., q].
    """

    def __init__(self):
        pass

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            q=None,
            Niter=25,
            lam_ridge=0,
            alpha=0.01,
            scale=True,
            verbose=True,
            mandatory_features=None,
            inner_tol=1e-4):
        """
        Fit the COMBSS model for binary logistic regression.

        Parameters
        ----------
        X_train : ndarray (n, p)
            Training design matrix (no intercept column).
        y_train : ndarray (n,)
            Binary labels {0, 1}.
        X_val : ndarray (n_val, p), optional
            Validation design matrix. When provided, the best subset
            is selected by validation accuracy and coef_ are computed.
        y_val : ndarray (n_val,), optional
            Validation labels.
        q : int, optional
            Maximum subset size. Defaults to min(n, p).
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

        n, p = X_train.shape
        if q is None:
            q = min(n, p)

        # Prepend intercept column
        X_fw = np.hstack([np.ones((n, 1)), X_train])

        result = oglm.fw(
            X_fw, y_train,
            q=q,
            Niter=Niter,
            lam=lam_ridge,
            alpha=alpha,
            scale=scale,
            verbose=verbose,
            mandatory_features=mandatory_features,
            model_type='logit',
            C=2,
            inner_tol=inner_tol,
        )

        # Convert 1-indexed models to 0-indexed subset_list
        self.subset_list = [np.array(m) - 1 for m in result.models]
        self.k_list = list(range(1, q + 1))
        self.lam_ridge = lam_ridge

        # If validation data provided, find best k* by accuracy
        self.subset = None
        self.accuracy = None
        self.coef_ = None
        if X_val is not None and y_val is not None:
            acc_list = []
            coef_list = []
            for sub in self.subset_list:
                if len(sub) == 0:
                    acc_list.append(0.0)
                    coef_list.append(np.zeros(p))
                    continue
                clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
                clf.fit(X_train[:, sub], y_train)
                acc = np.mean(clf.predict(X_val[:, sub]) == y_val)
                acc_list.append(acc)
                coef_pred = np.zeros(p)
                coef_pred[sub] = clf.coef_.ravel()
                coef_list.append(coef_pred)

            ind_opt = np.argmax(acc_list)
            self.subset = self.subset_list[ind_opt]
            self.accuracy = acc_list[ind_opt]
            self.coef_ = coef_list[ind_opt]

        return
