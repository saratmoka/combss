"""
combss.cv

Leave-one-out cross-validation for selecting the ridge penalty lambda
in the COMBSS Frank-Wolfe algorithm.

Workflow
--------
For each candidate lambda in a grid:
  1. Run COMBSS on the full training data -> selected features S_k for k = 1..q.
  2. For each k, evaluate S_k via LOOCV on the refit step:
       Classification  -- refit LogisticRegression on X[~i, S_k], predict X[i, S_k]
       Linear          -- exact hat-matrix shortcut (no per-fold loop required)
  3. LOOCV score for (lambda, k):
       Classification  -- loocv_acc  (fraction correct; higher is better)
       Linear          -- loocv_mse  (mean squared error; lower is better)

The best lambda is reported both overall (optimising mean LOOCV score across k)
and per-k (optimising LOOCV score for each individual k).

Note: feature *selection* uses the full dataset; only the final refit is
cross-validated. This is consistent with standard practice in
best-subset-selection literature.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import combss._opt_glm as oglm


# ================================================================
# Default lambda grid
# ================================================================

DEFAULT_LAMBDA_GRID = np.concatenate([[0.0], np.logspace(-3, 1, 10)])


# ================================================================
# LOOCV MSE for linear regression (exact hat-matrix shortcut)
# ================================================================

def loocv_mse_linear(X_sel, y, lambda_refit=0.0):
    """
    Exact leave-one-out MSE for linear regression via the hat-matrix identity.

    Uses the formula: e_i^{LOO} = (y_i - y_hat_i) / (1 - h_ii)
    where h_ii are the diagonal entries of the hat matrix.

    Cost O(n k^2) -- no per-fold loop required.

    Parameters
    ----------
    X_sel : ndarray (n, k)
        Columns restricted to selected features.
    y : ndarray (n,)
        Continuous response.
    lambda_refit : float
        Ridge penalty on slopes (0 = OLS); intercept is never penalised.

    Returns
    -------
    float : LOOCV mean squared error.
    """
    n = len(y)
    k = X_sel.shape[1]

    A = np.hstack([np.ones((n, 1)), X_sel])

    pen = np.zeros((k + 1, k + 1))
    if lambda_refit > 0.0:
        np.fill_diagonal(pen, 2.0 * lambda_refit * n)
        pen[0, 0] = 0.0

    M = np.linalg.solve(A.T @ A + pen, A.T)
    y_hat = A @ (M @ y)
    h_diag = np.einsum('ij,ji->i', A, M)

    residuals = y - y_hat
    denom = np.clip(1.0 - h_diag, 1e-10, None)
    return float(np.mean((residuals / denom) ** 2))


# ================================================================
# LOOCV accuracy for classification (logit / multinomial)
# ================================================================

def loocv_accuracy(X_sel, y, lambda_refit=0.0):
    """
    Leave-one-out cross-validation accuracy for classification.

    Parameters
    ----------
    X_sel : ndarray (n, k)
        Columns restricted to selected features.
    y : ndarray (n,)
        Class labels.
    lambda_refit : float
        Ridge penalty for the refit model (0 = unpenalised).

    Returns
    -------
    float : fraction of correctly predicted held-out labels.
    """
    n = len(y)
    classes = np.unique(y)
    n_correct = 0
    n_valid = 0

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        X_tr = X_sel[mask]
        y_tr = y[mask]
        X_te = X_sel[i:i + 1]
        y_te = y[i]

        if len(np.unique(y_tr)) < len(classes):
            continue

        if lambda_refit > 0:
            clf = LogisticRegression(
                penalty='l2', C=1.0 / (2.0 * lambda_refit),
                solver='lbfgs', max_iter=5000,
            )
        else:
            clf = LogisticRegression(
                penalty=None, solver='lbfgs', max_iter=5000,
            )

        try:
            clf.fit(X_tr, y_tr)
            n_correct += int(clf.predict(X_te)[0] == y_te)
            n_valid += 1
        except Exception:
            pass

    return n_correct / n_valid if n_valid > 0 else float('nan')


# ================================================================
# Main CV function
# ================================================================

def cv_select_lambda(X, y, q, C=None,
                     lambda_grid=None,
                     Niter=50,
                     alpha=0.01,
                     model_type='multinomial',
                     inner_tol=1e-4,
                     lambda_refit=0.0,
                     verbose=True):
    """
    Select the ridge regularisation penalty via leave-one-out cross-validation.

    For each candidate ridge penalty in lambda_grid, COMBSS selects features
    for k = 1..q on the full data.  Each selected model is then evaluated by
    LOOCV on the refit step.

    Note: this lambda is a ridge penalty on the coefficients in the inner
    solver (lam_ridge in the model classes), NOT the sparsity penalty used
    in the original COMBSS method.

    Parameters
    ----------
    X : ndarray (n, p)
        Feature matrix (no intercept column).
    y : ndarray (n,)
        Response / labels:
        - {1, ..., C} for multinomial
        - {0, 1} for logit
        - real-valued for linear
    q : int
        Maximum subset size.
    C : int or None
        Number of classes (required for logit/multinomial; ignored for linear).
    lambda_grid : array-like or None
        Candidate ridge penalty values (default: [0] + logspace(-3, 1, 10)).
    Niter : int
        COMBSS homotopy iterations (default 50).
    alpha : float
        Frank-Wolfe step size (default 0.01).
    model_type : str
        ``'multinomial'``, ``'logit'``, or ``'linear'``.
    inner_tol : float
        Inner solver tolerance (default 1e-4).
    lambda_refit : float
        Ridge penalty for LOOCV refit (default 0).
    verbose : bool
        Print progress (default True).

    Returns
    -------
    best_lambda : float
        Lambda maximising mean LOOCV accuracy (classification) or
        minimising mean LOOCV MSE (linear).
    best_lambda_per_k : dict
        Best lambda for each k individually.
    results_df : DataFrame
        Columns: lambda, k, loocv_acc/loocv_mse, selected.
    """
    if model_type in ('logit', 'multinomial') and C is None:
        raise ValueError("C (number of classes) is required for logit/multinomial.")

    if lambda_grid is None:
        lambda_grid = DEFAULT_LAMBDA_GRID
    lambda_grid = np.asarray(lambda_grid, dtype=float)

    is_linear = (model_type == 'linear')
    score_col = 'loocv_mse' if is_linear else 'loocv_acc'
    score_label = 'LOOCV MSE' if is_linear else 'LOOCV acc'

    n, p = X.shape
    X_fw = np.hstack([np.ones((n, 1)), X])

    rows = []

    for lam in lambda_grid:
        if verbose:
            print(f"\n{'=' * 55}")
            print(f"lambda = {lam:.5g}")
            print(f"{'=' * 55}")

        result = oglm.fw(
            X_fw, y,
            q=q,
            Niter=Niter,
            lam=lam,
            alpha=alpha,
            scale=True,
            model_type=model_type,
            C=C if C is not None else 2,
            inner_tol=inner_tol,
            verbose=False,
        )

        if verbose:
            print(f"  {'k':>3}  {score_label:>10}  selected features")
            print(f"  {'-' * 50}")

        for k in range(1, q + 1):
            feat_1idx = result.models[k - 1]
            feat_0idx = np.asarray(feat_1idx) - 1

            X_sel = X[:, feat_0idx]

            if is_linear:
                score = loocv_mse_linear(X_sel, y, lambda_refit=lambda_refit)
            else:
                score = loocv_accuracy(X_sel, y, lambda_refit=lambda_refit)

            rows.append({
                'lambda': lam,
                'k': k,
                score_col: score,
                'selected': sorted(feat_1idx.tolist()),
            })

            if verbose:
                print(f"  {k:>3}  {score:>10.4f}  {sorted(feat_1idx.tolist())}")

    results_df = pd.DataFrame(rows)

    if is_linear:
        best_lambda_per_k = {}
        for k in range(1, q + 1):
            sub = results_df[results_df['k'] == k]
            best_lambda_per_k[k] = float(sub.loc[sub[score_col].idxmin(), 'lambda'])
        mean_score = results_df.groupby('lambda')[score_col].mean()
        best_lambda = float(mean_score.idxmin())
    else:
        best_lambda_per_k = {}
        for k in range(1, q + 1):
            sub = results_df[results_df['k'] == k]
            best_lambda_per_k[k] = float(sub.loc[sub[score_col].idxmax(), 'lambda'])
        mean_score = results_df.groupby('lambda')[score_col].mean()
        best_lambda = float(mean_score.idxmax())

    if verbose:
        print(f"\n{'=' * 55}")
        print("Best lambda per k:")
        for k, lam in best_lambda_per_k.items():
            row = results_df[(results_df['lambda'] == lam) & (results_df['k'] == k)].iloc[0]
            print(f"  k={k:>2}  lambda={lam:.5g}  {score_label}={row[score_col]:.4f}"
                  f"  selected={row['selected']}")
        print(f"\nBest lambda overall (mean across k): {best_lambda:.5g}")
        print(f"{'=' * 55}")

    return best_lambda, best_lambda_per_k, results_df
