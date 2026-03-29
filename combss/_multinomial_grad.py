"""
combss._multinomial_grad.py

Private module containing the multinomial logistic regression inner solver
and gradient computation for the Frank-Wolfe COMBSS algorithm.

Adapted from the COMBSS-GLM project (Mathur, Liquet, Muller & Moka, 2026).

Functions:
- softmax_baseline(): Baseline-category softmax probabilities.
- multinom_loglik_baseline(): Log-likelihood for baseline-multinomial model.
- alpha_nm(): Non-mandatory row weights in Xi-parameterization.
- g_value(): Full GLM objective g_{delta,lambda}(t, xi0, Xi).
- grad_f_analytic(): Analytic envelope gradient via Danskin's theorem.
- move_mandatory_to_front(): Column permutation for mandatory features.
- solve_inner_g_sklearn(): Inner solver using sklearn multinomial ridge.
- grad_f_t_multinomial_fast(): Combined gradient + inner solve for use in FW loop.
"""

import numpy as np


# ================================================================
# Baseline-multinomial model + GLM objective pieces
# ================================================================

def softmax_baseline(scores):
    """
    Compute softmax probabilities with implicit baseline class.

    Parameters
    ----------
    scores : ndarray (n, C-1)
        Logits for classes 1..C-1; baseline class C has logit 0.

    Returns
    -------
    P : ndarray (n, C-1)
        Probabilities for classes 1..C-1.
    """
    z = scores - np.max(scores, axis=1, keepdims=True)
    ez = np.exp(z)
    denom = 1.0 + np.sum(ez, axis=1, keepdims=True)
    return ez / denom


def multinom_loglik_baseline(xi0, Xi, X, y, C):
    """
    Log-likelihood for baseline-multinomial with classes {1,...,C}, baseline C.

    Parameters
    ----------
    xi0 : ndarray (C-1,)
        Intercept for each non-baseline class.
    Xi : ndarray (p, C-1)
        Coefficient matrix.
    X : ndarray (n, p)
        Design matrix.
    y : ndarray (n,)
        Labels in {1, ..., C}.
    C : int
        Number of classes.

    Returns
    -------
    float : log-likelihood value.
    """
    n, p = X.shape
    K = C - 1
    scores = X @ Xi + xi0[None, :]

    max_s = np.max(scores, axis=1)
    z = scores - max_s[:, None]
    log_denom = np.log(1.0 + np.sum(np.exp(z), axis=1)) + max_s

    ell = -np.sum(log_denom)
    mask = (y >= 1) & (y <= K)
    idx = np.where(mask)[0]
    ell += np.sum(scores[idx, y[idx] - 1])
    return float(ell)


def alpha_nm(t, lam, delta, eps=1e-12):
    """
    Non-mandatory row weights in Xi-parameterization.

    alpha(t) = (lam + delta) / t^2 - delta

    Parameters
    ----------
    t : ndarray
        Weight vector for optional features.
    lam : float
        Ridge penalty parameter.
    delta : float
        Current penalty schedule value.
    eps : float
        Clamping floor for t.

    Returns
    -------
    ndarray : weight vector alpha(t).
    """
    tj = np.maximum(np.asarray(t, float), eps)
    return (lam + delta) / (tj ** 2) - delta


def g_value(t, xi0, Xi, X, y, C, m, lam, delta, eps=1e-12):
    """
    Evaluate the GLM objective g_{delta,lambda}(t, xi0, Xi).

    Parameters
    ----------
    t : ndarray (p-m,)
        Weights for optional features.
    xi0 : ndarray (C-1,)
        Intercept parameters.
    Xi : ndarray (p, C-1)
        Coefficient matrix.
    X : ndarray (n, p)
        Design matrix (mandatory features first).
    y : ndarray (n,)
        Labels in {1, ..., C}.
    C : int
        Number of classes.
    m : int
        Number of mandatory features.
    lam : float
        Ridge penalty.
    delta : float
        Penalty schedule value.
    eps : float
        Clamping floor.

    Returns
    -------
    float : objective value.
    """
    n, p = X.shape
    K = C - 1
    assert Xi.shape == (p, K)
    assert xi0.shape == (K,)
    assert t.shape == (p - m,)

    ell = multinom_loglik_baseline(xi0, Xi, X, y, C)
    nll = -ell / n
    a = alpha_nm(t, lam, delta, eps=eps)

    pen = lam * np.sum(Xi[:m, :] ** 2) + np.sum(a[:, None] * (Xi[m:, :] ** 2))
    return float(nll + pen)


def grad_f_analytic(t, Xi_hat, m, lam, delta, eps=1e-12):
    """
    Envelope gradient at interior t in Xi-parameterization.

    Only the penalty depends on t, so the gradient is:
        (nabla f(t))_j = -2(lam + delta) ||Xi_{m+j,:}||^2 / t_j^3

    Parameters
    ----------
    t : ndarray (p-m,)
        Weights for optional features.
    Xi_hat : ndarray (p, C-1)
        Coefficient matrix from inner solve.
    m : int
        Number of mandatory features.
    lam : float
        Ridge penalty.
    delta : float
        Penalty schedule value.
    eps : float
        Clamping floor.

    Returns
    -------
    ndarray (p-m,) : gradient with respect to optional feature weights.
    """
    tj = np.maximum(np.asarray(t, float), eps)
    rownorm2 = np.sum(Xi_hat[m:, :] ** 2, axis=1)
    return -2.0 * (lam + delta) * rownorm2 / (tj ** 3)


# ================================================================
# Mandatory-feature permutation helpers
# ================================================================

def move_mandatory_to_front(X, M):
    """
    Permute columns so mandatory set M (0-based) comes first.

    Parameters
    ----------
    X : ndarray (n, p)
        Design matrix.
    M : array-like of int
        0-based indices of mandatory features.

    Returns
    -------
    Xp : ndarray (n, p)
        Column-permuted design matrix.
    perm : ndarray (p,)
        Permutation vector.
    inv_perm : ndarray (p,)
        Inverse permutation vector.
    m : int
        Number of mandatory features.
    """
    X = np.asarray(X, float)
    n, p = X.shape
    M = np.array(sorted(set(M)), dtype=int)
    m = len(M)

    mask = np.ones(p, dtype=bool)
    mask[M] = False
    rest = np.where(mask)[0]

    perm = np.concatenate([M, rest]).astype(int)
    inv_perm = np.empty(p, dtype=int)
    inv_perm[perm] = np.arange(p, dtype=int)

    Xp = X[:, perm]
    return Xp, perm, inv_perm, m


# ================================================================
# Inner solver using sklearn multinomial ridge via feature scaling
# ================================================================

def solve_inner_g_sklearn(Xp, y, C, t, m, lam, delta, *,
                          sklearn_C=None, max_iter=5000, tol=1e-10, eps=1e-12):
    """
    Solve the inner minimisation of g(t, xi0, Xi) using sklearn's
    LogisticRegression with L-BFGS-B, by scaling features to convert
    the weighted ridge into a standard ridge problem.

    Parameters
    ----------
    Xp : ndarray (n, p)
        Design matrix with mandatory features first.
    y : ndarray (n,)
        Labels (1-indexed for multinomial, 0/1 for binary).
    C : int
        Number of classes.
    t : ndarray (p-m,)
        Weights for optional features.
    m : int
        Number of mandatory features.
    lam : float
        Ridge penalty.
    delta : float
        Penalty schedule value.
    sklearn_C : float or None
        sklearn regularisation parameter; auto-calibrated if None.
    max_iter : int
        Maximum iterations for L-BFGS-B.
    tol : float
        Solver convergence tolerance.
    eps : float
        Clamping floor.

    Returns
    -------
    xi0_hat : ndarray (C-1,)
        Fitted intercept.
    Xi_hat : ndarray (p, C-1)
        Fitted coefficient matrix (baseline representation).
    info : dict
        Solver diagnostics including Xi_hat_full (all C classes).
    """
    from sklearn.linear_model import LogisticRegression

    Xp = np.asarray(Xp, float)
    y = np.asarray(y, int).ravel()
    n, p = Xp.shape
    K = C - 1
    assert t.shape == (p - m,)

    # sklearn expects labels 0..C-1
    if y.min() == 1 and y.max() == C:
        y_skl = y - 1
    else:
        y_skl = y
    assert y_skl.min() >= 0 and y_skl.max() <= C - 1

    a = alpha_nm(t, lam, delta, eps=eps)
    if np.any(a <= 0):
        raise ValueError("alpha_nm(t) must be positive. Check lam, delta, t.")

    w_row = np.empty(p, dtype=float)
    w_row[:m] = lam
    w_row[m:] = a
    d = 1.0 / np.sqrt(np.maximum(w_row, eps))

    Xs = Xp * d[None, :]

    if sklearn_C is None:
        sklearn_C = 1.0 / (2.0 * n)

    clf = LogisticRegression(
        penalty="l2",
        C=sklearn_C,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=max_iter,
        tol=tol,
    )
    clf.fit(Xs, y_skl)

    coef_full = clf.coef_
    intercept_full = clf.intercept_

    if coef_full.shape[0] == 1 and C == 2:
        coef_full = np.vstack([np.zeros_like(coef_full), coef_full])
        intercept_full = np.array([0.0, float(intercept_full[0])])

    base = C - 1
    coef_diff = coef_full[:base, :] - coef_full[base, :][None, :]
    int_diff = intercept_full[:base] - intercept_full[base]

    Xi_hat = (d[None, :] * coef_diff).T
    xi0_hat = int_diff.copy()

    Xi_hat_full = (d[None, :] * coef_full).T

    info = {
        "sklearn_C": float(sklearn_C),
        "sklearn_n_iter": getattr(clf, "n_iter_", None),
        "sklearn_converged": bool(np.max(getattr(clf, "n_iter_", [max_iter])) < max_iter),
        "Xi_hat_full": Xi_hat_full,
    }
    return xi0_hat, Xi_hat, info


# ================================================================
# Combined gradient + inner solve for the Frank-Wolfe loop
# ================================================================

def grad_f_t_multinomial_fast(X, y, C, t, m, delta, lam,
                              xi0_ws=None, Xi_ws=None,
                              maxiter=5000, gtol=1e-12, ftol=1e-12, eps_t=1e-12):
    """
    Compute the Danskin envelope gradient for the multinomial model
    by solving the inner problem and evaluating the analytic gradient.

    Uses all C class coefficients (not just C-1 baseline) to match
    the symmetric objective minimised by sklearn.

    Parameters
    ----------
    X : ndarray (n, p)
        Design matrix (mandatory features first).
    y : ndarray (n,)
        Labels.
    C : int
        Number of classes.
    t : ndarray (p-m,)
        Weights for optional features.
    m : int
        Number of mandatory features.
    delta : float
        Penalty schedule value.
    lam : float
        Ridge penalty.
    xi0_ws, Xi_ws : ndarray or None
        Warm-start values (currently unused, kept for API compatibility).
    maxiter : int
        Maximum solver iterations.
    gtol, ftol : float
        Solver tolerances.
    eps_t : float
        Clamping floor.

    Returns
    -------
    grad_t : ndarray (p-m,)
        Gradient with respect to optional feature weights.
    xi0_hat : ndarray (C-1,)
        Fitted intercept.
    Xi_hat : ndarray (p, C-1)
        Fitted coefficients (baseline representation).
    info : dict
        Solver diagnostics.
    """
    xi0_hat, Xi_hat, info_sklearn = solve_inner_g_sklearn(
        X, y, C, t, m, lam, delta,
        sklearn_C=None, max_iter=maxiter, tol=ftol, eps=eps_t
    )

    Xi_full = info_sklearn["Xi_hat_full"]
    tj = np.maximum(np.asarray(t, float), eps_t)
    rownorm2 = np.sum(Xi_full[m:, :] ** 2, axis=1)
    grad_t = -2.0 * (lam + delta) * rownorm2 / (tj ** 3)

    info = {
        "success": bool(info_sklearn["sklearn_converged"]),
        "nit": info_sklearn["sklearn_n_iter"],
        "sklearn_C": info_sklearn["sklearn_C"],
    }

    return grad_t, xi0_hat, Xi_hat, info
