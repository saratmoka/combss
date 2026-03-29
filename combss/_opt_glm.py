"""
combss._opt_glm.py

Private module containing the Frank-Wolfe homotopy algorithm for COMBSS
applied to generalised linear models (linear, logistic, multinomial).

Adapted from the COMBSS-GLM project (Mathur, Liquet, Muller & Moka, 2026).

Functions:
- _beta_tilde_sklearn_logit(): Inner solver for binary logistic regression.
- _grad_logit(): Danskin gradient for binary logistic model.
- _solve_inner_linear(): Closed-form inner solver for linear regression.
- _grad_linear(): Danskin gradient for linear regression.
- _compute_nu_max(): Power iteration for lambda_max(X^T X).
- _calibrate(): Auto-calibrate penalty schedule (delta_min, delta_max).
- fw(): Frank-Wolfe homotopy algorithm for model sizes k = 1, ..., q.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import combss._multinomial_grad as mg


# ================================================================
# Result container
# ================================================================

class Result:
    """Container for COMBSS Frank-Wolfe results."""
    def __init__(self, models, niters):
        self.models = models
        self.niters = niters


# ================================================================
# Binary logistic inner solver (sklearn L-BFGS-B via feature scaling)
# ================================================================

def _sigmoid(w):
    """Sigmoid function applied elementwise."""
    return 1.0 / (1.0 + np.exp(-w))


def _solve_inner_logit(t, X, y, delta, lam, *, mandatory_features, optional_features,
                       clip_eps=1e-6, max_iter=1000, tol=1e-4):
    """
    Solve min_{beta} h(beta, t) for the binary logistic objective using
    sklearn LogisticRegression (L-BFGS-B) via feature scaling.

    Parameters
    ----------
    t : ndarray
        Weights for optional features.
    X : ndarray (n, p)
        Design matrix WITHOUT intercept.
    y : ndarray (n,)
        Binary labels {0, 1}.
    delta : float
        Penalty schedule value.
    lam : float
        Ridge penalty.
    mandatory_features : array-like
        1-indexed mandatory feature columns.
    optional_features : array-like
        1-indexed optional feature columns.
    clip_eps : float
        Clamping floor for t.
    max_iter : int
        Maximum solver iterations.
    tol : float
        Solver tolerance.

    Returns
    -------
    beta_tilde : ndarray (p+1,)
        Fitted coefficients including intercept at index 0.
    converged : bool
        Whether the solver converged.
    """
    n = y.shape[0]
    t = np.clip(np.asarray(t, float).ravel(), clip_eps, 1 - clip_eps)

    mandatory_features = np.asarray(mandatory_features, dtype=int)
    optional_features = np.asarray(optional_features, dtype=int)
    mandatory_idx = mandatory_features - 1
    optional_idx = optional_features - 1

    p = X.shape[1]
    t_all = np.ones(p)
    t_all[optional_idx] = t
    t_all[mandatory_idx] = 1.0

    K = lam + delta * (1.0 - t_all ** 2)
    scale_vec = t_all / np.sqrt(K)
    X_scaled = X.astype(float) * scale_vec

    clf = LogisticRegression(
        penalty="l2", solver="lbfgs",
        C=1.0 / (2.0 * n),
        fit_intercept=True,
        max_iter=max_iter, tol=tol,
    ).fit(X_scaled, y)

    beta_hat = clf.coef_.ravel() / np.sqrt(K)
    b_hat = float(clf.intercept_[0])
    return np.concatenate(([b_hat], beta_hat)), clf.n_iter_[0] < max_iter


def _grad_logit(t, X, y, delta, lam, mandatory_features, optional_features,
                tol=1e-4):
    """
    Danskin envelope gradient of f(t) for binary logistic regression.

    X has intercept column at index 0. The gradient is computed using
    the logistic score X^T(y - sigmoid(X @ theta)) and the penalty term.

    Parameters
    ----------
    t : ndarray
        Weights for optional features.
    X : ndarray (n, p+1)
        Design matrix with intercept at column 0.
    y : ndarray (n,)
        Binary labels {0, 1}.
    delta : float
        Penalty schedule value.
    lam : float
        Ridge penalty.
    mandatory_features : list
        1-indexed mandatory feature columns.
    optional_features : ndarray
        1-indexed optional feature columns.
    tol : float
        Inner solver tolerance.

    Returns
    -------
    ndarray : gradient with respect to optional feature weights.
    """
    n = y.shape[0]
    t = np.asarray(t)

    beta_tilde, converged = _solve_inner_logit(
        t, X[:, 1:], y, delta, lam,
        mandatory_features=mandatory_features,
        optional_features=optional_features,
        tol=tol,
    )
    if not converged:
        print("Warning: inner logit solver did not converge.")

    beta_sh = beta_tilde[1:]

    mandatory_features = np.asarray(mandatory_features, dtype=int)
    optional_features = np.asarray(optional_features, dtype=int)
    mandatory_idx = mandatory_features - 1
    optional_idx = optional_features - 1

    p = X.shape[1]
    t_all = np.ones(p - 1)
    t_all[optional_idx] = t
    t_all[mandatory_idx] = 1.0

    theta = np.concatenate(([1.0], t_all)) * beta_tilde
    probs = _sigmoid(X @ theta)
    score = X.T @ (y - probs)

    grad_loss = -(1.0 / n) * score[1:] * beta_sh
    grad_penalty = -2.0 * delta * t_all * (beta_sh ** 2)

    return (grad_loss + grad_penalty)[optional_idx]


# ================================================================
# Linear regression inner solver and gradient
# ================================================================

def _solve_inner_linear(t, X, y, delta, lam, *,
                        mandatory_features, optional_features, clip_eps=1e-6):
    """
    Closed-form inner solver for linear regression COMBSS.

    Solves: min_{beta0, beta} (1/2n)||y - beta0*1 - X_feat beta||^2
                              + sum_j alpha_j(t) beta_j^2
    where alpha_j = (lam + delta) / t_j^2 - delta.

    Uses the Woodbury identity when p > n for efficiency.

    Parameters
    ----------
    t : ndarray
        Weights for optional features.
    X : ndarray (n, p+1)
        Design matrix with intercept at column 0.
    y : ndarray (n,)
        Continuous response.
    delta : float
        Penalty schedule value.
    lam : float
        Ridge penalty.
    mandatory_features : array-like
        1-indexed mandatory feature columns.
    optional_features : array-like
        1-indexed optional feature columns.
    clip_eps : float
        Clamping floor for t.

    Returns
    -------
    ndarray (p+1,) : fitted coefficients [beta0, beta_1, ..., beta_p].
    """
    n = X.shape[0]
    p_feat = X.shape[1] - 1
    t = np.clip(np.asarray(t, float).ravel(), clip_eps, 1.0 - clip_eps)

    mandatory_features = np.asarray(mandatory_features, dtype=int)
    optional_features = np.asarray(optional_features, dtype=int)
    mandatory_idx = mandatory_features - 1
    optional_idx = optional_features - 1

    t_all = np.ones(p_feat)
    t_all[optional_idx] = t
    t_all[mandatory_idx] = 1.0

    alpha = (lam + delta) / (t_all ** 2) - delta
    d = 2.0 * alpha

    # Centre X and y to handle the unpenalised intercept
    X_feat = X[:, 1:]
    y_bar = np.mean(y)
    X_bar = np.mean(X_feat, axis=0)
    X_c = X_feat - X_bar
    y_c = y - y_bar

    q = X_c.T @ y_c / n

    if p_feat <= n:
        A = X_c.T @ X_c / n + np.diag(d)
        beta_feat = np.linalg.solve(A, q)
    else:
        # Woodbury identity for p >> n
        d_inv = 1.0 / d
        Xp = X_c / np.sqrt(n)
        r = d_inv * q
        v = Xp @ r
        M = np.eye(n) + (Xp * d_inv) @ Xp.T
        w = np.linalg.solve(M, v)
        beta_feat = r - d_inv * (Xp.T @ w)

    beta0 = y_bar - X_bar @ beta_feat
    return np.concatenate(([beta0], beta_feat))


def _grad_linear(t, X, y, delta, lam, mandatory_features, optional_features,
                 clip_eps=1e-6):
    """
    Danskin envelope gradient of f(t) for linear regression.

    df/dt_j = -2(lam + delta) / t_j^3 * beta_tilde_j^2

    Parameters
    ----------
    t : ndarray
        Weights for optional features.
    X : ndarray (n, p+1)
        Design matrix with intercept at column 0.
    y : ndarray (n,)
        Continuous response.
    delta : float
        Penalty schedule value.
    lam : float
        Ridge penalty.
    mandatory_features : list
        1-indexed mandatory feature columns.
    optional_features : ndarray
        1-indexed optional feature columns.
    clip_eps : float
        Clamping floor for t.

    Returns
    -------
    ndarray : gradient with respect to optional feature weights.
    """
    optional_features = np.asarray(optional_features, dtype=int)
    optional_idx = optional_features - 1

    beta_tilde = _solve_inner_linear(
        t, X, y, delta, lam,
        mandatory_features=mandatory_features,
        optional_features=optional_features,
        clip_eps=clip_eps,
    )
    beta_feat = beta_tilde[1:]

    t_opt = np.clip(np.asarray(t, float), clip_eps, 1.0 - clip_eps)
    grad = -2.0 * (lam + delta) / (t_opt ** 3) * beta_feat[optional_idx] ** 2
    return grad


# ================================================================
# Penalty schedule calibration (shared by all model types)
# ================================================================

def _compute_nu_max(X, n_iter=200, tol=1e-8):
    """
    Power iteration estimate of lambda_max(X^T X).

    Parameters
    ----------
    X : ndarray (n, p)
        Matrix whose largest eigenvalue of X^T X is sought.
    n_iter : int
        Maximum power iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    float : estimated lambda_max(X^T X).
    """
    rng = np.random.default_rng(42)
    v = rng.standard_normal(X.shape[1])
    v /= np.linalg.norm(v)
    nu_old = 0.0
    for _ in range(n_iter):
        Xv = X @ v
        XtXv = X.T @ Xv
        nu = float(Xv @ Xv)
        v = XtXv / np.linalg.norm(XtXv)
        if abs(nu - nu_old) < tol * max(nu, 1.0):
            break
        nu_old = nu
    return nu


def _calibrate(X_u, n, family='multinomial'):
    """
    Calibrate delta_min and delta_max from the non-mandatory columns.

    The concentration parameter delta_conc is set as:
        multinomial: nu_max / (4n)
        logit:       nu_max / (8n)
        linear:      3 * nu_max / (8n)

    Parameters
    ----------
    X_u : ndarray
        Non-mandatory columns of the design matrix.
    n : int
        Number of observations.
    family : str
        One of 'multinomial', 'logit', 'linear'.

    Returns
    -------
    dict : with keys 'nu_max', 'delta_conc', 'delta_max', 'delta_min'.
    """
    nu_max = _compute_nu_max(X_u)
    if family == 'multinomial':
        delta_conc = nu_max / (4.0 * n)
    elif family == 'linear':
        delta_conc = 3.0 * nu_max / (8.0 * n)
    else:  # 'logit'
        delta_conc = nu_max / (8.0 * n)
    delta_max = delta_conc
    delta_min = 1e-3 * delta_conc
    return {'nu_max': nu_max, 'delta_conc': delta_conc,
            'delta_max': delta_max, 'delta_min': delta_min}


# ================================================================
# Main algorithm: Frank-Wolfe homotopy (Algorithm 1)
# ================================================================

def fw(X, y, q,
       Niter=25,
       delta_min=None,
       delta_max=None,
       scale=True,
       lam=0,
       alpha=0.01,
       verbose=True,
       mandatory_features=None,
       model_type='logit',
       C=2,
       inner_tol=1e-4):
    """
    Run COMBSS Frank-Wolfe homotopy for model sizes k = 1, ..., q.

    Parameters
    ----------
    X : ndarray (n, p+1)
        Design matrix with intercept column at index 0.
    y : ndarray (n,)
        Response vector:
        - {0, 1} for logit
        - {1, ..., C} for multinomial
        - real-valued for linear
    q : int
        Maximum subset size to search.
    Niter : int
        Number of homotopy iterations (default 25).
    delta_min, delta_max : float or None
        Override auto-calibrated penalty schedule endpoints.
    scale : bool
        Column-normalise X before running (default True).
    lam : float
        Ridge regularisation parameter (default 0).
    alpha : float
        Frank-Wolfe step size (default 0.01).
    verbose : bool
        Print calibration and per-k progress (default True).
    mandatory_features : list or None
        List of 1-indexed column numbers to force into every model.
    model_type : str
        One of 'logit', 'multinomial', or 'linear'.
    C : int
        Number of classes (required for multinomial/logit; ignored for linear).
    inner_tol : float
        Convergence tolerance for the inner solver (default 1e-4).

    Returns
    -------
    Result
        Object with .models (list of sorted 1-indexed feature arrays)
        and .niters (list of iteration counts).
    """
    n, p = X.shape

    if mandatory_features is None:
        mandatory_features = []
    mandatory_features = set(mandatory_features)

    optional_features = np.array([j for j in range(1, p) if j not in mandatory_features])
    t_len = len(optional_features)

    # Column normalisation
    if scale:
        col_norms = np.linalg.norm(X[:, 1:], axis=0)
        X_norm = X.copy()
        X_norm[:, 1:] /= col_norms
    else:
        X_norm = X

    if model_type == 'multinomial':
        X_norm = X_norm[:, 1:]  # drop intercept column; handled internally
        mand_0 = np.array([j - 1 for j in mandatory_features], dtype=int)
        X_norm, perm, inv_perm, m = mg.move_mandatory_to_front(X_norm, mand_0)

    # Penalty schedule calibration
    if model_type == 'multinomial':
        X_sel = X_norm[:, m:]
    else:
        X_sel = X_norm[:, optional_features]

    cal = _calibrate(X_sel, n, family=model_type)
    if delta_min is None:
        delta_min = cal['delta_min']
    if delta_max is None:
        delta_max = cal['delta_max']
    r = (delta_max / delta_min) ** (1.0 / Niter)

    if verbose:
        print(f"  Calibration: nu_max={cal['nu_max']:.4f} | "
              f"delta_conc={cal['delta_conc']:.6f} | "
              f"delta_min={delta_min:.6f} | delta_max={delta_max:.6f} | r={r:.6f}")

    model_list = []
    niter_list = []

    for k in range(1, q + 1):
        if verbose:
            print(f"  k = {k}")

        t = np.ones(t_len) * (k / t_len)  # centroid of T_k

        if model_type == 'multinomial':
            xi0_hat = None
            Xi_hat = None

        for i in range(1, 2 * Niter):
            delta = min(delta_min * r ** i, delta_max)
            if model_type == 'logit':
                grad = _grad_logit(t, X_norm, y, delta, lam,
                                   sorted(mandatory_features), optional_features,
                                   tol=inner_tol)
            elif model_type == 'multinomial':
                grad, xi0_hat, Xi_hat, _ = mg.grad_f_t_multinomial_fast(
                    X_norm, y, C, t, m, delta, lam,
                    xi0_ws=xi0_hat, Xi_ws=Xi_hat,
                    ftol=inner_tol)
            else:  # linear
                grad = _grad_linear(t, X_norm, y, delta, lam,
                                    sorted(mandatory_features), optional_features)

            model = np.argsort(grad)[:k]
            s = np.zeros(t_len, dtype=int)
            s[model] = 1
            t = (1 - alpha) * t + alpha * s
            t = np.clip(t, 0.0001, 0.9999)

        chosen_optional = optional_features[model].tolist()
        selected = np.sort(sorted(mandatory_features) + chosen_optional)

        if verbose:
            print(f"    selected: {selected.tolist()}")

        model_list.append(selected)
        niter_list.append(Niter)

    return Result(model_list, niter_list)
