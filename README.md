<img src="combss_logo.png" alt="COMBSS Logo" width="200" style="
    border-radius: 6px;
    box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    filter: drop-shadow(0 2px 2px rgba(0,0,0,0.05));
">

# COMBSS: Best Subset Selection for Generalised Linear Models

[![PyPI version](https://img.shields.io/pypi/v/combss)](https://pypi.org/project/combss/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/saratmoka/combss/blob/main/LICENSE)

Python implementation of **COMBSS** (Continuous Optimisation for Best Subset Selection) for generalised linear models. COMBSS reformulates the NP-hard discrete subset selection problem as a continuous optimisation over the hypercube [0,1]^p, making it scalable to high-dimensional settings with p >> n.

**Supported model types:**
- **Linear regression** (continuous response)
- **Binary logistic regression** (two-class classification)
- **Multinomial logistic regression** (multi-class, C > 2)

## References

> Moka, Liquet, Zhu & Muller (2024).
> [COMBSS: best subset selection via continuous optimization](https://link.springer.com/article/10.1007/s11222-024-10387-8).
> *Statistics and Computing*.

> Mathur, Liquet, Muller & Moka (2026).
> [Parsimonious Subset Selection for Generalized Linear Models with Biomedical Applications](https://arxiv.org/abs/2603.21952v1).
> *arXiv preprint*.

## Installation

```bash
pip install combss
```

## Quick Start

### Linear regression

```python
import combss
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=50, n_informative=5,
                       noise=0.1, random_state=42)

model = combss.linear.model()
model.fit(X, y, q=10)

# Selected features for each subset size k = 1, ..., q
for k, feat in enumerate(model.models, 1):
    print(f"k={k:2d}  features={feat.tolist()}")
```

To use the original Adam + dynamic-lambda method from v1.x:

```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4)

model = combss.linear.model()
model.fit(X_train, y_train, X_val=X_val, y_val=y_val, q=10, method='original')

print("Best subset:", model.subset)
print("Validation MSE:", model.mse)
```

### Binary logistic regression

```python
import combss
import numpy as np

# X : (n, p) feature matrix
# y : (n,) binary labels {0, 1}

model = combss.logistic.model()
model.fit(X, y, q=15)

for k, feat in enumerate(model.models, 1):
    print(f"k={k:2d}  features={feat.tolist()}")
```

### Multinomial logistic regression

```python
import combss
import numpy as np
import pandas as pd

# Example with the Khan SRBCT dataset (4 tumour classes, 2308 genes)
train = pd.read_csv('data/Khan_train.csv')
y_train = train.iloc[:, 0].values.astype(int)
X_train = train.iloc[:, 1:].values.astype(float)

C = len(np.unique(y_train))

model = combss.multinomial.model()
model.fit(X_train, y_train, q=20, C=C)

for k, feat in enumerate(model.models, 1):
    print(f"k={k:2d}  features={feat.tolist()}")
```

### Lambda selection via LOOCV

Select the ridge penalty using leave-one-out cross-validation:

```python
import combss

# Classification
best_lam, best_lam_per_k, cv_df = combss.cv.cv_select_lambda(
    X_train, y_train,
    q=15, C=4,
    model_type='multinomial',
)

# Linear regression
best_lam, best_lam_per_k, cv_df = combss.cv.cv_select_lambda(
    X_train, y_train,
    q=15,
    model_type='linear',
)
```

## API Reference

### `combss.linear.model`

Best subset selection for linear regression.

```python
model = combss.linear.model()
model.fit(X_train, y_train, q=10)              # Frank-Wolfe method (default)
model.fit(X_train, y_train, X_val, y_val,      # Original method
          q=10, method='original')
```

#### Frank-Wolfe method parameters (method='fw', default)

Sparsity is controlled by `k` (model size): COMBSS returns selected features for
each k = 1, ..., q. The `lam_ridge` parameter is an optional ridge regularisation
on the coefficients in the inner solver, unrelated to the sparsity penalty `lambda`
used in the original method.

| Parameter | Default | Description |
|---|---|---|
| `q` | min(n,p) | Maximum subset size |
| `Niter` | 25 | Number of homotopy iterations |
| `lam_ridge` | 0 | Ridge penalty on coefficients in the inner solver |
| `alpha` | 0.01 | Frank-Wolfe step size |
| `scale` | True | Column-normalise X |
| `mandatory_features` | None | 1-indexed features always included |
| `inner_tol` | 1e-4 | Inner solver tolerance |
| `verbose` | True | Print progress |

#### Original method parameters

Sparsity is controlled by `lambda`: a grid of lambda values is searched, and each
lambda yields a different subset. The best subset is selected by validation MSE.

| Parameter | Default | Description |
|---|---|---|
| `q` | min(n,p) | Maximum subset size |
| `nlam` | 50 | Number of lambda values in dynamic grid |
| `scaling` | True | Enable feature scaling |
| `tau` | 0.5 | Threshold parameter |
| `delta_frac` | 1 | n/delta in objective function |
| `eta` | 0.001 | Truncation parameter |
| `patience` | 10 | Early stopping rounds |
| `gd_maxiter` | 1000 | Maximum Adam iterations |
| `gd_tol` | 1e-5 | Adam convergence tolerance |
| `cg_maxiter` | n | Conjugate gradient max iterations |
| `cg_tol` | 1e-5 | Conjugate gradient tolerance |

#### Output attributes

| Attribute | FW | Original | Description |
|---|:---:|:---:|---|
| `models` | x | | List of selected subsets for k = 1..q (1-indexed) |
| `subset` | x | x | Selected feature indices (0-indexed) |
| `coef_` | | x | Regression coefficients |
| `mse` | | x | Validation MSE |
| `lambda_` | x | x | Lambda value used / optimal lambda |
| `run_time` | x | x | Execution time (seconds) |
| `subset_list` | | x | Subsets over the lambda grid |
| `lambda_list` | | x | Lambda grid values |

### `combss.logistic.model`

Best subset selection for binary logistic regression.

```python
model = combss.logistic.model()
model.fit(X_train, y_train, q=15)
```

Parameters and attributes are the same as the Frank-Wolfe method in `combss.linear.model`, except labels `y` must be binary {0, 1}.

### `combss.multinomial.model`

Best subset selection for multinomial logistic regression.

```python
model = combss.multinomial.model()
model.fit(X_train, y_train, q=20, C=4)
```

| Additional Parameter | Default | Description |
|---|---|---|
| `C` | inferred | Number of classes |

Labels `y` must be in {1, ..., C}. All other parameters and attributes are the same as the Frank-Wolfe method in `combss.linear.model`.

### `combss.cv.cv_select_lambda`

LOOCV-based ridge penalty selection.

```python
best_lam, best_lam_per_k, results_df = combss.cv.cv_select_lambda(
    X, y, q, C=None, lambda_grid=None, model_type='multinomial', ...
)
```

| Parameter | Default | Description |
|---|---|---|
| `X` | -- | Feature matrix (n, p), no intercept |
| `y` | -- | Response / labels |
| `q` | -- | Maximum subset size |
| `C` | None | Number of classes (required for classification) |
| `lambda_grid` | [0] + logspace(-3,1,10) | Candidate ridge penalty values |
| `Niter` | 50 | Homotopy iterations |
| `model_type` | 'multinomial' | 'logit', 'multinomial', or 'linear' |
| `lambda_refit` | 0 | Ridge penalty for LOOCV refit |

**Returns:** `(best_lambda, best_lambda_per_k, results_df)`
- Classification: `results_df` has columns `lambda, k, loocv_acc, selected`
- Linear: `results_df` has columns `lambda, k, loocv_mse, selected`

### `combss.metrics`

Performance metrics for variable selection evaluation.

```python
result = combss.metrics.performance_metrics(X, beta_true, beta_pred)
# result keys: pe, mcc, acc, sens, spec, f1, prec
```

## Algorithm Overview

The Frank-Wolfe homotopy algorithm operates as follows:

1. **Initialise** t = (k/p, ..., k/p) -- centroid of the k-sparse simplex T_k
2. **For i = 1, ..., N** (homotopy loop):
   - Set delta_i on a geometric schedule from delta_min to delta_max
   - Solve the ridge-penalised GLM inner problem at current t
   - Compute the Danskin gradient (no Hessian required)
   - Find the Frank-Wolfe vertex (k smallest gradient components)
   - Update: t <- (1 - alpha) t + alpha s
3. **Select** the k features with the largest final t values

The penalty schedule is auto-calibrated from the spectral norm of X, so no tuning is needed in practice.

## Intercept Handling

The intercept is handled internally and is not subject to selection. For the Frank-Wolfe method, an intercept column is prepended automatically. For the original method, the intercept is treated the same as other features.

## Dependencies

- Python 3.8+
- NumPy (>= 1.21.0)
- SciPy (>= 1.7.0)
- scikit-learn (>= 1.0.0)
- pandas (>= 1.3.0)

## Developers

- [Sarat Moka](https://github.com/saratmoka)
- [Anant Mathur](https://anantmathur44.github.io/)
- Hua Yang Hu

## Contributing

Contributions are welcome! Please open an issue or submit a pull request at [github.com/saratmoka/combss](https://github.com/saratmoka/combss).

## License

Apache 2.0
