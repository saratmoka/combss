# COMBSS
This is the package for COMBSS, a novel continuous optimisation method toward best subset selection, developed from the paper Moka et al. (2024).

## Dependencies

This package relies on the following libraries:

- `numpy` (version 1.21.0 or later): Numerical computing.
- `scipy` (version 1.7.0 or later): Sparse matrix operations and linear algebra.
- `scikit-learn` (version 1.0.0 or later): Machine learning and evaluation metrics.

These will be installed automatically if you install the package via `pip`. Alternatively, they can also be installed manually.

# COMBSS Installation and Usage Guide

## Installation

Users can install **COMBSS** using the `pip` command-line tool:

```bash
pip install combss
```

## Usage Guide
For demonstrative purposes, we apply COMBSS on a dataset created beforehand, with X_train, y_train, X_test, y_test generated from a 80-20 train-test split prior to this example.

### Importing COMBSS

To import **COMBSS** after installation, use the following command:

```python
import combss
```

COMBSS is implemented as a class named `model` within the `linear` module. Users can instantiate an instance of the `model` class to utilize its methods:

```python
# Instantiating an instance of the combss class
optimiser = combss.linear.model()
```

### Fitting the Model

To use COMBSS for best subset selection, call the `fit` method within the `model` class. Here are some commonly used arguments:

- **q**: Maximum subset size. Defaults to min(number of observations, number of predictors).
- **nlam**: Number of \(λ\) values in the dynamic grid. Default is 50.
- **scaling**: Boolean to enable feature scaling. Default is `False`.

Example usage:

```python
# A sample usage of the commonly used arguments
optimiser.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, q=8, nlam=20, scaling=True)
```

### Additional Fitting Arguments

Other arguments include:

- **t_init**: Initial point for the vector \(t\).
- **tau**: Threshold parameter for subset mapping.
- **delta_frac**: Value of \(n/δ\) in the objective function.
- **eta**: Truncation parameter during gradient descent.
- **patience**: Number of iterations before termination.
- **gd_maxiter**: Maximum iterations for gradient descent.
- **gd_tol**: Tolerance for gradient descent.
- **cg_maxiter**: Maximum iterations for the conjugate gradient algorithm.
- **cg_tol**: Tolerance for the conjugate gradient algorithm.

Modified usage example:

```python
# A modified usage of the fit method
optimiser.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, q=10, nlam=50, scaling=True, tau=0.9, delta_frac=20)
```

### Model Attributes

After fitting, the following attributes can be accessed:

- **subset**: Indices of the optimal subset.
- **mse**: Mean squared error on test data.
- **coef_**: Coefficients of the linear model.
- **lambda_**: Optimal \(λ\) value.
- **run_time**: Time taken for fitting.
- **lambda_list**: List of \(λ\) values explored.
- **subset_list**: Subsets obtained for each \(λ\).

Example:

```python
optimiser.subset
# Output: array([0, 1, 2, 3, 4, 6, 7, 8])

optimiser.mse
# Output: 19.94
```

## Illustrative Examples

### Example 1

```python
optimiser.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, q=8, nlam=20, scaling=True)

print(optimiser.subset)
# Output: [0, 1, 2, 3, 4, 6, 7, 8]
```

### Example 2

```python
optimiser.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, q=10, nlam=50, scaling=True, tau=0.9, delta_frac=20)

print(optimiser.subset)
# Output: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

