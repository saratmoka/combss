# COMBSS
This is the package for COMBSS, a novel continuous optimisation method toward best subset selection, developed from the paper Moka et al. (2024).

For a more detaled overview of COMBSS, refer to https://link.springer.com/article/10.1007/s11222-024-10387-8.

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
- **nlam**: Number of λ values in the dynamic grid. Default is 50.
- **scaling**: Boolean to enable feature scaling. Default is `False`.

Example usage 1:

```python
# A sample usage of the commonly used arguments
optimiser.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, q=8, nlam=20, scaling=True)
```

### Additional Fitting Arguments

Other arguments include:

- **t_init**: Initial point for the vector t.
- **tau**: Threshold parameter for subset mapping.
- **delta_frac**: Value of δ/n in the objective function.
- **eta**: Truncation parameter during gradient descent.
- **patience**: Number of iterations before termination.
- **gd_maxiter**: Maximum iterations for gradient descent.
- **gd_tol**: Tolerance for gradient descent.
- **cg_maxiter**: Maximum iterations for the conjugate gradient algorithm.
- **cg_tol**: Tolerance for the conjugate gradient algorithm.

Modified usage example 2:

```python
# A modified usage of the fit method
optimiser.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, q=10, nlam=50, scaling=True, tau=0.9, delta_frac=20)
```

### Model Attributes

After fitting, the following attributes can be accessed:

- **subset**: Indices of the optimal subset.
- **mse**: Mean squared error on test data.
- **coef_**: Coefficients of the linear model.
- **lambda_**: Optimal λ value.
- **run_time**: Time taken for fitting.
- **lambda_list**: List of λ values explored.
- **subset_list**: Subsets obtained for each λ.

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
# A sample usage of the commonly used arguments
optimiser.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, q = 8, nlam = 20, scaling=True)

optimiser.subset
# array([0, 1, 2, 3, 4, 6, 7, 8])

optimiser.mse
# 19.940929997277212

optimiser.coef_
# array([ 0.85215,  1.50009,  0.39557,  2.3919,  -0.56994,
#         0.     ,  2.6758 ,  0.72726,  1.70696,  0.        ,
#         0.     ,  0.     ,  0.     ,  0.     ,  0.        ,
#         0.     ,  0.     ,  0.     ,  0.     ,  0.        ])

optimiser.lambda_
# 0.6401161339265333

optimiser.run_time
# 2.591932

optimiser.lambda_list
# [65.54789211407702,
# 32.77394605703851,
# 16.386973028519254,
# .
# .
# 0.5120929071412267,
# 0.6401161339265333]

optimiser.subset_list
# [array([], dtype=int64),
# array([], dtype=int64),
# array([], dtype=int64),
# .
# .
# array([0, 1, 2, 3, 4, 6, 7, 8]),
# array([0, 1, 2, 3, 4, 6, 7, 8])]
```
One can observe that a model of size q = 8 was recovered from the training data after approximately 2.59 seconds. The recovered model with elements of indices in the optimiser.subset array achieved a mean squared error of approximately 19.94 on the test data, after a series of up to nlam = 50 values of λ were explored in the dynamic grid search, starting with an null model explored when COMBSS was initialised with λ approximately equal to 65.548. 

One can additionally observe the following output after performing the fitting in the modified code example 2. In this setting, q is instead taken to equal 10, exploring 50 values of λ with feature scaling, a more stringent thresholding value of 𝜏 = 0.9, and taking the fraction delta/n for the objective function equal to 20. All other arguments take their default values.

### Example 2

```python
# A sample usage of additional arguments
combssOptimiser.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, q = 10, nlam = 50, scaling=True, tau = 0.9, delta_frac = 20)

optimiser.subset
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

optimiser.mse
# 19.50638319557191

optimiser.coef_
# array([ 0.76678,  1.51074,  0.49312,  2.45588,  -0.69150,
#         0.13782,  2.43072,  0.89641,  0.88130,  1.13421 ,  
#         0.     ,  0.     ,  0.     ,  0.     ,  0.        ,
#         0.     ,  0.     ,  0.     ,  0.     ,  0.        ])

optimiser.lambda_
# 0.022003992103724584

optimiser.run_time
# 5.400080000000001

optimiser.lambda_list
# [65.54789211407702,
# 32.77394605703851,
# 16.386973028519254,
# .
# .
# 0.020003629185204166,
# 0.016002903348163334]

optimiser.subset_list
# [array([], dtype=int64),
# array([], dtype=int64),
# array([], dtype=int64),
# .
# .
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])]
```

One can observe that the changes to tau, delta_frac and nlam result in different values of lambda being explored, with a different navigation of subsets as the threshold parameter 𝜏 is increased in the subset mapping process, and the landscape of the objective function is changed. Consequently, an additional predictor from the true model is recovered at the expense of a larger computational cost.

