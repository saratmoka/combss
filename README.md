<img src="combss_logo.png" alt="COMBSS Logo" width="200" style="
    border-radius: 6px;
    box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    filter: drop-shadow(0 2px 2px rgba(0,0,0,0.05));
">


# Continuous Optimization Method for Best Subset Selection

[![PyPI version](https://img.shields.io/pypi/v/combss)](https://pypi.org/project/combss/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/saratmoka/combss/blob/main/LICENSE)

Python implementation of a novel continuous optimization method for best subset selection in linear regression.

📄 **Reference**:  
Moka, Liquet, Zhu & Muller (2024)  
*[COMBSS: best subset selection via continuous optimization](https://link.springer.com/article/10.1007/s11222-024-10387-8)*  
*Statistics and Computing*  

🔗 **GitHub Repository**: [saratmoka/combss](https://github.com/saratmoka/combss)

## Key Features
- 🎯 Continuous relaxation of discrete subset selection
- ⚡  Scalable optimization for high-dimensional data

## Intercept Handling

The intercept term (if included) is subject to the same selection process as other features.

## Installation

```bash
pip install combss
```

## Quick Start

A simple example:

```python
import combss
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=50, noise=0.1, random_state=42)

# Split into training and validation sets (60-40 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

# Initialize and fit model with validation data
model = combss.linear.model()
model.fit(
    X_train=X_train, 
    y_train=y_train,
    X_val=X_val,      # Validation features
    y_val=y_val,      # Validation targets
    q=10,             # Maximum subset size
    nlam=50           # Number of λ values
)

# Results
print("Best subset indices:", model.subset)
print("Best coefficients:", model.coef_)
print("Validation MSE:", model.mse)
print("Optimal lambda:", model.lambda_)
print("Computation time (s):", model.run_time)
```

An example with known true coefficients:

```python

import combss
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Configuration
n_samples = 5000
n_features = 50
n_informative = 5  # the number of non-zero coefficients
noise_level = 0.1

# Generate data with exactly 5 informative features
X, y, true_coef = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative, 
    noise=noise_level,
    coef=True,  # Return the actual coefficients used
    random_state=42
)

# The true coefficients will be non-zero for first 5 features
print("Number of truly informative features:", sum(true_coef != 0))  

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

# Initialize and fit model
model = combss.linear.model()
model.fit(
    X_train=X_train, 
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    q=10,
    nlam=50
)

# Results analysis
print("\nTrue non-zero coefficients:", np.where(true_coef != 0)[0])
print("Estimated subset:", model.subset)
print("\nValidation MSE:", model.mse)
```

## Documentation

### Core Parameters

| Parameter   | Description                          | Default |
|-------------|--------------------------------------|---------|
| `q`         | Maximum subset size                  | min(n,p)|
| `nlam`      | Number of λ values                   | 50      |
| `scaling`   | Enable feature scaling               | True    |
| `tau`       | Threshold parameter                  | 0.5     |
| `delta_frac`| δ/n in objective function            | 1       |

### Other Parameters

```python
model.fit(
    ...,
    t_init=t_init,     # Initial point for vector t
    eta=0.001,         # Truncation parameter
    patience=10,       # Early stopping rounds
    gd_maxiter=1000,   # Maximum number of iterations for the gradient based optimization
    gd_tol=1e-5,       # Tolerance for the gradient based optimization
    cg_maxiter=1000,   # Maximum number of iterations allowed in the conjugate gradient method
    cg_tol=1e-6        # Conjugate gradient tolerance
)
```

### Output Attributes

| Attribute    | Description                          |
|--------------|--------------------------------------|
| `subset`     | Selected feature indices (0-based)   |
| `coef_`      | Regression coefficients              |
| `mse`        | Mean squared error                   |
| `lambda_`    | Optimal λ value                      |
| `run_time`   | Execution time (seconds)             |
| `subset_list`| The list of subsets over the grid    |
| `lambda_list`| The grid of λ values.                |

## Dependencies

- Python 3.7+
- NumPy (≥1.21.0)
- SciPy (≥1.7.0)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Developers

- Sarat Moka ([@saratmoka](https://github.com/saratmoka))
- Hua Yang Hu



