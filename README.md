# COMBSS: Continuous Optimization for Best Subset Selection

[![PyPI version](https://img.shields.io/pypi/v/combss)](https://pypi.org/project/combss/)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/saratmoka/combss/blob/main/LICENSE)

Python implementation of a novel continuous optimization method for best subset selection in linear regression.

📄 **Reference**:  
Moka, Liquet, Zhu & Muller (2024)  
*[COMBSS: best subset selection via continuous optimization](https://link.springer.com/article/10.1007/s11222-024-10387-8)*  
*Statistics and Computing*  

🔗 **GitHub Repository**: [saratmoka/combss](https://github.com/saratmoka/combss)

## Key Features
- 🎯 Continuous relaxation of discrete subset selection
- ⚡ Scalable optimization for high-dimensional data
- 🔌 Seamless integration with NumPy and SciPy

## Installation

```bash
pip install combss
```

## Quick Start

```python
import combss
from sklearn.datasets import make_regression

# Generate sample data
X, y = make_regression(n_samples=100, n_features=50, noise=0.1)

# Initialize and fit model
model = combss.linear.model()
model.fit(
    X_train=X, 
    y_train=y,
    q=10,          # Maximum subset size
    nlam=50        # Number of λ values
)

# Results
print("Optimal subset:", model.subset)
print("Coefficients:", model.coef_)
```

## Documentation

### Core Parameters

| Parameter    | Description                          | Default |
|-------------|--------------------------------------|---------|
| `q`         | Maximum subset size                  | min(n,p) |
| `nlam`      | Number of λ values                   | 50      |
| `scaling`   | Enable feature scaling               | True    |
| `tau`       | Threshold parameter                  | 0.9     |
| `delta_frac`| δ/n in objective function           | 20      |

### Advanced Options

```python
model.fit(
    ...,
    t_init=None,       # Initial point for vector t
    eta=0.1,           # Truncation parameter
    patience=5,        # Early stopping rounds
    gd_maxiter=1000,   # Gradient descent iterations
    cg_tol=1e-6        # Conjugate gradient tolerance
)
```

### Output Attributes

| Attribute     | Description                          |
|--------------|--------------------------------------|
| `subset`     | Indices of selected features         |
| `coef_`      | Regression coefficients              |
| `mse`        | Mean squared error                   |
| `lambda_`    | Optimal λ value                      |
| `run_time`   | Execution time (seconds)             |

## Dependencies

- Python 3.7+
- NumPy (≥1.21.0)
- SciPy (≥1.7.0)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Developers

- Sarat Moka ([@saratmoka](https://github.com/saratmoka))
- Hua Yang Hu

## License

[Apache](https://github.com/saratmoka/combss/blob/main/LICENSE)


