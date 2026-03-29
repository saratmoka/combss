COMBSS: Best Subset Selection for Generalised Linear Models
============================================================

**combss** is a Python package implementing COMBSS (Continuous Optimisation for
Best Subset Selection) for generalised linear models.  COMBSS reformulates the
NP-hard discrete subset selection problem as a continuous optimisation over the
hypercube :math:`[0,1]^p`, making it scalable to high-dimensional settings with
:math:`p \gg n`.

Supported model types
---------------------

- **Linear regression** (continuous response)
- **Binary logistic regression** (two-class classification)
- **Multinomial logistic regression** (multi-class, :math:`C > 2`)

Installation
------------

.. code-block:: bash

   pip install combss

References
----------

- Moka, Liquet, Zhu & Muller (2024).
  `COMBSS: best subset selection via continuous optimization <https://link.springer.com/article/10.1007/s11222-024-10387-8>`_.
  *Statistics and Computing*.

- Mathur, Liquet, Muller & Moka (2026).
  `Parsimonious Subset Selection for Generalized Linear Models with Biomedical Applications <https://arxiv.org/abs/2603.21952v1>`_.
  *arXiv preprint*.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   api
   algorithm
   changelog
