Quick Start
===========

Linear regression
-----------------

.. code-block:: python

   import combss
   import numpy as np
   from sklearn.datasets import make_regression

   X, y = make_regression(n_samples=1000, n_features=50, n_informative=5,
                          noise=0.1, random_state=42)

   model = combss.linear.model()
   model.fit(X, y, q=10)

   # Selected features for each subset size k = 1, ..., q
   for k, feat in enumerate(model.models, 1):
       print(f"k={k:2d}  features={feat.tolist()}")

To use the original Adam + dynamic-lambda method from v1.x:

.. code-block:: python

   from sklearn.model_selection import train_test_split

   X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4)

   model = combss.linear.model()
   model.fit(X_train, y_train, X_val=X_val, y_val=y_val,
             q=10, method='original')

   print("Best subset:", model.subset)
   print("Validation MSE:", model.mse)

Binary logistic regression
--------------------------

.. code-block:: python

   import combss
   import numpy as np

   # X : (n, p) feature matrix
   # y : (n,) binary labels {0, 1}

   model = combss.logistic.model()
   model.fit(X, y, q=15)

   for k, feat in enumerate(model.models, 1):
       print(f"k={k:2d}  features={feat.tolist()}")

Multinomial logistic regression
-------------------------------

.. code-block:: python

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

Lambda selection via LOOCV
--------------------------

Select the ridge penalty using leave-one-out cross-validation:

.. code-block:: python

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
