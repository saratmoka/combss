Quick Start
===========

Linear regression
-----------------

With validation data (selects best subset automatically):

.. code-block:: python

   import combss
   from sklearn.datasets import make_regression
   from sklearn.model_selection import train_test_split

   X, y = make_regression(n_samples=1000, n_features=50, n_informative=5,
                          noise=0.1, random_state=42)
   X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4)

   model = combss.linear.model()
   model.fit(X_train, y_train, X_val=X_val, y_val=y_val, q=10)

   print("Best subset:", model.subset)
   print("Best MSE:", model.mse)

   for k, feat in zip(model.k_list, model.subset_list):
       print(f"k={k:2d}  features={feat.tolist()}")

Without validation data (returns subset path only):

.. code-block:: python

   model = combss.linear.model()
   model.fit(X_train, y_train, q=10)

   for k, feat in zip(model.k_list, model.subset_list):
       print(f"k={k:2d}  features={feat.tolist()}")

To use the original Adam + dynamic-lambda method from v1.x:

.. code-block:: python

   model = combss.linear.model()
   model.fit(X_train, y_train, X_val=X_val, y_val=y_val,
             q=10, method='original')

   print("Best subset:", model.subset)
   print("Validation MSE:", model.mse)
   print("Optimal lambda:", model.lambda_)

Binary logistic regression
--------------------------

.. code-block:: python

   import combss

   # X_train, X_val : (n, p) feature matrices
   # y_train, y_val : binary labels {0, 1}

   model = combss.logistic.model()
   model.fit(X_train, y_train, X_val=X_val, y_val=y_val, q=15)

   print("Best subset:", model.subset)
   print("Best accuracy:", model.accuracy)

   for k, feat in zip(model.k_list, model.subset_list):
       print(f"k={k:2d}  features={feat.tolist()}")

Multinomial logistic regression
-------------------------------

.. code-block:: python

   import combss
   import numpy as np
   import pandas as pd

   # Example with the Khan SRBCT dataset (4 tumour classes, 2308 genes)
   train = pd.read_csv('data/Khan_train.csv')
   test  = pd.read_csv('data/Khan_test.csv')

   y_train = train.iloc[:, 0].values.astype(int)
   X_train = train.iloc[:, 1:].values.astype(float)
   y_test  = test.iloc[:, 0].values.astype(int)
   X_test  = test.iloc[:, 1:].values.astype(float)

   C = len(np.unique(y_train))

   model = combss.multinomial.model()
   model.fit(X_train, y_train, X_val=X_test, y_val=y_test, q=20, C=C)

   print("Best subset:", model.subset)
   print("Best accuracy:", model.accuracy)

   for k, feat in zip(model.k_list, model.subset_list):
       print(f"k={k:2d}  features={feat.tolist()}")

Lambda selection via LOOCV
--------------------------

Select the ridge penalty using leave-one-out cross-validation:

.. code-block:: python

   import combss

   # Classification
   best_lam, best_lam_per_k, cv_df = combss.cv.select_lambda(
       X_train, y_train,
       q=15, C=4,
       model_type='multinomial',
   )

   # Linear regression
   best_lam, best_lam_per_k, cv_df = combss.cv.select_lambda(
       X_train, y_train,
       q=15,
       model_type='linear',
   )
