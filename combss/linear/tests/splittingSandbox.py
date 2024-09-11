#%%

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import variant6

# Sample data
df = pd.read_csv('sandbox.csv')

# Extract X and Y
X = df.iloc[:, :-1].values  # All columns except the last one
y = df.iloc[:, -1].values   # The last column

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

model_opt, mse_opt, beta_opt, q_opt, timer = variant6.combssV6(X_train, y_train, X_test, y_test, q = 10)
# %%
