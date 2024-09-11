#%%
import pandas as pd
import itertools
import numpy as np


# Load the CSV file into a DataFrame
df = pd.read_csv('sandbox.csv')

# Extract X and Y
X = df.iloc[:, :-1].values  # All columns except the last one
Y = df.iloc[:, -1].values   # The last column

#%%
def compute_mse(combo):
    # Select columns from X based on the current combination
    X_subset = X[:, combo]
    
    # Compute Y_pred using the subset of X
    B_subset = np.linalg.lstsq(X_subset, Y, rcond=None)[0]
    Y_pred = X_subset @ B_subset
    
    # Compute the Mean Squared Error
    mse = np.mean((Y - Y_pred) ** 2)
    return mse

#%%
min_mse = np.full(11, np.inf)
best_combination = np.full(11, None)
indices = range(10)

for r in [0,1,2,3,4,5,6,7,8,9,10]:
    for combo in itertools.combinations(indices, r):
        try:
            mse = compute_mse(combo)
            if mse < min_mse[r]:
                min_mse[r] = mse
                best_combination[r] = combo
        except np.linalg.LinAlgError:
            # Handle cases where X_subset is singular or not invertible
            continue

print("Minimal MSE:", min_mse)
print("Best combination of indices:", best_combination)

# %%
betas = [0,1,2,3,4,5,6,7,8,9,10]

for r in [0,1,2,3,4,5,6,7,8,9,10]:
    betas[r] = np.linalg.lstsq(X[:, best_combination[r]], Y, rcond=None)[0]

B_sol = np.linalg.lstsq(X, Y, rcond=None)[0]

print("Best coefficients for each size:", betas)

print(B_sol)


# %%
