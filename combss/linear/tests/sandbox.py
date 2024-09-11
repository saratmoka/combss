#%%

import numpy as np
import random

random.seed(1)

# Define dimensions
n = 10
p = 10

# Generate X, B, and Y as before
X = np.random.randn(n, p)
B = np.zeros((p, 1))

for i in range(5):
    B[i] = 0.5**i

# noise = np.random.randn(n, 1)
# Y = X @ B + noise
Y = X @ B

# Combine X and Y
data = np.hstack((X, Y))

# Save to CSV
np.savetxt('sandbox.csv', data, delimiter=',', header=','.join([f'X{i+1}' for i in range(p)] + ['Y']), comments='', fmt='%.6f')
# %%
