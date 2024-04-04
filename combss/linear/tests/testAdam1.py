#%%
import sys
import os
 
import numpy as np
from numpy.linalg import inv, pinv, norm
from scipy.sparse.linalg import cg
from tqdm import tqdm
from sklearn import metrics
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import pandas as pd
import time

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import optimize
import data_gen


#%%
np.random.seed(0)
p = 20
k0 = 10
case = 2

corr = 0.8

n = 100
mean = 10*np.ones(20)
noise_var = 0.2

beta0, true_model = data_gen.gen_beta0(p, k0, case)
cov = data_gen.cov_X(p, corr)
data = data_gen.gen_data(n, p, mean, cov, noise_var, beta0, centralize=False)

X = data[0]
y = data[1]


#%%
lam = 0
adam_res = optimize.adam_v1(X, y, lam, gam1 = 0.9, gam2 = 0.999, alpha = 0.001, epsilon = 10e-8, maxiter = 1e5, tol = 1e-5, tau = 0.5)

#%%
beta_pred = adam_res[0]
t_pred = adam_res[1]
model_pred = adam_res[2]
converge = adam_res[3]
i = adam_res[4]
betas = adam_res[5]
ts = adam_res[6]

print(f"Final Result: Adam Result Indicies: {model_pred}\n")


# %%
for i in range(12):
    beta_array = [beta[i] for beta in betas]
    indices = range(len(beta_array))

    plt.plot(indices, beta_array, marker='o', linestyle='-')

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel(f'Beta_{i+1}')
    plt.title(f'Plot of Beta_{i+1} Value by Iteration')

    # Show the plot
    plt.show()

# %%
for i in range(20):
    t_array = [t[i] for t in ts]
    iterations = range(len(t_array))

    plt.plot(iterations, t_array, marker='o', linestyle='-')

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel(f't_{i+1}')
    plt.title(f'Plot of t_{i+1} Value by Iteration')

    # Show the plot
    plt.show()

# %%
