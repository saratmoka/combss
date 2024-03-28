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

p = 20
k0 = 10
case = 2

corr = 0.8

n = 100
mean = [5]*20
noise_var = 0

beta0, true_model = data_gen.gen_beta0(p, k0, case)
cov = data_gen.cov_X(p, corr)
print(cov.shape)
data = data_gen.gen_data(n, p, mean, cov, noise_var, beta0, centralize=False)

X = data[0]
y = data[1]

print(f"X: {X}")
print(f"y: {y}")


adam_res = optimize.adam_v1(X, y, gam1 = 0.9, gam2 = 0.999, alpha = 0.1, epsilon = 10e-8, maxiter = 1e8, tol = 1e-2, tau = 0.5)

beta_pred = adam_res[0]
t_pred = adam_res[1]
model_pred = adam_res[2]
converge = adam_res[3]
i = adam_res[4]

print(f"Final Result: Adam Result Indicies: {model_pred}\n")
