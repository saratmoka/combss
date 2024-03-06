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
from itertools import combinations


current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import optimize

# Parameters copied from prediction_main.py
t_init= []
delta_frac = 1
n = 100
p = 20
snr = 1
nlam = 100
eta = 0.001
case = 1
q = 20
corr = 0.8
bulk_results = np.zeros((1, 100))
k = 5

def calculate_MSE(n, X, y, indicies):
	X_s = X[:, indicies]
	beta_s = np.linalg.pinv(X_s.T @ X_s) @ X_s.T @ (y)
	residual = y - X_s @ beta_s
	mse = np.linalg.norm(residual)**2 / n
	return mse

dir = os.getcwd()
# path_train = '../n-%s-p%sSNR-%s.csv' %(n,p,snr)
path_train = dir + '/n-100-p20SNR-1Train.csv'
df = pd.read_csv(path_train, sep='\t', header=None)
data = df.to_numpy()
y_train = data[:, 0]
X_train = data[:, 1:]

path_test = dir + '/n-100-p20SNR-1Test.csv'
df = pd.read_csv(path_test, sep='\t', header=None)
data = df.to_numpy()
y_test = data[:, 0]
X_test = data[:, 1:]

n_tinit = len(t_init)
running_time = 0

if n_tinit == 0:
	t_init = [np.ones(p)*0.5]
	n_tinit = 1

br_result = optimize.BR_combss(X_train, y_train, t_init=t_init[0], k = 5, delta_frac=delta_frac)

print(f"Raw BR Result Indicies: {br_result}\n")

result = np.where(br_result > 0.75, 1, 0)
result_indices = np.where(result == 1)[0]

result_MSE = calculate_MSE(n, X_train, y_train, result_indices)
print(f"Boolean Relaxation Result Indicies: {result_indices}, MSE: {result_MSE}\n")

combinations_list = list(combinations(range(0, p), k))
total_combinations = [np.array(combination) for combination in combinations_list]

# Convert the list of arrays into a NumPy array
np_combinations = np.array(total_combinations)

min_MSE = np.inf
min_indices = [0]*k
for index_comb in np_combinations:
	comb_MSE = calculate_MSE(n, X_train, y_train, index_comb)
	if comb_MSE < min_MSE:
		print(f"Exact Result: {index_comb}, exact MSE: {comb_MSE}")
		min_MSE = comb_MSE
		min_indicies = index_comb
	
print(f"Final Result: Exact Result Indicies: {min_indicies}, Exact MSE: {min_MSE}\n")

max_MSE = -np.inf
max_indices = [0]*k
for index_comb in np_combinations:
	comb_MSE = calculate_MSE(n, X_train, y_train, index_comb)
	if comb_MSE > max_MSE:
		print(f"Exact Result: {index_comb}, exact MSE: {comb_MSE}")
		max_MSE = comb_MSE
		max_indicies = index_comb

print(f"Sanity Test: Max MSE Indicies: {max_indicies}, Exact MSE: {max_MSE}")

