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
print("dec")

import optimize

# Parameters copied from prediction_main.py
t_init= []
delta_frac = 1
n = 100
p = 1000
snr = 1
nlam = 100
eta = 0.001
case = 1
q = 20
corr = 0.8

bulk_results = np.zeros((1, 100))

dir = os.getcwd()
# path_train = '../n-%s-p%sSNR-%s.csv' %(n,p,snr)
path_train = dir + '/n-100-p1000SNR-4Train.csv'
df = pd.read_csv(path_train, sep='\t', header=None)
data = df.to_numpy()
y_train = data[:, 0]
X_train = data[:, 1:]

path_test = dir + '/n-100-p1000SNR-4Train.csv'
df = pd.read_csv(path_test, sep='\t', header=None)
data = df.to_numpy()
y_test = data[:, 0]
X_test = data[:, 1:]

n_tinit = len(t_init)
running_time = 0

if n_tinit == 0:
	t_init = [np.ones(p)*0.5]
	n_tinit = 1

br_result = optimize.BR_combss(X_train, y_train, t_init=t_init[0], k = 15, delta_frac=delta_frac)
s = br_result > 0.9

print(f"Boolean Relaxation: {br_result}, Chosen Vectors: {s}, Sum of S: {sum(s)}")
