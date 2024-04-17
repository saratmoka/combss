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
import metric, optimize

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

metrics = ['MSE', 'PE', 'MCC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1_score', 'Precision', 'Time', 'Opt Lambda']
names = metrics + [ str(i) for i in range(p)] 

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


result = optimize.adam_v1(X_train, y_train, gam1 = 0.9, gam2 = 0.999, alpha = 0.1, epsilon = 10e-8, maxiter = 1e3, tol = 1e-2, tau = 0.5)
"""
Note that
	result = [beta_pred, t_pred, model, converge, i]
"""

beta_pred = result[0]
t_pred = result[1]
model_pred = result[2]
converge = result[3]
i = result[4]

beta_true = np.zeros(p)
if case == 1:
	beta_true[0:10] = 1
elif case == 2:
	beta_true[0:10] = np.array([0.5**k for k in range(10)])
else: 
	print("Error: wrong case!")
	
print(f"Predicted Betas: {beta_pred}")
print(f"Predicted T: {t_pred}")
print(f"Predicted Model: {model_pred}")
print(f"Predicted i: {i}")