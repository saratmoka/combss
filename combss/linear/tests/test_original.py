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
p = 1000
snr = 4
nlam = 100
eta = 0.001
case = 1
q = 100
corr = 0.8

metrics = ['MSE', 'PE', 'MCC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1_score', 'Precision', 'Time', 'Opt Lambda']
names = metrics + [ str(i) for i in range(p)] 

nmetrics = len(metrics)

bulk_results = np.zeros((1, len(names)))

dir = os.getcwd()
# path_train = '../n-%s-p%sSNR-%s.csv' %(n,p,snr)
path_train = dir + '/n-100-p1000SNR-4Train.csv'
df = pd.read_csv(path_train, sep='\t', header=None)
data = df.to_numpy()
y_train = data[:, 0]
X_train = data[:, 1:]

path_test = dir + '/n-100-p1000SNR-4Test.csv'
df = pd.read_csv(path_test, sep='\t', header=None)
data = df.to_numpy()
y_test = data[:, 0]
X_test = data[:, 1:]

n_tinit = len(t_init)
running_time = 0

if n_tinit == 0:
	t_init = [np.ones(p)*0.5]
	n_tinit = 1

result1 = optimize.combssV0(X_train, y_train, X_test, y_test, t_init=t_init[0], 
				delta_frac=delta_frac,  q = q, nlam = nlam, eta=eta)
"""
Note that,
	result1 = [model_opt, mse_opt, beta_opt, lam_opt, time]
"""
running_time += result1[4]

for i in range(n_tinit-1):
	
	result_temp = optimize.combssV0(X_train, y_train, X_test, y_test, t_init=t_init[i+1], 
							delta_frac=delta_frac, q = q, nlam = nlam, eta=eta)
	
	running_time += result_temp[4]

	if result1[1] > result_temp[1]:
		result1 = result_temp

bulk_results[0, nmetrics + result1[0]] = 1
bulk_results[0, 0] = result1[1]
bulk_results[0, nmetrics-2] = running_time
bulk_results[0, nmetrics-1] = result1[3]

beta_pred = result1[2]

beta_true = np.zeros(p)
if case == 1:
	beta_true[0:10] = 1
elif case == 2:
	beta_true[0:10] = np.array([0.5**k for k in range(10)])
else: 
	print("Error: wrong case!")
	
result2 = metric.performance_metrics(X_train, beta_true, beta_pred)
bulk_results[0, 1: nmetrics-2] = np.array(result2)
	
df = pd.DataFrame(bulk_results, columns = names)

df.to_csv(dir + "/COMBSS-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(case, n, p, q, corr, n_tinit, snr,  1, nlam, eta))