#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun April 8 2024

@author: Hua Yang Hu
"""

def reset():
    try:
        get_ipython().magic('reset -sf')  #analysis:ignore
    except NameError:
        pass
reset()

#%%
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import cm
# import matplotlib.colors as mcolors

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import optimize
import metric
np.set_printoptions(suppress=True)

#%%
def bulk_sim_given_dataV0(n, p, q, beta_type, K0, snr, corr,
					 t_init = [],
					 delta_frac = 1,
					 n_datasets=1, 
					 nlam=50,
					 eta=0.01,
					 seed=1):
	metrics = ['MSE', 'PE', 'MCC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1_score', 'Precision', 'Time', 'Opt Lambda']
	names = metrics + [ str(i) for i in range(p)] 
	
	nmetrics = len(metrics)

	bulk_results = np.zeros((n_datasets, len(names)))
	
	for j in tqdm(range(n_datasets)):
		
		path_train = './n-%s-p%sSNR-%sTrain.csv' %(n, p, snr)
		df = pd.read_csv(path_train, sep='\t', header=None)
		data = df.to_numpy()
		y_train = data[:, 0]
		X_train = data[:, 1:]
		
		path_test= './n-%s-p%sSNR-%sTest.csv' %(n,p,snr)
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

		bulk_results[j, nmetrics + result1[0]] = 1
		bulk_results[j, 0] = result1[1]
		bulk_results[j, nmetrics-2] = running_time
		bulk_results[j, nmetrics-1] = result1[3]
		
		beta_pred = result1[2]
		
		beta_true = np.zeros(p)
		if beta_type == 1:
			beta_true[0:10] = 1
		elif beta_type == 2:
			beta_true[0:10] = np.array([0.5**k for k in range(10)])
		else: 
			print("Error: wrong case!")
			
		result2 = metric.performance_metrics(X_train, beta_true, beta_pred)
		bulk_results[j, 1: nmetrics-2] = np.array(result2)
		
	df = pd.DataFrame(bulk_results, columns = names)
	df.to_csv("./COMBSSV0-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))
	return df

#%%
""" 
Function for running COMBSS on several datasets and 
save the corresponding results in a file
"""
def bulk_sim_given_dataV1(n, p, q, beta_type, K0, snr, corr,
					 t_init = [],
					 delta_frac = 1,
					 n_datasets=1, 
					 nlam=50,
					 eta=0.01,
					 seed=1):
	metrics = ['MSE', 'PE', 'MCC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1_score', 'Precision', 'Time', 'Opt Lambda']
	names = metrics + [ str(i) for i in range(p)] 
	
	nmetrics = len(metrics)

	bulk_results = np.zeros((n_datasets, len(names)))
	
	for j in tqdm(range(n_datasets)):
		
		path_train = './n-%s-p%sSNR-%sTrain.csv' %(n, p, snr)
		df = pd.read_csv(path_train, sep='\t', header=None)
		data = df.to_numpy()
		y_train = data[:, 0]
		X_train = data[:, 1:]
		
		path_test= './n-%s-p%sSNR-%sTest.csv' %(n,p,snr)
		df = pd.read_csv(path_test, sep='\t', header=None)
		data = df.to_numpy()
		y_test = data[:, 0]
		X_test = data[:, 1:]

		n_tinit = len(t_init)
		running_time = 0
		
		if n_tinit == 0:
			t_init = [np.ones(p)*0.5]
			n_tinit = 1
		

		result1 = optimize.combssV1(X_train, y_train, X_test, y_test, t_init=t_init[0], 
						delta_frac=delta_frac,  q = q, nlam = nlam, eta=eta)
		"""
		Note that,
			result1 = [model_opt, mse_opt, beta_opt, lam_opt, time]
		"""
		running_time += result1[4]
		
		for i in range(n_tinit-1):
			
			result_temp = optimize.combssV1(X_train, y_train, X_test, y_test, t_init=t_init[i+1], 
								 delta_frac=delta_frac, q = q, nlam = nlam, eta=eta)
			
			running_time += result_temp[4]

			if result1[1] > result_temp[1]:
				result1 = result_temp

		bulk_results[j, nmetrics + result1[0]] = 1
		bulk_results[j, 0] = result1[1]
		bulk_results[j, nmetrics-2] = running_time
		bulk_results[j, nmetrics-1] = result1[3]
		
		beta_pred = result1[2]
		
		beta_true = np.zeros(p)
		if beta_type == 1:
			beta_true[0:10] = 1
		elif beta_type == 2:
			beta_true[0:10] = np.array([0.5**k for k in range(10)])
		else: 
			print("Error: wrong case!")
			
		result2 = metric.performance_metrics(X_train, beta_true, beta_pred)
		bulk_results[j, 1: nmetrics-2] = np.array(result2)
		
	df = pd.DataFrame(bulk_results, columns = names)
	df.to_csv("./COMBSSV1-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))
	return df

#%%
## Time to run the code....
# Specify the parameters here

n = 100
p = 1000
q = 20

beta_type = 2
K0 = 10
eta = 0.001

corr = 0.8

#t_init = [np.ones(p)*0.99, np.ones(p)*0.75, np.ones(p)*0.5, np.ones(p)*0.3]
t_init = [np.ones(p)*0.5]
ninit = len(t_init)
nlam = 100

delta_frac = 1
n_datasets = 1
#%%
snr = 4
print('SNR = ', snr)
dfV1 = bulk_sim_given_dataV0(n, p, q, beta_type, K0, snr, corr,
                     t_init = t_init,
                     delta_frac=delta_frac,
                     n_datasets=n_datasets, 
                     nlam=nlam,
                     eta = eta,
                     seed=1234)

#%%
snr = 4
print('SNR = ', snr)
dfV1 = bulk_sim_given_dataV1(n, p, q, beta_type, K0, snr, corr,
                     t_init = t_init,
                     delta_frac=delta_frac,
                     n_datasets=n_datasets, 
                     nlam=nlam,
                     eta = eta,
                     seed=1234)
#%%
"""
snr_low_list = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
snr_high_list = [2, 3, 4, 5, 6, 7, 8]
for snr in snr_high_list:
    print('SNR = ', snr)
    df = bulk_sim_given_data(n, p, q, beta_type, K0, snr, corr,
                         t_init = t_init,
                         delta_frac=delta_frac,
                         n_datasets=n_datasets, 
                         nlam=nlam,
                         eta = eta,
                         seed=1234)


"""
# %%
