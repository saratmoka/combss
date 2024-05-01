#%%
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

#%%
""" 
Function for running COMBSS on several datasets and 
save the corresponding results in a file
"""
def bulk_simV0(n, p, q, beta_type, K0, snr, corr,
					 t_init = [],
					 delta_frac = 1,
					 n_datasets=1, 
					 nlam=25,
					 eta=0.01,
					 seed=1):
	metrics = ['MSE', 'PE', 'MCC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1_score', 'Precision', 'Time', 'Opt Lambda']
	names = metrics + [ str(i) for i in range(p)] 
	
	nmetrics = len(metrics)

	bulk_results = np.zeros((n_datasets, len(names)))
	
	for j in tqdm(range(n_datasets)):
		
		path_train = './DATA/CASE%s/n-%s-p%sSNR-%sReplica%s.csv' %(beta_type, n, p, snr,j+1)
		df = pd.read_csv(path_train, sep='\t', header=None)
		data = df.to_numpy()
		y_train = data[:, 0]
		X_train = data[:, 1:]
		
		path_test= './DATA/CASE%s/n-%s-p%sSNR-%sTest-Replica%s.csv' %(beta_type, n,p,snr,j+1)
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
			
		result2 = metric.performance_metrics(X_test, beta_true, beta_pred)
		bulk_results[j, 1: nmetrics-2] = np.array(result2)
		
	df = pd.DataFrame(bulk_results, columns = names)
	df.to_csv("./bulk_sim_res/minisim/COMBSSV0-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))
	return

#%%
""" 
Function for running COMBSS on several datasets and 
save the corresponding results in a file
"""
def bulk_simV1(n, p, q, beta_type, K0, snr, corr,
					 t_init = [],
					 delta_frac = 1,
					 n_datasets=1, 
					 nlam=25,
					 eta=0.01,
					 seed=1):
	metrics = ['MSE', 'PE', 'MCC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1_score', 'Precision', 'Time', 'Opt Lambda']
	names = metrics + [ str(i) for i in range(p)] 
	
	nmetrics = len(metrics)

	bulk_results = np.zeros((n_datasets, len(names)))
	
	for j in tqdm(range(n_datasets)):
		
		path_train = './DATA/CASE%s/n-%s-p%sSNR-%sReplica%s.csv' %(beta_type, n, p, snr,j+1)
		df = pd.read_csv(path_train, sep='\t', header=None)
		data = df.to_numpy()
		y_train = data[:, 0]
		X_train = data[:, 1:]
		
		path_test= './DATA/CASE%s/n-%s-p%sSNR-%sTest-Replica%s.csv' %(beta_type, n,p,snr,j+1)
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
			
		result2 = metric.performance_metrics(X_test, beta_true, beta_pred)
		bulk_results[j, 1: nmetrics-2] = np.array(result2)
		
	df = pd.DataFrame(bulk_results, columns = names)
	df.to_csv("./bulk_sim_res/minisim/COMBSSV1-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))
	return
# %%

def bulk_simV2(n, p, q, beta_type, K0, snr, corr,
					 t_init = [],
					 delta_frac = 1,
					 n_datasets=1, 
					 nlam=25,
					 eta=0.01,
					 seed=1):
	metrics = ['MSE', 'PE', 'MCC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1_score', 'Precision', 'Time', 'Opt Lambda']
	names = metrics + [ str(i) for i in range(p)] 
	
	nmetrics = len(metrics)

	bulk_results = np.zeros((n_datasets, len(names)))
	
	for j in tqdm(range(n_datasets)):
		
		path_train = './DATA/CASE%s/n-%s-p%sSNR-%sReplica%s.csv' %(beta_type, n, p, snr,j+1)
		df = pd.read_csv(path_train, sep='\t', header=None)
		data = df.to_numpy()
		y_train = data[:, 0]
		X_train = data[:, 1:]
		
		path_test= './DATA/CASE%s/n-%s-p%sSNR-%sTest-Replica%s.csv' %(beta_type, n,p,snr,j+1)
		df = pd.read_csv(path_test, sep='\t', header=None)
		data = df.to_numpy()
		y_test = data[:, 0]
		X_test = data[:, 1:]

		n_tinit = len(t_init)
		running_time = 0
		
		if n_tinit == 0:
			t_init = [np.ones(p)*0.5]
			n_tinit = 1
		

		result1 = optimize.combssV2(X_train, y_train, X_test, y_test, t_init=t_init[0], 
						delta_frac=delta_frac,  q = q, nlam = nlam, eta=eta)
		"""
		Note that,
			result1 = [model_opt, mse_opt, beta_opt, lam_opt, time]
		"""
		running_time += result1[4]
		
		for i in range(n_tinit-1):
			
			result_temp = optimize.combssV2(X_train, y_train, X_test, y_test, t_init=t_init[i+1], 
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
			
		result2 = metric.performance_metrics(X_test, beta_true, beta_pred)
		bulk_results[j, 1: nmetrics-2] = np.array(result2)
		
	df = pd.DataFrame(bulk_results, columns = names)
	df.to_csv("./bulk_sim_res/minisim/COMBSSV2-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))
	return