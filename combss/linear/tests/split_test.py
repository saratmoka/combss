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
import numpy as np
import bulk_sim

np.set_printoptions(suppress=True)

# %%
n = 100
p = 20
q = 20

K0 = 10
eta = 0.001

corr = 0.8

#t_init = [np.ones(p)*0.99, np.ones(p)*0.75, np.ones(p)*0.5, np.ones(p)*0.3]
t_init = [np.ones(p)*0.5]
ninit = len(t_init)
nlam = 25

delta_frac = 1
n_datasets = 25

beta_list = [1]

snr_high_list = [2,3,4,5,6,7,8]
for snr in snr_high_list:

	for beta_type in beta_list:
	
		bulk_sim.bulk_simV6(n, p, q, beta_type, K0, snr, corr,
							t_init = t_init,
							delta_frac=delta_frac,
							n_datasets=n_datasets, 
							nlam=nlam,
							eta = eta,
							seed=1234)


'''
	   	bulk_sim.bulk_simV0(n, p, q, beta_type, K0, snr, corr, t_init = t_init, delta_frac=delta_frac, n_datasets=n_datasets, nlam=nlam, eta = eta, seed=1234)

'''	   
		
		

	    


# %%
