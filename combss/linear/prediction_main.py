#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 10:22:47 2023

@author: Sarat Babu Moka
w/ additions by Hua Yang Hu, January 2024
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

from prediction_helper import *
import time
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)

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
"""
snr_low_list = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
snr_high_list = [2, 3, 4, 5, 6, 7, 8]
for snr in snr_low_list:
    print('SNR = ', snr)
    df = bulk_sim_given_data(n, p, q, beta_type, K0, snr, corr,
                         t_init = t_init,
                         delta_frac=delta_frac,
                         n_datasets=n_datasets, 
                         nlam=nlam,
                         eta = eta,
                         seed=1234)


"""

'''
#%%

snr = 5
print('SNR = ', snr)
df = bulk_sim_given_data(n, p, q, beta_type, K0, snr, corr,
                     t_init = t_init,
                     delta_frac=delta_frac,
                     n_datasets=n_datasets, 
                     nlam=nlam,
                     eta = eta,
                     seed=1234)
'''