#%%
import pandas as pd
import matplotlib.pyplot as plt

beta_type = 1
corr = 0.8

n = 100
pL = 20

pH = 1000

nlam = 25
n_datasets = 50
eta = 0.001
n_tinit = 1

q = 20

snr_list = [2, 3, 4, 5, 6, 7, 8]
for snr in snr_list:
    dfL = pd.read_csv("./finalResults/COMBSSImproved-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, pL, q, corr, n_tinit, snr,  n_datasets, nlam, eta))
    dfH = pd.read_csv("./finalResults/COMBSSImproved-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, pH, q, corr, n_tinit, snr,  n_datasets, nlam, eta))

    TimeV0 = dfL['Time'].mean()
    TimeV1 = dfH['Time'].mean()
    print(f'SNR: {snr}, Low Dimension Average Time: {TimeV0}, High Dimension Average Time: {TimeV1}')



# %%
