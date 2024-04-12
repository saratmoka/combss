#%%
import pandas as pd

pairing_type = 1
beta_type = 2
corr = 0.8

n = 100
nlam = 25
n_datasets = 50
eta = 0.001
n_tinit = 1

if pairing_type == 1:
    p = 20
    q = 20
else:
    p = 1000
    q = 1000


# snr_low_list = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
snr_high_list = [2, 3, 4, 5, 6, 7, 8]
for snr in snr_high_list:
    df0 = pd.read_csv("./bulk_sim_res/COMBSSV0-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))

    avg_MSEV0 = df0['MSE'].mean()
    avg_PEV0 = df0['PE'].mean()
    avg_MCCV0 = df0['MCC'].mean()
    avg_AccuracyV0 = df0['Accuracy'].mean()
    avg_SensitivityV0 = df0['Sensitivity'].mean()
    avg_SpecificityV0 = df0['Specificity'].mean()
    avg_F1V0 = df0['F1_score'].mean()
    avg_PrecisionV0 = df0['Precision'].mean()
    avg_TimeV0 = df0['Time'].mean()
    avg_LambdaV0 = df0['Opt Lambda'].mean()

    df1 = pd.read_csv("./bulk_sim_res/COMBSSV1-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))

    avg_MSEV0 = df1['MSE'].mean()
    avg_PEV0 = df1['PE'].mean()
    avg_MCCV0 = df1['MCC'].mean()
    avg_AccuracyV0 = df1['Accuracy'].mean()
    avg_SensitivityV0 = df1['Sensitivity'].mean()
    avg_SpecificityV0 = df1['Specificity'].mean()
    avg_F1V0 = df1['F1_score'].mean()
    avg_PrecisionV0 = df1['Precision'].mean()
    avg_TimeV0 = df1['Time'].mean()
    avg_LambdaV0 = df1['Opt Lambda'].mean()









