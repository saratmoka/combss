#%%
import pandas as pd
import matplotlib.pyplot as plt

beta_type = 2
corr = 0.8

n = 100
nlam = 25
n_datasets = 50
eta = 0.001
n_tinit = 1

p = 1000
q = 20

snr_list = [2, 3, 4, 5, 6, 7, 8]
for snr in snr_list:
    df0 = pd.read_csv("./finalResults/COMBSSV0-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))
    df1 = pd.read_csv("./finalResults/COMBSSImproved-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))


    MSEV0 = df0['MSE']
    MSEV1 = df1['MSE']
    print(f'The MSE are identical: {MSEV0.equals(MSEV1)}')

    PEV0 = df0['PE']
    PEV1 = df1['PE']
    print(f'The PE are identical: {PEV0.equals(PEV1)}')


    MCCV0 = df0['MCC']
    MCCV1 = df1['MCC']
    print(f'The MCC are identical: {MCCV0.equals(MCCV1)}')


    AccuracyV0 = df0['Accuracy']
    AccuracyV1 = df1['Accuracy']
    print(f'The Accuracy are identical: {AccuracyV0.equals(AccuracyV1)}')


    SensitivityV0 = df0['Sensitivity']
    SensitivityV1 = df1['Sensitivity']
    print(f'The Sensitivity are identical: {SensitivityV0.equals(SensitivityV1)}')


    SpecificityV0 = df0['Specificity']
    SpecificityV1 = df1['Specificity']
    print(f'The Specificity are identical: {SpecificityV0.equals(SpecificityV1)}')


    F1V0 = df0['F1_score']
    F1V1 = df1['F1_score']
    print(f'The F1_score are identical: {F1V0.equals(F1V1)}')

    PrecisionV0 = df0['Precision']
    PrecisionV1 = df1['Precision']
    print(f'The Precision are identical: {PrecisionV0.equals(PrecisionV1)}')

    TimeV0 = df0['Time']
    TimeV1 = df1['Time']
    print(f'The Time are identical: {TimeV0.equals(TimeV1)}')


# %%
