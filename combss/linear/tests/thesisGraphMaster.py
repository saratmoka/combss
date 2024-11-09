#%%
import pandas as pd
import matplotlib.pyplot as plt

beta_type = 1
corr = 0.8

n = 100
nlam = 25
n_datasets = 50
eta = 0.001
n_tinit = 1

p = 1000
q = 20

snr_low_list = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
snr_high_list = [2, 3, 4, 5, 6, 7, 8]

if p == 20:
    dimension = 'Low'
    snr_list = snr_low_list
elif p == 1000:
    dimension = 'High'
    snr_list = snr_high_list


snr_res = {}

for snr in snr_list:
    df0 = pd.read_csv("./finalResults/COMBSSV0-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))
    
    version0 = {}
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

    version0.update({"MSE": avg_MSEV0})
    version0.update({"PE": avg_PEV0})
    version0.update({"MCC": avg_MCCV0})
    version0.update({"Accuracy": avg_AccuracyV0})
    version0.update({"Sensitivity": avg_SensitivityV0})
    version0.update({"Specificity": avg_SpecificityV0})
    version0.update({"F1": avg_F1V0})
    version0.update({"Precision": avg_PrecisionV0})
    version0.update({"Time": avg_TimeV0})


    df1 = pd.read_csv("./finalResults/COMBSS-normalised-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))

    version1 = {}
    avg_MSEV1 = df1['MSE'].mean()
    avg_PEV1 = df1['PE'].mean()
    avg_MCCV1 = df1['MCC'].mean()
    avg_AccuracyV1 = df1['Accuracy'].mean()
    avg_SensitivityV1 = df1['Sensitivity'].mean()
    avg_SpecificityV1 = df1['Specificity'].mean()
    avg_F1V1 = df1['F1_score'].mean()
    avg_PrecisionV1 = df1['Precision'].mean()
    avg_TimeV1 = df1['Time'].mean()
    avg_LambdaV1 = df1['Opt Lambda'].mean()

    version1.update({"MSE": avg_MSEV1})
    version1.update({"PE": avg_PEV1})
    version1.update({"MCC": avg_MCCV1})
    version1.update({"Accuracy": avg_AccuracyV1})
    version1.update({"Sensitivity": avg_SensitivityV1})
    version1.update({"Specificity": avg_SpecificityV1})
    version1.update({"F1": avg_F1V1})
    version1.update({"Precision": avg_PrecisionV1})
    version1.update({"Time": avg_TimeV1})

    df2 = pd.read_csv("./finalResults/COMBSSImproved-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))

    version2 = {}
    avg_MSEV2 = df2['MSE'].mean()
    avg_PEV2 = df2['PE'].mean()
    avg_MCCV2 = df2['MCC'].mean()
    avg_AccuracyV2 = df2['Accuracy'].mean()
    avg_SensitivityV2 = df2['Sensitivity'].mean()
    avg_SpecificityV2 = df2['Specificity'].mean()
    avg_F1V2= df2['F1_score'].mean()
    avg_PrecisionV2 = df2['Precision'].mean()
    avg_TimeV2 = df2['Time'].mean()
    avg_LambdaV2 = df2['Opt Lambda'].mean()

    version2.update({"MSE": avg_MSEV2})
    version2.update({"PE": avg_PEV2})
    version2.update({"MCC": avg_MCCV2})
    version2.update({"Accuracy": avg_AccuracyV2})
    version2.update({"Sensitivity": avg_SensitivityV2})
    version2.update({"Specificity": avg_SpecificityV2})
    version2.update({"F1": avg_F1V2})
    version2.update({"Precision": avg_PrecisionV2})
    version2.update({"Time": avg_TimeV2})

    df3 = pd.read_csv("./finalResults/COMBSSV0-Normalised-Improved-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))

    version3 = {}
    avg_MSEV3 = df3['MSE'].mean()
    avg_PEV3 = df3['PE'].mean()
    avg_MCCV3 = df3['MCC'].mean()
    avg_AccuracyV3 = df3['Accuracy'].mean()
    avg_SensitivityV3 = df3['Sensitivity'].mean()
    avg_SpecificityV3 = df3['Specificity'].mean()
    avg_F1V3 = df3['F1_score'].mean()
    avg_PrecisionV3 = df3['Precision'].mean()
    avg_TimeV3 = df3['Time'].mean()
    avg_LambdaV3 = df3['Opt Lambda'].mean()

    version3.update({"MSE": avg_MSEV3})
    version3.update({"PE": avg_PEV3})
    version3.update({"MCC": avg_MCCV3})
    version3.update({"Accuracy": avg_AccuracyV3})
    version3.update({"Sensitivity": avg_SensitivityV3})
    version3.update({"Specificity": avg_SpecificityV3})
    version3.update({"F1": avg_F1V3})
    version3.update({"Precision": avg_PrecisionV3})
    version3.update({"Time": avg_TimeV3})

    snr_res.update({snr: [version0, version1, version2, version3]})

#%%
# Plot Accuracy
mseV0 = []
peV0 = []
mccV0 = []
accuracyV0 = []
sensitivityV0 = []
specificityV0 = []
f1V0 = []
precisionV0 = []
timeV0 = []
lambdaV0 = []

mseV1 = []
peV1 = []
mccV1 = []
accuracyV1 = []
sensitivityV1 = []
specificityV1 = []
f1V1 = []
precisionV1 = []
timeV1 = []
lambdaV1 = []

mseV2 = []
peV2 = []
mccV2 = []
accuracyV2 = []
sensitivityV2 = []
specificityV2 = []
f1V2 = []
precisionV2 = []
timeV2 = []
lambdaV2 = []

mseV3 = []
peV3 = []
mccV3 = []
accuracyV3 = []
sensitivityV3 = []
specificityV3 = []
f1V3 = []
precisionV3 = []
timeV3 = []
lambdaV3 = []

snr_list = snr_res.keys()

#%%

for snr in snr_res.values():
    mseV0.append(snr[0].get("MSE"))
    peV0.append(snr[0].get("PE"))
    mccV0.append(snr[0].get("MCC"))
    accuracyV0.append(snr[0].get("Accuracy"))
    sensitivityV0.append(snr[0].get("Sensitivity"))
    specificityV0.append(snr[0].get("Specificity"))
    f1V0.append(snr[0].get("F1"))
    precisionV0.append(snr[0].get("Precision"))
    timeV0.append(snr[0].get("Time"))

    mseV1.append(snr[1].get("MSE"))
    peV1.append(snr[1].get("PE"))
    mccV1.append(snr[1].get("MCC"))
    accuracyV1.append(snr[1].get("Accuracy"))
    sensitivityV1.append(snr[1].get("Sensitivity"))
    specificityV1.append(snr[1].get("Specificity"))
    f1V1.append(snr[1].get("F1"))
    precisionV1.append(snr[1].get("Precision"))
    timeV1.append(snr[1].get("Time"))

    mseV2.append(snr[2].get("MSE"))
    peV2.append(snr[2].get("PE"))
    mccV2.append(snr[2].get("MCC"))
    accuracyV2.append(snr[2].get("Accuracy"))
    sensitivityV2.append(snr[2].get("Sensitivity"))
    specificityV2.append(snr[2].get("Specificity"))
    f1V2.append(snr[2].get("F1"))
    precisionV2.append(snr[2].get("Precision"))
    timeV2.append(snr[2].get("Time"))

    mseV3.append(snr[3].get("MSE"))
    peV3.append(snr[3].get("PE"))
    mccV3.append(snr[3].get("MCC"))
    accuracyV3.append(snr[3].get("Accuracy"))
    sensitivityV3.append(snr[3].get("Sensitivity"))
    specificityV3.append(snr[3].get("Specificity"))
    f1V3.append(snr[3].get("F1"))
    precisionV3.append(snr[3].get("Precision"))
    timeV3.append(snr[3].get("Time"))

#%%
# Plot MSE
fig, ax = plt.subplots()
combssO, = plt.plot(snr_list, mseV0, label = "COMBSS Original", color = "black", marker='o')
combssN, = plt.plot(snr_list, mseV1, label = "COMBSS Normalised", color = "red", marker='^')
combssI, = plt.plot(snr_list, mseV2, label = "COMBSS Improved Gradient", color = "green", marker='s')
combssIN, = plt.plot(snr_list, mseV3, label = "COMBSS Improved Gradient, Normalised", color = "dodgerblue", marker='H')


plt.xlabel('Signal to Noise Ratio')
plt.ylabel('MSE')
plt.title(f'COMBSS Variants Mean Squared Error, Beta Case {beta_type}, {dimension} Dimension')
plt.legend(handles = [combssO, combssN, combssI, combssIN])
ax.set_ylim(bottom=0)
plt.figure(dpi=1200)
plt.show()

#%%
# Plot Prediction Error
fig, ax = plt.subplots()
combssO, = plt.plot(snr_list, peV0, label = "COMBSS Original", color = "black", marker='o')
combssN, = plt.plot(snr_list, peV1, label = "COMBSS Normalised", color = "red", marker='^')
combssI, = plt.plot(snr_list, peV2, label = "COMBSS Improved Gradient", color = "green", marker='s')
combssIN, = plt.plot(snr_list, peV3, label = "COMBSS Improved Gradient, Normalised", color = "dodgerblue", marker='H')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Prediction Error')
plt.title(f'COMBSS Variants Prediction Error, Beta Case {beta_type}, {dimension} Dimension')
plt.legend(handles = [combssO, combssN, combssI, combssIN])
ax.set_ylim(bottom=0)
plt.figure(dpi=1200)
plt.show()

#%%
# Plot MCC
fig, ax = plt.subplots()
combssO, = plt.plot(snr_list, mccV0, label = "COMBSS Original", color = "black", marker='o')
combssN, = plt.plot(snr_list, mccV1, label = "COMBSS Normalised", color = "red", marker='^')
combssI, = plt.plot(snr_list, mccV2, label = "COMBSS Improved Gradient", color = "green", marker='s')
combssIN, = plt.plot(snr_list, mccV3, label = "COMBSS Improved Gradient, Normalised", color = "dodgerblue", marker='H')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('MCC')
plt.title(f'COMBSS Variants MCC, Beta Case {beta_type}, {dimension} Dimension')
plt.legend(handles = [combssO, combssN, combssI, combssIN])
ax.set_ylim(bottom=0)
plt.figure(dpi=1200)
plt.show()

#%%
# Plot Accuracy
fig, ax = plt.subplots()
combssO, = plt.plot(snr_list, accuracyV0, label = "COMBSS Original", color = "black", marker='o')
combssN, = plt.plot(snr_list, accuracyV1, label = "COMBSS Normalised", color = "red", marker='^')
combssI, = plt.plot(snr_list, accuracyV2, label = "COMBSS Improved Gradient", color = "green", marker='s')
combssIN, = plt.plot(snr_list, accuracyV3, label = "COMBSS Improved Gradient, Normalised", color = "dodgerblue", marker='H')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Accuracy')
plt.title(f'COMBSS Variants Accuracy, Beta Case {beta_type}, {dimension} Dimension')
plt.legend(handles = [combssO, combssN, combssI, combssIN])
ax.set_ylim(bottom=0)
ax.set_ylim(top=1)
plt.figure(dpi=1200)
plt.show()

#%%
# Plot F1 Score
fig, ax = plt.subplots()
combssO, = plt.plot(snr_list, f1V0, label = "COMBSS Original", color = "black", marker='o')
combssN, = plt.plot(snr_list, f1V1, label = "COMBSS Normalised", color = "red", marker='^')
combssI, = plt.plot(snr_list, f1V2, label = "COMBSS Improved Gradient", color = "green", marker='s')
combssIN, = plt.plot(snr_list, f1V3, label = "COMBSS Improved Gradient, Normalised", color = "dodgerblue", marker='H')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('F1 Score')
plt.title(f'COMBSS Variants F1 Score, Beta Case {beta_type}, {dimension} Dimension')
plt.legend(handles = [combssO, combssN, combssI, combssIN])
ax.set_ylim(bottom=0)
ax.set_ylim(top=1)
plt.figure(dpi=1200)
plt.show()

#%%
# Plot Sensitivity
fig, ax = plt.subplots()
combssO, = plt.plot(snr_list, sensitivityV0, label = "COMBSS Original", color = "black", marker='o')
combssN, = plt.plot(snr_list, sensitivityV1, label = "COMBSS Normalised", color = "red", marker='^')
combssI, = plt.plot(snr_list, sensitivityV2, label = "COMBSS Improved Gradient", color = "green", marker='s')
combssIN, = plt.plot(snr_list, sensitivityV3, label = "COMBSS Improved Gradient, Normalised", color = "dodgerblue", marker='H')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Sensitivity')
plt.title(f'COMBSS Variants Sensitivity, Beta Case {beta_type}, {dimension} Dimension')
plt.legend(handles = [combssO, combssN, combssI, combssIN])
ax.set_ylim(bottom=0)
ax.set_ylim(top=1)
plt.figure(dpi=1200)
plt.show()

#%%
# Plot Specificity
fig, ax = plt.subplots()
combssO, = plt.plot(snr_list, specificityV0, label = "COMBSS Original", color = "black", marker='o')
combssN, = plt.plot(snr_list, specificityV1, label = "COMBSS Normalised", color = "red", marker='^')
combssI, = plt.plot(snr_list, specificityV2, label = "COMBSS Improved Gradient", color = "green", marker='s')
combssIN, = plt.plot(snr_list, specificityV3, label = "COMBSS Improved Gradient, Normalised", color = "dodgerblue", marker='H')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Specificity')
plt.title(f'COMBSS Variants Specificity, Beta Case {beta_type}, {dimension} Dimension')
plt.legend(handles = [combssO, combssN, combssI, combssIN])
ax.set_ylim(bottom=0)
ax.set_ylim(top=1)
plt.figure(dpi=1200)
plt.show()

# %%

fig, ax = plt.subplots()
combssO, = plt.plot(snr_list, timeV0, label = "COMBSS Original", color = "black", marker='o')
combssN, = plt.plot(snr_list, timeV1, label = "COMBSS Normalised", color = "red", marker='^')
combssI, = plt.plot(snr_list, timeV2, label = "COMBSS Improved Gradient", color = "green", marker='s')
combssIN, = plt.plot(snr_list, timeV3, label = "COMBSS Improved Gradient, Normalised", color = "dodgerblue", marker='H')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Average Time per Dataset (seconds)')
plt.title(f'COMBSS Variants Average Time, Beta Case {beta_type}, {dimension} Dimension')
plt.legend(handles = [combssO, combssN, combssI, combssIN])
ax.set_ylim(bottom=0)
plt.figure(dpi=1200)
plt.show()
# %%
