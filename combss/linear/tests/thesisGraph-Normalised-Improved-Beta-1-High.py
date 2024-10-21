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

snr_res = {}

snr_low_list = [2, 3, 4, 5, 6, 7, 8]
for snr in snr_low_list:
    df0 = pd.read_csv("./finalResults/COMBSSImproved-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))
    
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


    df1 = pd.read_csv("./finalResults/COMBSSV0-Normalised-Improved-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))

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

    snr_res.update({snr: [version0, version1]})

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

#%%
# Plot MSE
fig, ax = plt.subplots()
combssO, = plt.plot(snr_list, mseV0, label = "COMBSS Improved Gradient", color = "black", marker='x')
combssN, = plt.plot(snr_list, mseV1, label = "COMBSS Normalised Improved Gradient", color = "red", marker='x')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('MSE')
plt.title('COMBSS Improved Gradient vs COMBSS Normalised Improved Gradient MSE, Beta Case 1 Low Dimension')
plt.legend(handles = [combssO, combssN])
ax.set_ylim(bottom=0)
plt.show()

#%%
# Plot Prediction Error
fig, ax = plt.subplots()
combssO, = plt.plot(snr_list, peV0, label = "COMBSS Improved Gradient", color = "black", marker='x')
combssN, = plt.plot(snr_list, peV1, label = "COMBSS Normalised Improved Gradient", color = "red", marker='x')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Prediction Error')
plt.title('COMBSS Improved Gradient vs COMBSS Normalised Improved Gradient PE, Beta Case 1 Low Dimension')
plt.legend(handles = [combssO, combssN])
ax.set_ylim(bottom=0)
plt.show()

#%%
# Plot MCC
fig, ax = plt.subplots()
combssO, = plt.plot(snr_list, mccV0, label = "COMBSS Improved Gradient", color = "black", marker='x')
combssN, = plt.plot(snr_list, mccV1, label = "COMBSS Normalised Improved Gradient", color = "red", marker='x')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('MCC')
plt.title('COMBSS Improved Gradient vs COMBSS Normalised Improved Gradient MCC, Beta Case 1 Low Dimension')
plt.legend(handles = [combssO, combssN])
ax.set_ylim(bottom=0)
plt.show()

#%%
# Plot F1 Score
fig, ax = plt.subplots()
combssO, = plt.plot(snr_list, f1V0, label = "COMBSS Improved Gradient", color = "black", marker='x')
combssN, = plt.plot(snr_list, f1V1, label = "COMBSS Normalised Improved Gradient", color = "red", marker='x')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('F1 Score')
plt.title('COMBSS Improved Gradient vs COMBSS Normalised Improved Gradient F1 Score, Beta Case 1 Low Dimension')
plt.legend(handles = [combssO, combssN])
ax.set_ylim(bottom=0)
ax.set_ylim(top=1)
plt.show()

#%%
# Plot Sensitivity
fig, ax = plt.subplots()
combssO, = plt.plot(snr_list, sensitivityV0, label = "COMBSS Improved Gradient", color = "black", marker='x')
combssN, = plt.plot(snr_list, sensitivityV1, label = "COMBSS Normalised Improved Gradient", color = "red", marker='x')
plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Sensitivity')
plt.title('COMBSS Improved Gradient vs COMBSS Normalised Improved Gradient Sensitivity, Beta Case 1 Low Dimension')
plt.legend(handles = [combssO, combssN])
ax.set_ylim(bottom=0)
ax.set_ylim(top=1)
plt.show()

#%%
# Plot Specificity
fig, ax = plt.subplots()
combssO, = plt.plot(snr_list, specificityV0, label = "COMBSS Improved Gradient", color = "black", marker='x')
combssN, = plt.plot(snr_list, specificityV1, label = "COMBSS Normalised Improved Gradient", color = "red", marker='x')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Specificity')
plt.title('COMBSS Improved Gradient vs COMBSS Normalised Improved Gradient Specificity, Beta Case 1 Low Dimension')
plt.legend(handles = [combssO, combssN])
ax.set_ylim(bottom=0)
ax.set_ylim(top=1)
plt.show()

# %%
# Plot Time
fig, ax = plt.subplots()
combssO, = plt.plot(snr_list, timeV0, label = "COMBSS Improved Gradient", color = "black", marker='x')
combssN, = plt.plot(snr_list, timeV1, label = "COMBSS Normalised Improved Gradient", color = "red", marker='x')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Average Time per Dataset (seconds)')
plt.title('COMBSS Improved Gradient vs COMBSS Normalised Improved Gradient Average Time, Beta Case 1 Low Dimension')
plt.legend(handles = [combssO, combssN])
ax.set_ylim(bottom=0)
plt.show()
# %%
