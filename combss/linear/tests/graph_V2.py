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

snr_res = {}

snr_low_list = [3, 5, 7]
for snr in snr_low_list:
    df0 = pd.read_csv("./sim_scenario/version0/COMBSSV0-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))
    
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


    df1 = pd.read_csv("./sim_scenario/scenario1/COMBSSV2-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))

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


    df2 = pd.read_csv("./sim_scenario/scenario2/COMBSSV2-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))
    
    version2 = {}
    avg_MSEV2 = df2['MSE'].mean()
    avg_PEV2 = df2['PE'].mean()
    avg_MCCV2 = df2['MCC'].mean()
    avg_AccuracyV2 = df2['Accuracy'].mean()
    avg_SensitivityV2 = df2['Sensitivity'].mean()
    avg_SpecificityV2 = df2['Specificity'].mean()
    avg_F1V2 = df2['F1_score'].mean()
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

    df3 = pd.read_csv("./sim_scenario/scenario3/COMBSSV2-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))
    
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

    df4 = pd.read_csv("./sim_scenario/scenario4/COMBSSV2-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))
    
    version4 = {}
    avg_MSEV4 = df4['MSE'].mean()
    avg_PEV4 = df4['PE'].mean()
    avg_MCCV4 = df4['MCC'].mean()
    avg_AccuracyV4 = df4['Accuracy'].mean()
    avg_SensitivityV4 = df4['Sensitivity'].mean()
    avg_SpecificityV4 = df4['Specificity'].mean()
    avg_F1V4 = df4['F1_score'].mean()
    avg_PrecisionV4 = df4['Precision'].mean()
    avg_TimeV4 = df4['Time'].mean()
    avg_LambdaV4 = df4['Opt Lambda'].mean()

    version4.update({"MSE": avg_MSEV4})
    version4.update({"PE": avg_PEV4})
    version4.update({"MCC": avg_MCCV4})
    version4.update({"Accuracy": avg_AccuracyV4})
    version4.update({"Sensitivity": avg_SensitivityV4})
    version4.update({"Specificity": avg_SpecificityV4})
    version4.update({"F1": avg_F1V4})
    version4.update({"Precision": avg_PrecisionV4})
    version4.update({"Time": avg_TimeV4})

    df5 = pd.read_csv("./bulk_sim_res/COMBSSV1-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))
    
    version5 = {}
    avg_MSEV5 = df5['MSE'].mean()
    avg_PEV5 = df5['PE'].mean()
    avg_MCCV5 = df5['MCC'].mean()
    avg_AccuracyV5 = df5['Accuracy'].mean()
    avg_SensitivityV5 = df5['Sensitivity'].mean()
    avg_SpecificityV5 = df5['Specificity'].mean()
    avg_F1V5 = df5['F1_score'].mean()
    avg_PrecisionV5 = df5['Precision'].mean()
    avg_TimeV5 = df5['Time'].mean()
    avg_LambdaV5 = df5['Opt Lambda'].mean()

    version5.update({"MSE": avg_MSEV5})
    version5.update({"PE": avg_PEV5})
    version5.update({"MCC": avg_MCCV5})
    version5.update({"Accuracy": avg_AccuracyV5})
    version5.update({"Sensitivity": avg_SensitivityV5})
    version5.update({"Specificity": avg_SpecificityV5})
    version5.update({"F1": avg_F1V5})
    version5.update({"Precision": avg_PrecisionV5})
    version5.update({"Time": avg_TimeV5})

    snr_res.update({snr: [version0, version1, version2, version3, version4, version5]})




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

mseV4 = []
peV4 = []
mccV4 = []
accuracyV4 = []
sensitivityV4 = []
specificityV4 = []
f1V4 = []
precisionV4 = []
timeV4 = []
lambdaV4 = []

mseV5 = []
peV5 = []
mccV5 = []
accuracyV5 = []
sensitivityV5 = []
specificityV5 = []
f1V5 = []
precisionV5 = []
timeV5 = []
lambdaV5 = []

snr_list = snr_res.keys()

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

    mseV4.append(snr[4].get("MSE"))
    peV4.append(snr[4].get("PE"))
    mccV4.append(snr[4].get("MCC"))
    accuracyV4.append(snr[4].get("Accuracy"))
    sensitivityV4.append(snr[4].get("Sensitivity"))
    specificityV4.append(snr[4].get("Specificity"))
    f1V4.append(snr[4].get("F1"))
    precisionV4.append(snr[4].get("Precision"))
    timeV4.append(snr[4].get("Time"))

    mseV5.append(snr[5].get("MSE"))
    peV5.append(snr[5].get("PE"))
    mccV5.append(snr[5].get("MCC"))
    accuracyV5.append(snr[5].get("Accuracy"))
    sensitivityV5.append(snr[5].get("Sensitivity"))
    specificityV5.append(snr[5].get("Specificity"))
    f1V5.append(snr[5].get("F1"))
    precisionV5.append(snr[5].get("Precision"))
    timeV5.append(snr[5].get("Time"))

#%%
# Plot MSE
fig, ax = plt.subplots()
combssv0, = plt.plot(snr_list, mseV0, label = "COMBSSV0", color = "black", marker='x')
combssc1, = plt.plot(snr_list, mseV1, label = "COMBSSV2: Full Convergence", color = "red", marker='x')
combssc2, = plt.plot(snr_list, mseV2, label = "COMBSSV2: cg_maxiter = 1, adam_maxiter = 1", color = "green", marker='x')
combssc3, = plt.plot(snr_list, mseV3, label = "COMBSSV2: cg_maxiter = 10, adam_maxiter = 10", color = "dodgerblue", marker='x')
combssc4, = plt.plot(snr_list, mseV4, label = "COMBSSV2: cg_maxiter = n, adam_maxiter = 1", color = "yellow", marker='x')
combssv1, = plt.plot(snr_list, mseV5, label = "COMBSSV1", color = "purple", marker='x')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('MSE')
plt.title('COMBSSV0 vs COMBSSV2 Variants')
plt.legend(handles = [combssv0, combssc1, combssc2, combssc3, combssc4, combssv1])
ax.set_ylim(bottom=0)
plt.show()

#%%
# Plot Prediction Error
fig, ax = plt.subplots()
combssv0, = plt.plot(snr_list, peV0, label = "COMBSSV0", color = "black", marker='x')
combssc1, = plt.plot(snr_list, peV1, label = "COMBSSV2: Full Convergence", color = "red", marker='x')
combssc2, = plt.plot(snr_list, peV2, label = "COMBSSV2: cg_maxiter = 1, adam_maxiter = 1", color = "green", marker='x')
combssc3, = plt.plot(snr_list, peV3, label = "COMBSSV2: cg_maxiter = 10, adam_maxiter = 10", color = "dodgerblue", marker='x')
combssc4, = plt.plot(snr_list, peV4, label = "COMBSSV2: cg_maxiter = n, adam_maxiter = 1", color = "yellow", marker='x')
combssv1, = plt.plot(snr_list, peV5, label = "COMBSSV1", color = "purple", marker='x')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Prediction Error')
plt.title('COMBSSV0 vs COMBSSV2 Variants')
plt.legend(handles = [combssv0, combssc1, combssc2, combssc3, combssc4, combssv1])
ax.set_ylim(bottom=0)
plt.show()

#%%
# Plot MCC
fig, ax = plt.subplots()
combssv0, = plt.plot(snr_list, mccV0, label = "COMBSSV0", color = "black", marker='x')
combssc1, = plt.plot(snr_list, mccV1, label = "COMBSSV2: Full Convergence", color = "red", marker='x')
combssc2, = plt.plot(snr_list, mccV2, label = "COMBSSV2: cg_maxiter = 1, adam_maxiter = 1", color = "green", marker='x')
combssc3, = plt.plot(snr_list, mccV3, label = "COMBSSV2: cg_maxiter = 10, adam_maxiter = 10", color = "dodgerblue", marker='x')
combssc4, = plt.plot(snr_list, mccV4, label = "COMBSSV2: cg_maxiter = n, adam_maxiter = 1", color = "yellow", marker='x')
combssv1, = plt.plot(snr_list, mccV5, label = "COMBSSV1", color = "purple", marker='x')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('MCC')
plt.title('COMBSSV0 vs COMBSSV2 Variants')
plt.legend(handles = [combssv0, combssc1, combssc2, combssc3, combssc4, combssv1])
ax.set_ylim(bottom=0)
plt.show()

#%%
# Plot Accuracy
fig, ax = plt.subplots()
combssv0, = plt.plot(snr_list, accuracyV0, label = "COMBSSV0", color = "black", marker='x')
combssc1, = plt.plot(snr_list, accuracyV1, label = "COMBSSV2: Full Convergence", color = "red", marker='x')
combssc2, = plt.plot(snr_list, accuracyV2, label = "COMBSSV2: cg_maxiter = 1, adam_maxiter = 1", color = "green", marker='x')
combssc3, = plt.plot(snr_list, accuracyV3, label = "COMBSSV2: cg_maxiter = 10, adam_maxiter = 10", color = "dodgerblue", marker='x')
combssc4, = plt.plot(snr_list, accuracyV4, label = "COMBSSV2: cg_maxiter = n, adam_maxiter = 1", color = "yellow", marker='x')
combssv1, = plt.plot(snr_list, accuracyV5, label = "COMBSSV1", color = "purple", marker='x')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Accuracy')
plt.title('COMBSSV0 vs COMBSSV2 Variants')
plt.legend(handles = [combssv0, combssc1, combssc2, combssc3, combssc4, combssv1])
ax.set_ylim(bottom=0)
ax.set_ylim(top=1)
plt.show()

#%%
# Plot F1 Score
fig, ax = plt.subplots()
combssv0, = plt.plot(snr_list, f1V0, label = "COMBSSV0", color = "black", marker='x')
combssc1, = plt.plot(snr_list, f1V1, label = "COMBSSV2: Full Convergence", color = "red", marker='x')
combssc2, = plt.plot(snr_list, f1V2, label = "COMBSSV2: cg_maxiter = 1, adam_maxiter = 1", color = "green", marker='x')
combssc3, = plt.plot(snr_list, f1V3, label = "COMBSSV2: cg_maxiter = 10, adam_maxiter = 10", color = "dodgerblue", marker='x')
combssc4, = plt.plot(snr_list, f1V4, label = "COMBSSV2: cg_maxiter = n, adam_maxiter = 1", color = "yellow", marker='x')
combssv1, = plt.plot(snr_list, f1V5, label = "COMBSSV1", color = "purple", marker='x')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('F1 Score')
plt.title('COMBSSV0 vs COMBSSV2 Variants')
plt.legend(handles = [combssv0, combssc1, combssc2, combssc3, combssc4, combssv1])
ax.set_ylim(bottom=0)
ax.set_ylim(top=1)
plt.show()

#%%
# Plot Sensitivity
fig, ax = plt.subplots()
combssv0, = plt.plot(snr_list, sensitivityV0, label = "COMBSSV0", color = "black", marker='x')
combssc1, = plt.plot(snr_list, sensitivityV1, label = "COMBSSV2: Full Convergence", color = "red", marker='x')
combssc2, = plt.plot(snr_list, sensitivityV2, label = "COMBSSV2: cg_maxiter = 1, adam_maxiter = 1", color = "green", marker='x')
combssc3, = plt.plot(snr_list, sensitivityV3, label = "COMBSSV2: cg_maxiter = 10, adam_maxiter = 10", color = "dodgerblue", marker='x')
combssc4, = plt.plot(snr_list, sensitivityV4, label = "COMBSSV2: cg_maxiter = n, adam_maxiter = 1", color = "yellow", marker='x')
combssv1, = plt.plot(snr_list, sensitivityV5, label = "COMBSSV1", color = "purple", marker='x')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Sensitivity')
plt.title('COMBSSV0 vs COMBSSV2 Variants')
plt.legend(handles = [combssv0, combssc1, combssc2, combssc3, combssc4, combssv1])
ax.set_ylim(bottom=0)
ax.set_ylim(top=1)
plt.show()

#%%
# Plot Specificity
fig, ax = plt.subplots()
combssv0, = plt.plot(snr_list, specificityV0, label = "COMBSSV0", color = "black", marker='x')
combssc1, = plt.plot(snr_list, specificityV1, label = "COMBSSV2: Full Convergence", color = "red", marker='x')
combssc2, = plt.plot(snr_list, specificityV2, label = "COMBSSV2: cg_maxiter = 1, adam_maxiter = 1", color = "green", marker='x')
combssc3, = plt.plot(snr_list, specificityV3, label = "COMBSSV2: cg_maxiter = 10, adam_maxiter = 10", color = "dodgerblue", marker='x')
combssc4, = plt.plot(snr_list, specificityV4, label = "COMBSSV2: cg_maxiter = n, adam_maxiter = 1", color = "yellow", marker='x')
combssv1, = plt.plot(snr_list, specificityV5, label = "COMBSSV1", color = "purple", marker='x')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Specificity')
plt.title('COMBSSV0 vs COMBSSV2 Variants')
plt.legend(handles = [combssv0, combssc1, combssc2, combssc3, combssc4, combssv1])
ax.set_ylim(bottom=0)
ax.set_ylim(top=1)
plt.show()

# %%

fig, ax = plt.subplots()
combssv0, = plt.plot(snr_list, timeV0, label = "COMBSSV0", color = "black", marker='x')
combssc1, = plt.plot(snr_list, timeV1, label = "COMBSSV2: Full Convergence", color = "red", marker='x')
combssc2, = plt.plot(snr_list, timeV2, label = "COMBSSV2: cg_maxiter = 1, adam_maxiter = 1", color = "green", marker='x')
combssc3, = plt.plot(snr_list, timeV3, label = "COMBSSV2: cg_maxiter = 10, adam_maxiter = 10", color = "dodgerblue", marker='x')
combssc4, = plt.plot(snr_list, timeV4, label = "COMBSSV2: cg_maxiter = n, adam_maxiter = 1", color = "yellow", marker='x')
combssv1, = plt.plot(snr_list, timeV5, label = "COMBSSV1", color = "purple", marker='x')

plt.xlabel('Signal to Noise Ratio')
plt.ylabel('Average Time per Dataset (seconds)')
plt.title('COMBSSV0 vs COMBSSV2 Variants')
plt.legend(handles = [combssv0, combssc1, combssc2, combssc3, combssc4, combssv1])
ax.set_ylim(bottom=0)
plt.show()
# %%
