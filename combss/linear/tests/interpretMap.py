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

p = 20
q = 20

df0 = pd.read_csv("./COMBSS-Original-case-1-n-100-p-20-q-20-corr-0.8-ninit-1-snr-4-ndatasets-50-nlam-25-eta-0.001.csv")
df1 = pd.read_csv("./COMBSS-Map-case-1-n-100-p-20-q-20-corr-0.8-ninit-1-snr-4-ndatasets-50-nlam-25-eta-0.001.csv")

axes = ["Sigmoid Mapping", "Original Mapping"]

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

mse = [avg_MSEV0, avg_MSEV1]
pe = [avg_PEV0, avg_PEV1]
mcc = [avg_MCCV0, avg_MCCV1]
accuracy = [avg_AccuracyV0, avg_AccuracyV1]
sensitivity = [avg_SensitivityV0, avg_SensitivityV1]
specificity = [avg_SpecificityV0, avg_SpecificityV1]
f1 = [avg_F1V0, avg_F1V1]

#%%
# Plot MSE
plt.bar(axes, mse, color = "skyblue")
plt.xlabel('Mapping Types')
plt.ylabel('MSE')
plt.title('Sigmoid vs Original Mapping: MSE')
plt.show()

#%%
# Plot Prediction Error
plt.bar(axes, pe, color = "skyblue")
plt.xlabel('Mapping Types')
plt.ylabel('Prediction Error')
plt.title('Sigmoid vs Original Mapping: Prediction Error')
plt.show()

#%%
# Plot MCC
plt.bar(axes, mcc, color = "skyblue")
plt.xlabel('Mapping Types')
plt.ylabel('Matthews Correlation Coefficient')
plt.title('Sigmoid vs Original Mapping: MCC')
plt.show()

#%%
# Plot Accuracy
plt.bar(axes, accuracy, color = "skyblue")
plt.xlabel('Mapping Types')
plt.ylabel('Accuracy')
plt.title('Sigmoid vs Original Mapping: Accuracy')
plt.show()

#%%
# Plot F1 Score
plt.bar(axes, f1, color = "skyblue")
plt.xlabel('Mapping Types')
plt.ylabel('F1 Score')
plt.title('Sigmoid vs Original Mapping: F1 Score')
plt.show()

#%%
# Plot Sensitivity
plt.bar(axes, sensitivity, color = "skyblue")
plt.xlabel('Mapping Types')
plt.ylabel('Sensitivity')
plt.title('Sigmoid vs Original Mapping: Sensitivity')
plt.show()

#%%
# Plot Specificity
plt.bar(axes, specificity, color = "skyblue")
plt.xlabel('Mapping Types')
plt.ylabel('Specificity')
plt.title('Sigmoid vs Original Mapping: Specificity')
plt.show()

# %%
df0 = pd.read_csv("./COMBSS-Original-case-1-n-100-p-1000-q-20-corr-0.8-ninit-1-snr-4-ndatasets-50-nlam-25-eta-0.001.csv")
df1 = pd.read_csv("./COMBSS-Map-case-1-n-100-p-1000-q-100-corr-0.8-ninit-1-snr-4-ndatasets-1-nlam-100-eta-0.001.csv")

axes = ["Sigmoid Mapping", "Original Mapping"]

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

mse = [avg_MSEV0, avg_MSEV1]
pe = [avg_PEV0, avg_PEV1]
mcc = [avg_MCCV0, avg_MCCV1]
accuracy = [avg_AccuracyV0, avg_AccuracyV1]
sensitivity = [avg_SensitivityV0, avg_SensitivityV1]
specificity = [avg_SpecificityV0, avg_SpecificityV1]
f1 = [avg_F1V0, avg_F1V1]

#%%
# Plot MSE
plt.bar(axes, mse, color = "skyblue")
plt.xlabel('Mapping Types')
plt.ylabel('MSE')
plt.title('Sigmoid vs Original Mapping: MSE')
plt.show()

#%%
# Plot Prediction Error
plt.bar(axes, pe, color = "skyblue")
plt.xlabel('Mapping Types')
plt.ylabel('Prediction Error')
plt.title('Sigmoid vs Original Mapping: Prediction Error')
plt.show()

#%%
# Plot MCC
plt.bar(axes, mcc, color = "skyblue")
plt.xlabel('Mapping Types')
plt.ylabel('Matthews Correlation Coefficient')
plt.title('Sigmoid vs Original Mapping: MCC')
plt.show()

#%%
# Plot Accuracy
plt.bar(axes, accuracy, color = "skyblue")
plt.xlabel('Mapping Types')
plt.ylabel('Accuracy')
plt.title('Sigmoid vs Original Mapping: Accuracy')
plt.show()

#%%
# Plot F1 Score
plt.bar(axes, f1, color = "skyblue")
plt.xlabel('Mapping Types')
plt.ylabel('F1 Score')
plt.title('Sigmoid vs Original Mapping: F1 Score')
plt.show()

#%%
# Plot Sensitivity
plt.bar(axes, sensitivity, color = "skyblue")
plt.xlabel('Mapping Types')
plt.ylabel('Sensitivity')
plt.title('Sigmoid vs Original Mapping: Sensitivity')
plt.show()

#%%
# Plot Specificity
plt.bar(axes, specificity, color = "skyblue")
plt.xlabel('Mapping Types')
plt.ylabel('Specificity')
plt.title('Sigmoid vs Original Mapping: Specificity')
plt.show()

# %%