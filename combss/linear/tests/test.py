#%%
import pandas as pd
import numpy as np

def gen_lam_max(A):
	b_k = np.random.rand(A.shape[1])

	for _ in range(1000):
		b_ki = np.dot(A, b_k)

		b_ki_norm = np.linalg.norm(b_ki)
		b_k = b_ki / b_ki_norm

	eta_max = int(1.1*np.dot(b_k, np.dot(A, b_k))**2)
	return eta_max

n = 100
p = 1000
snr = 4
path_train = './n-%s-p%sSNR-%sTrain.csv' %(n, p, snr)
df = pd.read_csv(path_train, sep='\t', header=None)
data = df.to_numpy()
X = data[:, 1:]

gen_lam_max(X)

# %%
