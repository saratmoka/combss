#%%

from tqdm import tqdm
import pandas as pd
from numpy.linalg import pinv, norm

beta_type = 1
n = 100
p = 1000
snr_list = [2,3,4,5,6,7,8]

n_datasets = 50

full = 0
null = 0
for snr in snr_list:
    for j in tqdm(range(n_datasets)):
            
            path_train = './DATA/CASE%s/n-%s-p%sSNR-%sReplica%s.csv' %(beta_type, n, p, snr,j+1)
            df = pd.read_csv(path_train, sep='\t', header=None)
            data = df.to_numpy()
            y = data[:, 0]
            x = data[:, 1:]

            delta = n

            beta = pinv(x.T@x)@(x.T@y)

            fullMod = 1/n*(y-x@beta).T@(y-x@beta) + p/n*(y.T@y)
            nullMod = 1/n*y.T@y

            print(f'full model: {fullMod}')
            print(f'null model: {nullMod}')

            if fullMod > nullMod:
                full += 1
            elif nullMod > fullMod:
                null += 1

print(f'full: {full/(full + null)}')
print(f'null: {null/(full + null)}')







# %%
