#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def obj_fun(t1, t2, y, X, n, p, lam):
    t = np.ones(p)*0.5
    t[0] = t1
    t[1] = t2

    Xt = X*t

    delta = n
    Lt = 1/n*(Xt.T@Xt + delta*(1-t**2))

    Bt = np.linalg.inv(Lt)@(Xt.T@y/n)

    return 1/n*np.linalg.norm((y - Xt@Bt))**2 + lam*np.sum(t)

def sigmoid(t):
    return 1 / (1 + np.exp(-t))


n = 100
p = 100

m = np.zeros(p)
m[0] = 1

# Generate random Gaussian data for x (100x2 matrix)
X = np.random.normal(size=(100, 100))

# Calculate the signal y = x @ m (matrix multiplication)
signal = X @ m

# Calculate the power of the signal (mean squared value)
signal_power = np.mean(signal**2)

# Define the desired signal-to-noise ratio (SNR = 3)
snr = 0.5

# Calculate the noise power (signal power / SNR)
noise_power = signal_power / snr

# Generate Gaussian noise with the calculated noise power
noise = np.random.normal(scale=np.sqrt(noise_power), size=signal.shape)

# Add the noise to the signal
y = signal + noise

lambda_list = [0,2]


t1_min = 0.01
t1_max = 1
t2_min = 0.01
t2_max = 1

t1_range = np.arange(t1_min, t1_max, 0.01)
t2_range = np.arange(t2_min, t2_max, 0.01)


t1_grid, t2_grid = np.meshgrid(t1_range, t2_range)
# loss_grid = obj_fun(t1=t1_grid, t2=t2_grid, y=y, X=X, n=n, p=p, lam = 1)

# Initialize loss grid
loss_grid = np.zeros_like(t1_grid)

# Loop over the grid to compute loss values

for lam in lambda_list:
    for i in range(t1_grid.shape[0]):
        for j in range(t2_grid.shape[1]):
            t1 = t1_grid[i, j]
            t2 = t2_grid[i, j]
            loss_grid[i, j] = obj_fun(t1=t1, t2=t2, y=y, X=X, n=n, p=p, lam=lam)

    loss_grid_df = pd.DataFrame(loss_grid)

    # Save the DataFrame to a CSV file
    loss_grid_df.to_csv(f"loss_grid-lam-{lam}.csv", index=False, header=False)

#%%
fig = plt.figure()
ax = plt.axes(projection='3d')

lam = 0
loss_grid_df = pd.read_csv(f"loss_grid-lam-{lam}.csv", header=None)

# If you want to convert the DataFrame back to a numpy array
loss_grid = loss_grid_df.values

# Create a grid of t1 and t2 values
t1 = np.linspace(0, 1, 100)
t2 = np.linspace(0, 1, 100)


T1, T2 = np.meshgrid(t1, t2)
Z = loss_grid[:2,]

# Create the figure and 3D axes
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(T1, T2, Z, cmap='Blues', edgecolor='white')

# Set labels
ax.set_xlabel(r'$\tilde{t}_1$', fontsize=10)
ax.set_ylabel(r'$t_2$', fontsize=10)
ax.set_zlabel(r'$f_\lambda(\boldsymbol{t})$', fontsize=10)

# Customize view angle
ax.view_init(elev=30, azim=-60)

# Show the plot
plt.tight_layout()
plt.show()
# %%
