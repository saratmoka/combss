import numpy as np
from numpy.linalg import norm

""" Generates a grid of lambda values for COMBSS, with lambda values exponentially increasing.

	Parameters
	----------
	y : array-like of shape (n_samples, 1)
        The response matrix, where `n_samples` is the number of samples observed.

    size : int
        The size of the grid of lambdas explored.

    c : float
        Parameter such that the ith element of lambda is dictated by lambda[i] = c^i.


	Returns
	-------
    lam_grid : array-like of floats
        The grid of lambda values generated such that lambda values exponentially increasing.
"""
def gen_lam_grid_exp(y, size, c):
    lam_max = norm(y)**2/(y.shape[0])  
    lam_grid = lam_max*np.array([c**i for i in range(size)])
    # lam_grid = np.append(lam_grid, 0.0)
    lam_grid = np.flip(lam_grid)
    
    return lam_grid


""" Generates a harmonic grid of lambda values for COMBSS, such that lambda[i] = 1/(c^i +1).

	Parameters
	----------
	y : array-like of shape (n_samples, 1)
        The response matrix, where `n_samples` is the number of samples observed.

    size : int
        The size of the grid of lambdas explored.

    c : float
        Parameter such that the ith element of lambda is dictated by lambda[i] = 1/(c^i +1).


	Returns
	-------
    lam_grid : array-like of floats
        The harmonic grid of lambda values generated such that lambda[i] = 1/(c^i +1).
"""
def gen_lam_grid_harm(y, size, c):
    lam_max = norm(y)**2/(y.shape[0])  
    lam_grid = lam_max*np.array([1/(c*i + 1) for i in range(size+1)])
    lam_grid = np.append(lam_grid, 0.0)
    lam_grid = np.flip(lam_grid)
    
    return lam_grid


""" Generates a square-rooted harmonic grid of lambda values for COMBSS, such that lambda[i] = 1/(c^sqrt(i) +1).

	Parameters
	----------
	y : array-like of shape (n_samples, 1)
        The response matrix, where `n_samples` is the number of samples observed.

    size : int
        The size of the grid of lambdas explored.

    c : float
        Parameter such that the ith element of lambda is dictated by lambda[i] = 1/(c^sqrt(i) +1).


	Returns
	-------
    lam_grid : array-like of floats
        The square-rooted harmonic grid of lambda values generated such that lambda[i] = 1/(c^sqrt(i) +1).
"""
def gen_lam_grid_sqrt_harm(y, size, c):
    lam_max = norm(y)**2/(y.shape[0])  
    lam_grid = lam_max*np.array([1/(c*np.sqrt(i) +1 ) for i in range(size+1)])
    lam_grid = np.flip(lam_grid)
    
    return lam_grid


""" Generates a linear grid of lambda values for COMBSS, such that lambda[i] = lambda[i-1] - d, 
    where d is a positive constant and lambda[size - 1] = 0.

	Parameters
	----------
	y : array-like of shape (n_samples, 1)
        The response matrix, where `n_samples` is the number of samples observed.

    size : int
        The size of the grid of lambdas explored.

	Returns
	-------
    lam_grid : array-like of floats
        The linear grid of lambda values generated such that lam_grid[i] = lam_grid[i-1] - d, 
        where d is a positive constant and lambda[size - 1] = 0.
"""
def gen_lam_grid_lin(y, size):
    lam_max = norm(y)**2/(y.shape[0])  
    lam_grid = np.arange(0, lam_max, step=lam_max/size)
    return lam_grid
