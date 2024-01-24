import numpy as np
from numpy.linalg import norm



def gen_lam_grid_exp(y, size, c):
    """ 
    Function to generate exponential grid of lambda values.

    y: response vector of the data
    size: grid size
    c: parameter such that lambda[i] = c^i
    """
    lam_max = norm(y)**2/(y.shape[0])  
    lam_grid = lam_max*np.array([c**i for i in range(size)])
    # lam_grid = np.append(lam_grid, 0.0)
    lam_grid = np.flip(lam_grid)
    
    return lam_grid

def gen_lam_grid_harm(y, size, c):
    """ 
    Function to generate harmonic grid of lambda values.

    y: response vector of the data
    size: grid size
    c: parameter such that lambda[i] = 1/(c^i +1)
    """
    lam_max = norm(y)**2/(y.shape[0])  
    #print("Lambda max:", lam_max)
    lam_grid = lam_max*np.array([1/(c*i + 1) for i in range(size+1)])
    lam_grid = np.append(lam_grid, 0.0)
    lam_grid = np.flip(lam_grid)
    
    return lam_grid


def gen_lam_grid_sqrt_harm(y, size, c):
    """ 
    Function to generate sqaure-root harmonic grid of lambda values.

    y: response vector of the data
    size: grid size
    c: parameter such that lambda[i] = 1/(c^sqrt(i) +1)
    """
  
    lam_max = norm(y)**2/(y.shape[0])  
    #print("Lambda max:", lam_max)
    lam_grid = lam_max*np.array([1/(c*np.sqrt(i) +1 ) for i in range(size+1)])
    lam_grid = np.flip(lam_grid)
    
    return lam_grid

def gen_lam_grid_lin(y, size):
    """ 
    Function to generate linear grid of lambda values.

    y: response vector of the data
    size: grid size
    """
    
    lam_max = norm(y)**2/(y.shape[0])  
    lam_grid = np.arange(0, lam_max, step=lam_max/size)
    return lam_grid
