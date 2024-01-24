import numpy as np
import sys

def gen_beta0(p, k0, case):
    """
    Function for generating the true beta parameters. 

    p: number of features (i.e., columns of the design matrix X)
    K0: number of non-zero elements in the coefficient vector beta
    
    case has three options:
        case 0: k0 non-zero coefficients of true beta are placed at equl distance
        case 1: The first k0 components of beta are equal to 1 and all others are equal to 0
        case 2: the first k0 components of beta decrease as beta_i = 0.5^i for i = 0, 1, ...., k0, and all other elements are equal to 0
    """
  
    if k0 > p:
        print("Error: k0 is greater than p")
        print(sys.path)
        sys.tracebacklimit = 1
        raise ValueError()
        
    beta0 = np.zeros(p)
    if case == 0:
        gap = int(p/k0)
        S0 = np.array([i*gap for i in range(k0)])  #indices of non-zero coefficients (i.e., true model)
        beta0[S0] = 1
        return beta0, S0
    
    elif case == 1:
        S0 = np.arange(k0)
        beta0[S0] = 1
        return beta0, S0

    elif case == 2:
        S0 = np.arange(k0)
        beta0[S0] = np.array([0.5**i for i in S0])
        return beta0, S0
        
    # elif case == 3:
    #     S0 = np.arange(k0)
    #     beta0[S0] = np.linspace(-10, 10, k0)
    #     return beta0, S0
        
    # elif case == 4:
    #     S0 = np.arange(k0)
    #     beta0[S0] = 1
    #     beta0[k0:] = np.array([0.5**i for i in range(p-k0)])
    #     return beta0, S0
        
    # elif case == 5:
    #     s0_size = np.random.choice(min(p, k0+1))
    #     S0 = np.sort(np.random.choice(p, size=s0_size, replace=False))
    #     for i in S0:
    #         beta0[i] = np.random.uniform(-10, 10)
    #     return beta0, S0

def cov_X(p, corr):
        
    """
    Function for generating covariance matrix cov of the data

    p: number of features (i.e., columns of the design matrix X)
    corr: Correlation coefficient such that 
                cov[i, j] = corr^{|i - j|}
                
    """
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = corr**(np.abs(i - j))
            
    return cov 


def gen_data(n, p, mean, cov, noise_var, beta0, centralize=False):
    """ 
    Function for generating data (X, y)

    n: Number of data samples
    p: number of features (i.e., columns of the design matrix X)
    cov: covariance matrix of the data
    noise_var: variance of the additive noise
    beta0: true coefficient vector
    centralize: Data is centralized if True, and not centralized if False
    
    """

    X = np.random.multivariate_normal(mean, cov, n)
    
    if centralize:
        # centralize X
        cmean = np.mean(X, axis=0)
        X = np.subtract(X, cmean)
        if (np.sum(X) == np.zeros(p)).all(): 
            print("Error: centralization didn't work")
            print(sys.path)
            sys.tracebacklimit = 1
            raise ValueError()
            
    y = [np.random.normal(X[i]@beta0, np.sqrt(noise_var)) for i in range(n)]
    y = np.array(y)
    return [X, y]



