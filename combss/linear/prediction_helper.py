#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:46:06 2022

@author: sarathmoka
w/ additions by Hua Yang Hu, January 2024
"""


#%%
"""
Required packages
"""
import numpy as np
from numpy.linalg import inv, pinv, norm
from scipy.sparse.linalg import cg
from tqdm import tqdm
from sklearn import metrics
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import pandas as pd

#%%
"""
None of the functions in this cell are used for the simulations in the paper 
as Benoit has created the data before running the simulations.
"""

"""
Function for generating the true set of model parameters. 
"""
def gen_beta0(p, k0, beta_type):
    if k0 > p:
        print("Error: k0 is greater than p")
        print(sys.path)
        sys.tracebacklimit = 1
        raise ValueError()
        
    beta0 = np.zeros(p)
    if beta_type == 1:
        gap = int(p/k0)
        S0 = np.array([i*gap for i in range(k0)])  #indices of non-zero coefficients (i.e., true model)
        beta0[S0] = 1
        return beta0, S0
    
    elif beta_type == 2:
        S0 = np.arange(k0)
        beta0[S0] = 1
        return beta0, S0

    # elif beta_type == 3:
    #     S0 = np.arange(k0)
    #     beta0[S0] = np.linspace(0.5, 10, k0)
    #     return beta0, S0
        
    # elif beta_type == 4:
    #     S0 = np.arange(k0)
    #     beta0[S0] = np.linspace(-10, 10, k0)
    #     return beta0, S0
        
    # elif beta_type == 5:
    #     S0 = np.arange(k0)
    #     beta0[S0] = 1
    #     beta0[k0:] = np.array([0.5**i for i in range(p-k0)])
    #     return beta0, S0
        
    # elif beta_type == 6:
    #     s0_size = np.random.choice(min(p, k0+1))
    #     S0 = np.sort(np.random.choice(p, size=s0_size, replace=False))
    #     for i in S0:
    #         beta0[i] = np.random.uniform(-10, 10)
    #     return beta0, S0
    
"""
Function for generating covariance matrix
"""
def cov_X(p, corr):

    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = corr**(np.abs(i - j))
            
    return cov 


""" 
Function for generating data
"""
def gen_data(n, p, mean, cov, noise_var, beta0, centralize=False):

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

""" 
Function to generate exponential grid.
"""
def gen_lam_grid_exp(y, size, para):
    
    lam_max = norm(y)**2/(y.shape[0])  
    lam_grid = lam_max*np.array([para**i for i in range(size)])
    # lam_grid = np.append(lam_grid, 0.0)
    lam_grid = np.flip(lam_grid)
    
    return lam_grid

""" 
Function to generate harmonic grid.
"""
def gen_lam_grid_harm(y, size, para):
    
    lam_max = norm(y)**2/(y.shape[0])  
    #print("Lambda max:", lam_max)
    lam_grid = lam_max*np.array([1/(para*i +1 ) for i in range(size+1)])
    lam_grid = np.append(lam_grid, 0.0)
    lam_grid = np.flip(lam_grid)
    
    return lam_grid

""" 
Function to generate sqaure-root harmonic grid.
"""
def gen_lam_grid_sqrt_harm(y, size, para):
    
    lam_max = norm(y)**2/(y.shape[0])  
    #print("Lambda max:", lam_max)
    lam_grid = lam_max*np.array([1/(para*np.sqrt(i) +1 ) for i in range(size+1)])
    lam_grid = np.flip(lam_grid)
    
    return lam_grid

""" 
Function to generate linear grid.
"""
def gen_lam_grid_lin(y, size, para):
    
    #lam_max = norm(y)**2/(y.shape[0])  
    lam_grid = np.arange(0,0.3,step=0.005)

    
    return lam_grid

#%%
"""
 Functions to convert t to w and w to a t. These are important in converting the constraint problem to an unconstraint one.
"""

def t_to_w(t):
    w = np.sqrt(-np.log(1 - t))
    return w

def w_to_t(w):
    t = 1 - np.exp(-w*w)
    return t


#%%
"""
Performance metrics
"""
def performance_metrics(data_X, beta_true, beta_pred):
    s_true = [beta_true != 0][0]
    s_pred = [beta_pred != 0][0]
    c_matrix = metrics.confusion_matrix(s_true, s_pred)
    
    TN = c_matrix[0, 0]
    FN = c_matrix[1, 0]
    FP = c_matrix[0, 1]
    TP = c_matrix[1, 1]
    
    
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    sensitivity = TP/(TP + FN)
    specificity = TN/(TN + FP)
    
    
    if (sum(s_pred) == 0):
        pe = 1
        precision = 'NA'
        MCC = 'NA'
    else:
        Xbeta_true = data_X@beta_true
        pe = np.square(Xbeta_true - data_X@beta_pred).mean()/np.square(Xbeta_true).mean()        
        precision =  TP/(TP + FP)
        # print('Pre + Sens:', precision + sensitivity)
        MCC = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    if TP == 0:
        f1_score = 0.0
    else:
        f1_score = TP/(TP + (FP + FN)/2)
        #f1_score = (2*precision*sensitivity)/(precision + sensitivity)
    
    # accuracy = round(accuracy, 4)
    # sensitivity = round(sensitivity, 4)
    # specificity = round(specificity, 4)
    # pe = round(pe, 4)
    # precision = round(precision, 4)
    # f1_score = round(f1_score, 4)
    # MCC = round(MCC, 4)
    
    #print(pe, MCC, accuracy, sensitivity, specificity, f1_score, precision)
        
    return [pe, MCC, accuracy, sensitivity, specificity, f1_score, precision]

#%%
"""
Function to estimate gradient of f(t) via conjugate gradient
Here, XX = (X.T@X)/n, Xy = (X.T@y)/n, Z = XX - (delta/n) I
"""

def f_grad_cg(t, X, y, XX, Xy, Z, lam, delta, beta,  c,  g1, g2,
                cg_maxiter=None,
                cg_tol=1e-5):
    
    p = t.shape[0]
    n = y.shape[0]
    
    if n >= p:
        ## Construct Lt
        ZT = np.multiply(Z, t)
        Lt = np.multiply(ZT, t[:, np.newaxis])
        diag_Lt = np.diagonal(Lt) + (delta/n)
        np.fill_diagonal(Lt, diag_Lt)
        TXy = np.multiply(t, Xy)
        
        ## Constructing beta estimate
        beta, _ = cg(Lt, TXy, x0=beta, maxiter=cg_maxiter, tol=cg_tol)
        
        ## Constructing a and b
        gamma = t*beta
        a = -Xy
        a += XX@gamma
        b = a - (delta/n)*gamma
        
        ## Constructing c and d
        c, _ = cg(Lt, t*a, x0=c, maxiter=cg_maxiter, tol=cg_tol) 
        
        d = Z@(t*c)
        
        ## Constructing gradient
        grad = 2*(beta*(a - d) - (b*c)) + lam
        
    else:
        ## constructing Lt_tilde
        temp = 1 - t*t
        temp[temp < 1e-8] = 1e-8 
        """
        Above we map all the values of temp smaller than 1e-8 to 1e-8 to avoid numerical instability that can 
        arise in the following line.
        """
        S = n*np.divide(1, temp)/delta
        
        
        Xt = np.multiply(X, t)/np.sqrt(n)
        XtS = np.multiply(Xt, S)
        Lt_tilde = Xt@(XtS.T)
        np.fill_diagonal(Lt_tilde, np.diagonal(Lt_tilde) + 1)
        
       
        ## estimate beta
        tXy = t*Xy
        XtStXy = XtS@tXy         
        g1, _ = cg(Lt_tilde, XtStXy, x0=g1, maxiter=cg_maxiter, tol=cg_tol)
        beta = S*(tXy - Xt.T@g1)


        ## Constructing a and b
        gamma = t*beta
        a = -Xy
        a += XX@gamma
        b = a - (delta/n)*gamma
        
        ## Constructing c and d
        ta = t*a
        g2, _ = cg(Lt_tilde, XtS@ta, x0=g2, maxiter=cg_maxiter, tol=cg_tol) 
        c = S*(ta - Xt.T@g2)
        d = Z@(t*c)
        
        ## Constructing gradient
        grad = 2*(beta*(a - d) - (b*c)) + lam

    return grad, beta, c, g1, g2


#%%
"""
Implementation of the ADAM optimizer for combss implementation. See the description in the paper.
"""
def ADAM_combss(X, y,  lam, t_init,
        delta_frac = 1,
        CG = True,

        ## Adam parameters
        xi1 = 0.9, 
        xi2 = 0.999,            
        alpha = 0.1, 
        epsilon = 10e-8,

        
        ## Parameters for Termination
        gd_maxiter = 1e5,
        gd_tol = 1e-5,
        max_norm = True, # default we use max norm as the termination condition.
        epoch=10,
        
        ## Truncation parameters
        tau = 0.5,
        eta = 0.0, 
        
        ## Parameters for Conjugate Gradient method
        cg_maxiter = None,
        cg_tol = 1e-5):
    
    (n, p) = X.shape
    
    ## One time operations
    delta = delta_frac*n
    Xy = (X.T@y)/n
    XX = (X.T@X)/n
    Z = XX.copy()
    np.fill_diagonal(Z, np.diagonal(Z) - (delta/n))
    
    ## Initialization
    t = t_init.copy()
        
    w = t_to_w(t)
    
    t_trun = t.copy()
    t_prev = t.copy()
    active = p
    
    u = np.zeros(p)
    v = np.zeros(p)
    
    beta_trun = np.zeros(p)  

    c = np.zeros(p)
    g1 = np.zeros(n)
    g2 = np.zeros(n)
    
    
    count_to_term = 0
    
    # t_seq = []
    # beta_seq = []
    
    for l in range(gd_maxiter):
        M = np.nonzero(t)[0] ## Indices of t correponds to elements greater than eta. 
        M_trun = np.nonzero(t_trun)[0] 
        active_new = M_trun.shape[0]
        
        if active_new != active:
            ## Find the effective set by removing the columns and rows corresponds to zero t's
            XX = XX[M_trun][:, M_trun]
            Z = Z[M_trun][:, M_trun]
            X = X[:, M_trun]
            Xy = Xy[M_trun]
            active = active_new
            t_trun = t_trun[M_trun]
        
        ## Compute gradient for the effective terms
        grad_trun, beta_trun, c, g1, g2 = f_grad_cg(t_trun, X, y, XX, Xy, Z, lam, delta, beta_trun[M_trun],  c[M_trun], g1, g2)
        w_trun = w[M]
        grad_trun = 2*grad_trun*(w_trun*np.exp(- w_trun*w_trun))
        
        ## ADAM Updates
        u = xi1*u[M_trun] - (1 - xi1)*grad_trun
        v = xi2*v[M_trun] + (1 - xi2)*(grad_trun*grad_trun) 
    
        u_hat = u/(1 - xi1**(l+1))
        v_hat = v/(1 - xi2**(l+1))
        
        w_trun = w_trun + alpha*np.divide(u_hat, epsilon + np.sqrt(v_hat)) 
        w[M] = w_trun
        t[M] = w_to_t(w_trun)
        
        w[t <= eta] = 0.0
        t[t <= eta] = 0.0

     
        
        beta = np.zeros(p)
        beta[M] = beta_trun

        t_trun = t[M] 
        
        # t_seq.append(t.copy())
        # beta_seq.append(beta)

        if max_norm:
            norm_t = max(np.abs(t - t_prev))
            if l > 10000:
                print('l', l)
                
                if l%100 == 0:
                    print('\t norm diff', norm_t)
            if norm_t <= gd_tol:
                count_to_term += 1
                if count_to_term >= epoch:
                    break
            else:
                count_to_term = 0
                
        else:
            norm_t = norm(t)
            if norm_t == 0:
                break
            
            elif norm(t_prev - t)/norm_t <= gd_tol:
                count_to_term += 1
                if count_to_term >= epoch:
                    break
            else:
                count_to_term = 0
        t_prev = t.copy()
    
    model = np.where(t > tau)[0]

    if l+1 < gd_maxiter:
        converge = True
    else:
        converge = False
    return  t, model, converge, l+1

#%% 
"""
Implementation of the Basic Gradient Descent (BGD) for best model selection.  
We do not use this for the simulations in the paper.
"""
def BGD_combss(X, y, lam, t_init,
        delta_frac = 1,
        
        ## BGD parameters           
        alpha = 0.1, 
        epsilon = 10e-8,
        
        ## Parameters for Termination
        gd_tol = 1e-5,
        gd_maxiter = 1e5,
        max_norm = True,
        epoch = 10,
        
        ## Truncation parameters
        tau = 0.5,
        eta = 0.0, 
        
        ## Parameters for Conjugate Gradient method
        cg_maxiter = None,
        cg_tol = 1e-5):


    (n, p) = X.shape
    
    ## One time operations
    delta = delta_frac*n
    Xy = (X.T@y)/n
    XX = (X.T@X)/n
    Z = XX.copy()
    np.fill_diagonal(Z, np.diagonal(Z) - (delta/n))
    
    ## Initialization
    t = t_init.copy()
        
    w = t_to_w(t)
    
    t_trun = t.copy()
    t_prev = t.copy()
    active = p
    
    beta_trun = np.zeros(p)  

    c = np.zeros(p)
    g1 = np.zeros(n)
    g2 = np.zeros(n)
    
    count_to_term = 0
    
    # t_seq = []
    # beta_seq = []
    
    for l in range(gd_maxiter):
        
        M = np.nonzero(t)[0] ## Indices of t correponds to elements greater than eta. 
        M_trun = np.nonzero(t_trun)[0] 
        active_new = M_trun.shape[0]
        
        if active_new != active:
            ## Find the effective set by removing the columns and rows corresponds to zero t's
            XX = XX[M_trun][:, M_trun]
            Z = Z[M_trun][:, M_trun]
            X = X[:, M_trun]
            Xy = Xy[M_trun]
            active = active_new
            t_trun = t_trun[M_trun]
        
        ## Compute gradient for the effective terms
        grad_trun, beta_trun, c, g1, g2 = f_grad_cg(t_trun, X, y, XX, Xy, Z, lam, delta, beta_trun[M_trun],  c[M_trun], g1, g2)
        w_trun = w[M]
        grad_trun = 2*grad_trun*(w_trun*np.exp(- w_trun*w_trun))
        
        ## BGD Updates
        w_trun = w_trun - alpha*grad_trun 
        w[M] = w_trun
        t[M] = w_to_t(w_trun)
        
        w[t <= eta] = 0.0
        t[t <= eta] = 0.0
        
        beta = np.zeros(p)
        beta[M] = beta_trun

        t_trun = t[M] 
        
        # t_seq.append(t.copy())
        # beta_seq.append(beta)
        
        if max_norm:
            norm_temp = max(np.abs(t - t_prev))
            if norm_temp <= gd_tol:
                count_to_term += 1
                if count_to_term >= epoch:
                    break
            else:
                count_to_term = 0
                
        else:
            norm_t = norm(t)
            if norm_t == 0:
                break
            
            elif norm(t_prev - t)/norm_t <= gd_tol:
                count_to_term += 1
                if count_to_term >= epoch:
                    break
            else:
                count_to_term = 0
        t_prev = t.copy()
    
    
    model = np.where(t > tau)[0]

    if l+1 < gd_maxiter:
        converge = True
    else:
        converge = False
    return  t, model, converge, l+1

#%% 
"""
Dynamic grid of lambda is generated as follows: We are given maximum model size $q$ of interest. 

First pass: We start with $\lambda = \lambda_{\max} = \mathbf{y}^\top \mathbf{y}/n$, 
            where an empty model is selected, and use $\lambda \leftarrow \lambda/2$ 
            until we find model of size larger than $q$. 

Second pass: Then, suppose $\lambda_{grid}$ is (sorted) vector of $\lambda$ valued exploited in 
             the first pass, we move from the smallest value to the large value on this grid, 
             and run COMBSS at $\lambda = (\lambda_{grid}[k] + \lambda_{grid}[k+1])/2$ if $\lambda_{grid}[k]$ 
             and $\lambda_{grid}[k+1]$ produced models with different sizes. 
             We repeat this until the size of $\lambda_{grid}$ is larger than a fixed number $nlam$.
"""
def combss_dynamic(X, y, 
                   q = None,
                   nlam = None,
                   t_init= [],         # Initial t vector
                   tau=0.5,               # tau parameter
                   delta_frac=1, # delta_frac = n/delta
                   fstage_frac = 0.5,    #fraction lambda values explored in first stage of dynamic grid
                   eta=0.0,               # Truncation parameter
                   epoch=10,           # Epoch for termination 
                   gd_maxiter=1000, # Maximum number of iterations allowed by GD
                   gd_tol=1e-5,         # Tolerance of GD
                   cg_maxiter=None, # Maximum number of iterations allowed by CG
                   cg_tol=1e-5):        # Tolerance of CG
    
    (n, p) = X.shape
    
    # If q is not given, take q = n.
    if q == None:
        q = min(n, p)
    
    # If number of lambda is not given, take it to be n.
    if nlam == None:
        nlam == n
    t_init = np.array(t_init) 
    if t_init.shape[0] == 0:
        t_init = np.ones(p)*0.5
    
    if cg_maxiter == None:
        cg_maxiter = n
    
    lam_max = y@y/n # max value for lambda

    # Lists to store the findings
    model_list = []
    #model_seq_list = []
    
    # t_list = []
    # t_seq_list = []
    
    # beta_list = []
    # beta_seq_list = []
    
    lam_list = []
    lam_vs_size = []
    
    # converge_list = []

    lam = lam_max
    count_lam = 0

    ## First pass on the dynamic lambda grid
    stop = False
    #print('First pass of lambda grid is running with fraction %s' %fstage_frac)
    while not stop:
        t_final, model, converge, _ = ADAM_combss(X, y, lam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, epoch=epoch, gd_maxiter=gd_maxiter,gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)

        len_model = model.shape[0]

        lam_list.append(lam)
        # t_list.append(t_final)
        # beta_list.append(beta)
        # t_seq_list.append(t_seq)
        # beta_seq_list.append(beta_seq)
        model_list.append(model)
        # converge_list.append(converge)
        lam_vs_size.append(np.array((lam, len_model)))
        count_lam += 1
        print(len_model)
        if len_model >= q or count_lam > nlam*fstage_frac:
            stop = True
        lam = lam/2
        #print('lam = ', lam, 'len of model = ', len_model)
        


    ## Second pass on the dynamic lambda grid
    stop = False
    #print('Second pass of lambda grid is running')
    while not stop:
        temp = np.array(lam_vs_size)
        order = np.argsort(temp[:, 1])
        lam_vs_size_ordered = np.flip(temp[order], axis=0)        

        ## Find the next index
        for i in range(order.shape[0]-1):

            if count_lam <= nlam and lam_vs_size_ordered[i+1][1] <= q and  (lam_vs_size_ordered[i+1][1] != lam_vs_size_ordered[i][1]):

                lam = (lam_vs_size_ordered[i][0] + lam_vs_size_ordered[i+1][0])/2

                
                t_final, model, converge, _ = ADAM_combss(X, y, lam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, epoch=epoch, gd_maxiter=gd_maxiter,gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)
                
                len_model = model.shape[0]

                lam_list.append(lam)
                # t_list.append(t_final)
                # beta_list.append(beta)
                # t_seq_list.append(t_seq)
                # beta_seq_list.append(beta_seq)
                model_list.append(model)
                # converge_list.append(converge)
                lam_vs_size.append(np.array((lam, len_model)))    
                count_lam += 1

        stop = True
    
    return  (model_list, lam_list)





#%%
""" COMBSS with SubsetMapV1

This is the first version of COMBSS available in the paper. 
In particular, we only look at the final t obtained by 
the gradient descent algorithm (ADAM Optimizer) and consider the model corresponds 
to significant elements of t.
"""

def combss(X_train, y_train, X_test, y_test, 
            q = None,           # maximum model size
            nlam = 50,        # number of values in the lambda grid
            t_init= [],         # Initial t vector
            tau=0.5,               # tau parameter
            delta_frac=1, # delta_frac = n/delta
            eta=0.001,               # Truncation parameter
            epoch=10,           # Epoch for termination 
            gd_maxiter=1000, # Maximum number of iterations allowed by GD
            gd_tol=1e-5,         # Tolerance of GD
            cg_maxiter=None, # Maximum number of iterations allowed by CG
            cg_tol=1e-5):
    
    # Call COMBSS_dynamic with ADAM optimizer
    (n, p) = X_train.shape
    t_init = np.array(t_init) 
    if t_init.shape[0] == 0:
        t_init = np.ones(p)*0.5
        
    # If q is not given, take q = n
    if q == None:
        q = min(n, p)
    
    #print('Dynamic combss is called')
    tic = time.process_time()
    (model_list, lam_list) = combss_dynamic(X_train, y_train, q = q, nlam = nlam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, epoch=epoch, gd_maxiter= gd_maxiter, gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)
    toc = time.process_time()
    #print('Dynamic combss is completed')

    # t_arr = np.array(t_list)
    
    """
    Computing the MSE on the test data
    """
    nlam = len(lam_list)
    mse_list = [] # to strore prediction error for each lam
    beta_list = []
    
    for i in range(nlam):
        model_final = model_list[i]
        # len_s = s_final.shape[0]

        # if 0 < len_s < n:
        X_hat = X_train[:, model_final]
        X_hatT = X_hat.T

        X_hatTy = X_hatT@y_train
        XX_hat = X_hatT@X_hat
        

        beta_hat = pinv(XX_hat)@X_hatTy 
        X_hat = X_test[:, model_final]
        mse = np.square(y_test - X_hat@beta_hat).mean()
        mse_list.append(mse)
        beta_pred = np.zeros(p)
        beta_pred[model_final] = beta_hat
        beta_list.append(beta_pred)
        # elif len_s >= n: 
        #     mse = 2*np.square(y_test).mean()
        #     beta_list.append(beta_hat)
        # else:
        #     mse = np.square(y_test).mean()



    
    #print(pred_err)
    ## Convert to numpy array
    # mse_arr = np.array(mse_list)  
    ind_opt = np.argmin(mse_list)
    lam_opt = lam_list[ind_opt]
    model_opt = model_list[ind_opt]
    mse_opt = mse_list[ind_opt] 
    beta_opt = beta_list[ind_opt]
    
    return model_opt, mse_opt, beta_opt, lam_opt, toc - tic

'''
#%%
"""
Function for running COMBSS on several datasets and 
save the corresponding results in a file
"""
def bulk_sim_given_data(n, p, q, beta_type, K0, snr, corr,
                     t_init = [],
                     delta_frac = 1,
                     n_datasets=1, 
                     nlam=50,
                     eta=0.01,
                     seed=1):
    metrics = ['MSE', 'PE', 'MCC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1_score', 'Precision', 'Time', 'Opt Lambda']
    names = metrics + [ str(i) for i in range(p)] 
    
    nmetrics = len(metrics)

    bulk_results = np.zeros((n_datasets, len(names)))
    
    for j in tqdm(range(n_datasets)):
        #print('------------------------------------')
        #print('working on dataset:', j+1)
        #path= '/Users/uqsbabum/Library/CloudStorage/Dropbox/MQ Research/With Sam, Houying and Benoit/Model Selection/Simulations/StatsComp/COMBSS-versions/DATA/CASE%s/n-%s-p%sSNR-%sReplica%s.csv' %(beta_type, n,p,snr,j+1)
        path_train = '../RESULT-PRED/DATA/CASE%s/n-%s-p%sSNR-%sReplica%s.csv' %(beta_type, n,p,snr,j+1)
        df = pd.read_csv(path_train, sep='\t', header=None)
        data = df.to_numpy()
        y_train = data[:, 0]
        X_train = data[:, 1:]
        
        path_test= '../RESULT-PRED/DATA/CASE%s/n-%s-p%sSNR-%sTest-Replica%s.csv' %(beta_type, n,p,snr,j+1)
        df = pd.read_csv(path_test, sep='\t', header=None)
        data = df.to_numpy()
        y_test = data[:, 0]
        X_test = data[:, 1:]

        n_tinit = len(t_init)
        running_time = 0
        
        if n_tinit == 0:
            t_init = [np.ones(p)*0.5]
            n_tinit = 1
        

        result1 = combss(X_train, y_train, X_test, y_test, t_init=t_init[0], 
                        delta_frac=delta_frac,  q = q, nlam = nlam, eta=eta)
        """
        Note that,
            result1 = [model_opt, mse_opt, beta_opt, lam_opt, time]
        """
        running_time += result1[4]
        
        for i in range(n_tinit-1):
            
            result_temp = combss(X_train, y_train, X_test, y_test, t_init=t_init[i+1], 
                                 delta_frac=delta_frac, q = q, nlam = nlam, eta=eta)
            
            running_time += result_temp[4]

            if result1[1] > result_temp[1]:
                result1 = result_temp

        bulk_results[j, nmetrics + result1[0]] = 1
        bulk_results[j, 0] = result1[1]
        bulk_results[j, nmetrics-2] = running_time
        bulk_results[j, nmetrics-1] = result1[3]
        
        beta_pred = result1[2]
        
        beta_true = np.zeros(p)
        if beta_type == 2:
            beta_true[0:10] = 1
        elif beta_type == 3:
            beta_true[0:10] = np.array([0.5**k for k in range(10)])
        else: 
            print("Error: wrong case!")
            
        result2 = performance_metrics(X_train, beta_true, beta_pred)
        bulk_results[j, 1: nmetrics-2] = np.array(result2)
        
    df = pd.DataFrame(bulk_results, columns = names)
    
    df.to_csv("../RESULT-PRED/COMBSS-nLAM/COMBSS-case-%s-n-%d-p-%s-q-%s-corr-%s-ninit-%s-snr-%s-ndatasets-%s-nlam-%s-eta-%s.csv" %(beta_type, n, p, q, corr, n_tinit, snr,  n_datasets, nlam, eta))

    return df
'''




