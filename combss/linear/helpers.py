import numpy as np
from scipy.sparse.linalg import cg
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge


'''
Helper functions for COMBSS
'''


""" Transform t to w using a sigmoid mapping. Used for interchanging between functions 
	that use t for model selection, and w for unconstrained optimisation. Consequently, 
	this function converts a box-constrained problem to an unconstrained one.

	Parameters
	----------
	t :  array-like of float of length n_covariates.
		An array of floats of range [0, 1], where values close to 0 for a particular t[i] support 
		the non-selection of the ith covariate in model selection, and values close to 1 support 
		selection of the ith covariate in model selection.

	Returns
	-------
	w : array-like of float of length n_covariates
		An array of floats, that can be derived from the signoid mapping from t to w. 
		A mapping of t to w values using the sigmoid mapping allows for continuous optimisation
		methods to be applied on a now unconstrained variable.
"""
def t_to_w(t):
	w = np.log(t/(1-t))
	return w

""" Transform w to t using a sigmoid mapping. Used for interchanging between functions 
	that use w for unconstrained optimisation, and t for model selection. 

	Parameters
	----------
	w : array-like of float of length n_covariates
		An array of floats, that can be derived from the signoid mapping from t to w. 
		A mapping of t to w values using the sigmoid mapping allows for continuous optimisation
		methods to be applied on a now unconstrained variable.

	Returns
	-------
	t :  array-like of float of length n_covariates.
		An array of floats of range [0, 1], where values close to 0 for a particular t[i] support 
		the non-selection of the ith covariate in model selection, and values close to 1 support 
		selection of the ith covariate in model selection.
	
"""
def w_to_t(w):
	t = 1/(1+np.exp(-w))
	return t


""" Calculates the gradient of the objective function with respect to parameters t, as well as the 
	corresponding estimate of beta.

	In other words, returns the gradients of 

	Parameters
	----------
	t : array-like of floats.
		The t vector used for calculations.

	X : array-like of shape (n_samples, n_covariates)
		The design matrix, where `n_samples` is the number of samples observed
		and `n_covariates` is the number of covariates measured in each sample.

	y : array-like of shape (n_samples)
		The response data, where `n_samples` is the number of response elements.

	XX : array-like of shape (n_covariates, n_covariates).
		The matrix XX is defined as (X.T@X)/n, as featured in the original COMBSS paper.

	Xy : array-like of shape (n_covariates, 1).
		The matrix Xy is defined as (X.T@y)/n, as featured in the original COMBSS paper.
	
	Z : array-like of shape (n_covariates, n_covariates).
		The matrix Z is defined as a copy of matrix XX, and is used to construct Lt
		from the original COMBSS paper.

	lam : float
		The penalty parameter used within the objective function. Referred to as
		'lambda' in the original COMBSS paper.

	delta : float
		The tuning parameter delta as referenced in the original COMBSS paper. 

	beta : array-like of floats of shape (n_covariates, 1)
		The current values of beta, calculated from the X matrix, y vector and current 
		values of vector t.

	c : array-like of floats of shape (n_covariates - n_truncated, 1)
		The vector c as referenced in the original COMBSS paper, where 'n_covariates' is 
		the number of covariates, and 'n_truncated' is the number of columns of X ignored 
		as a result of the truncation process.
	
	g1 : array-like of floats of shape (n_samples, 1)
		The vector g1 is used in constructing the estimate of beta when presented with high 
		dimensional data. In particular, it is a byproduct of the implementation of the 
		Woodbury matrix in the original COMBSS paper, section 6.1.

	g2 : array-like of floats of shape (n_samples, 1)
		The vector g1 is used in constructing the estimate of the gradient of the objective 
		function with respect to t when presented with high dimensional data. In particular, 
		it is a byproduct of the implementation of the Woodbury matrix in the original 
		COMBSS paper, section 6.1.

	cg_maxiter (Conjugate gradient parameter) : int
		The maximum number of iterations for the conjugate gradient algortihm used 
		to approximate the gradient of the function with respect to t and the gradient 
		of the objective function with respect to beta before the conjugate gradient 
		algorithm terminates.
		Default value = None.

	cg_tol (Conjugate gradient parameter) : float
		The acceptable tolerance used for the termination condition in the conjugate gradient 
		algortihms used to approximate the gradient of the function with respect to t and the 
		gradient of the objective function with respect to beta.
		Default value = 1e5.

	
	Returns
	-------
	grad : array-like of floats (n_covariates, 1).
		The derivative of the objective function with respect to t.

	beta : array-like of floats of shape (n_covariates, 1).
		The associated value of beta with respect to the existing ts.

	g1 : array-like of floats of shape (n_samples, 1)
		The vector g1 is used in constructing the estimate of beta when presented with high 
		dimensional data. In particular, it is a byproduct of the implementation of the 
		Woodbury matrix in the original COMBSS paper, section 6.1.

	g2 : array-like of floats of shape (n_samples, 1)
		The vector g1 is used in constructing the estimate of the gradient of the objective 
		function with respect to t when presented with high dimensional data. In particular, 
		it is a byproduct of the implementation of the Woodbury matrix in the original 
		COMBSS paper, section 6.1.

	Notes
	-----
	
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
