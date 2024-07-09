import numpy as np
from numpy.linalg import pinv, norm
from scipy.sparse.linalg import cg
from numpy.linalg import inv
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge


'''
Helper functions for COMBSS
'''

""" Transform t to w using a sigmoid mapping.

	In other words, return an input `X_original` whose transform would be X.

	Parameters
	----------
	t :  

	Returns
	-------
	w : 

	Notes
	-----
	
"""
def t_to_w(t):
	"""
	Function to convert t to w, and it is used in converting box-constraint problem to an unconstraint one.
	"""
	w = np.log(t/(1-t))
	return w


""" Transform w to t using a logit mapping.

	In other words, return an input `X_original` whose transform would be X.

	Parameters
	----------
	w :  

	Returns
	-------
	t : 

	Notes
	-----
	
"""
def w_to_t(w):
	"""
	Function to convert w to t, and it is used in converting solution of the unconstraint problem to the constrained case.
	"""  
	t = 1/(1+np.exp(-w))
	return t


def gen_lam_max(A):
	b_k = np.random.rand(A.shape[1])

	for _ in range(1000):
		b_ki = np.dot(A, b_k)

		b_ki_norm = np.linalg.norm(b_ki)
		b_k = b_ki / b_ki_norm

	eigenvalue = np.dot(b_k, np.dot(A, b_k))

	#lam_max = int(1.1*eigenvalue^2)
	return eigenvalue


""" Calculates the gradients of the objective function with respect to parameters t and beta.

	In other words, returns the gradients of 

	Parameters
	----------
	t :  

	X :  

	y :  

	XX : 

	Xy :

	Z :

	lam :

	delta :

	beta :

	c : 
	
	g1 :

	g2 :

	cg_maxiter :

	cg_tol :

	
	Returns
	-------
	grad : 

	beta : 

	g1 : 

	g2 :

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

def beta_cg(t, X, y, XX, Xy, Z, lam, delta, beta,  c,  g1, g2,
			  cg_maxiter=None,
			  cg_tol=1e-5):
	"""
	Function to estimate gradient of f(t) via conjugate gradient
	Here, XX = (X.T@X)/n, Xy = (X.T@y)/n, Z = XX - (delta/n) I

	QQQQ Describe each argument QQQQ
	
	"""    
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

	return beta, c, g1, g2

def grad_v1_t(X, t, beta, delta, y):
	n = y.shape[0]
	b_mult_t = beta*t
	bracket_term = X.T@(X@b_mult_t) - delta*b_mult_t - X.T@y
	return (2/n)*beta*bracket_term

def grad_v1_w(X, t, beta, delta, lam, y, w):
	grad_t = grad_v1_t(X, t, beta, delta, y) + lam
	sigmoid_w = w_to_t(w)
	second_term = sigmoid_w*(1-sigmoid_w)
	return grad_t*second_term

def grad_v1_beta(X, t, beta, delta, y):
	n = y.shape[0]
	b_mult_t = beta*t
	temp = t*(X.T@(X@b_mult_t) - X.T@y - delta*b_mult_t) + delta*beta 
	return (2/n)*temp