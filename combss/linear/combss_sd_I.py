"""
Packages required
"""
import numpy as np
import pandas as pd
from numpy.linalg import pinv, norm, inv
import time
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge

#%%
'''
Helper functions for COMBSS
'''


def t_to_w(t):

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

	w = np.log(t/(1-t))
	return w


def w_to_t(w):

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
	t = 1/(1+np.exp(-w))
	return t

def h(u, delta_frac):
	return (2*delta_frac)/(u**3)


def f_grad_cg(t, X, y, Xy, lam, delta_frac, gamma,  upsilon, g1, g2, 
			  cg_maxiter=None, 
			  cg_tol=1e-5):
	""" 
	Calculates the gradient of the objective function with respect to parameters t, as well as the 
	corresponding estimate of beta. Also returns components of the objective function used for recurrent 
	calls of this function.

	Parameters
	----------
	t : array-like of floats.
		The t vector used for calculations.

	X : array-like of shape (n_samples, n_covariates)
		The design matrix, where `n_samples` is the number of samples observed
		and `n_covariates` is the number of covariates measured in each sample.

	y : array-like of shape (n_samples)
		The response data, where `n_samples` is the number of response elements.

	Xy : array-like of shape (n_covariates, 1).
		The matrix Xy is defined as (X.T@y)/n, as featured in the original COMBSS paper.

	lam : float
		The penalty parameter used within the objective function. Referred to as
		'lambda' in the original COMBSS paper.

	delta_frac : float
		The value of delta/n, where delta is a tuning parameter as referenced in the original COMBSS paper. 

	gamma : array-like of floats of shape (n_covariates, 1)
		The current values of beta times t, calculated from the X matrix, y vector and current 
		values of vector t.

	upsilon : array-like of floats of shape (n_covariates, 1).
		The associated value of upsilon with respect to the existing ts.
	
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

	upsilon : array-like of floats of shape (n_covariates, 1).
		The associated value of upsilon with respect to the existing ts.

	g1 : array-like of floats of shape (n_samples, 1)
		The vector g1 is used in constructing the estimate of beta when presented with high 
		dimensional data. In particular, it is a byproduct of the implementation of the 
		Woodbury matrix in the original COMBSS paper, section 6.1.

	g2 : array-like of floats of shape (n_samples, 1)
		The vector g1 is used in constructing the estimate of the gradient of the objective 
		function with respect to t when presented with high dimensional data. In particular, 
		it is a byproduct of the implementation of the Woodbury matrix in the original 
		COMBSS paper, section 6.1.
	"""
	
	p = t.shape[0]
	n = y.shape[0]
	
	if cg_maxiter == None:
		cg_maxiter = min(n, p)
   
	if n >= p:
		t_sqr = t*t
		t_sqr[t_sqr < 1e-8] = 1e-8  #For numerical stability, small values are mapped to 1e-8. This is not required if truncation is used.
		dia = delta_frac*(1 - t_sqr)/t_sqr    

		## Construct Mt
		def matvec(v):
			Mv = X.T@(X@v)/n + dia*v 
			return Mv
		
		M = LinearOperator((p, p), matvec=matvec)
		
		
		## Obtaining gamma estimate
		gamma, _ = cg(M, Xy, x0=gamma, maxiter=cg_maxiter, tol=cg_tol)
		
		b = X.T@(X@gamma)/n 
		upsilon, _ = cg(M, b, x0=upsilon, maxiter=cg_maxiter, tol=cg_tol) 
		
		## Constructing gradient
		grad = 2*h(t, delta_frac)*gamma*(upsilon - gamma) + lam
		
	else:
		## constructing Lt_tilde
		t_sqr = t*t
		temp = 1 - t_sqr
		temp[temp < 1e-8] = 1e-8   #For numerical stability
		inv_dia = (1/delta_frac)*t_sqr/temp    
		
		D_invXy = inv_dia*Xy
		XD_invXy = X@D_invXy
		
		def matvec(v):
			Mv = v + X@(inv_dia*(X.T@v))/n
			return Mv   
		
		M = LinearOperator((n, n), matvec=matvec)
		
		g1, _ = cg(M, XD_invXy, x0=g1, maxiter=cg_maxiter, tol=cg_tol)
		
	   
		## estimate gamma
		gamma = D_invXy - inv_dia*(X.T@g1)/n
		
		b = X.T@(X@gamma)/n 
		D_invb = inv_dia*b
		XD_invb = X@D_invb
		g2, _ = cg(M, XD_invb, x0=g2, maxiter=cg_maxiter, tol=cg_tol)
		upsilon = D_invb - inv_dia*(X.T@g2)/n
		'''
		Check this code again
		'''
		
		
		## Constructing gradient
		grad = 2*h(t, delta_frac)*gamma*(upsilon - gamma) + lam

	return grad, gamma, upsilon, g1, g2


#%%

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
		max_norm = True,     # By default, we use max norm as the termination condition.
		epoch=10,
		
		## Truncation parameters
		tau = 0.5,
		eta = 0.0, 
		
		## Parameters for Conjugate Gradient method
		cg_maxiter = None,
		cg_tol = 1e-5):
	
	""" The Adam optimiser for COMBSS.

	Parameters
	----------
	X : array-like of shape (n_samples, n_covariates)
		The design matrix, where `n_samples` is the number of samples observed
		and `n_covariates` is the number of covariates measured in each sample.

	y : array-like of shape (n_samples)
		The response data, where `n_samples` is the number of response elements.

	lam : float
		The penalty parameter used within the objective function. Referred to as
		'lambda' in the original COMBSS paper.

	t_init : array-like of floats of shape (n_covariates, 1)
		The initial values of t passed into Adam.
		Default value = [].

	delta_frac : float
		 The value of n/delta as found in the objective function for COMBSS.
		Default value = 1.
	
	xi1 (Adam parameter) : float
		The exponential decay rate for the first moment estimates in Adam. 
		Default value = 0.9.

	xi2 (Adam parameter) : float
		The exponential decay rate for the second-moment estimates.
		Default value = 0.99.

	alpha (Adam parameter) : float
		The learning rate for Adam.
		Default value = 0.1.

	epsilon (Adam parameter) : float
		A small number used to avoid numerical instability when dividing by 
		very small numbers within Adam.
		Default value = 1e-8.

	gd_maxiter (Gradient descent parameter) : int
		The maximum number of iterations for gradient descent before the algorithm terminates.
		Default value = 1e5.

	gd_tol (Gradient descent parameter) : float
		The acceptable tolerance used for the termination condition in gradient descent.
		Default value = 1e-5.

	max_norm : Boolean
		Boolean value that signifies if max norm is used for the termination condition in gradient descent.
		If max_norm is set to be True, the termination condition is evaluated using max norm. Otherwise, 
		the L2 norm will be used instead.
		Default value = True

	epoch : int
		The integer that specifies how many consecutive times the termination condiiton has to be satisfied
		before the function terminates.
		Default value = 10.

	tau : float
		The cutoff value for t that signifies its selection in the model. 
		If t[i] > tau, the ith covariate is selected in the model. 
		If t[i] < tau, the ith covariate is not selected in the model.
		Default value = 0.5.

	eta : float
		The parameter that dictates the upper limit used for truncating matrices.
		If the value of t[i] is less than eta, t[i] will be approximated to zero,
		and the ith column of X will be ignored in calculations to improve algorithm perfomance.
		Default value = 0.

	cg_maxiter (Conjugate gradient parameter) : int
		The maximum number of iterations for the conjugate gradient algortihm used 
		to approximate the gradient of the function with respect to t and the gradient 
		of the objective function with respect to beta before the conjugate gradient 
		algorithm terminates.
		Default value = 1e5

	cg_tol (Conjugate gradient parameter) : float
		The acceptable tolerance used for the termination condition in the conjugate gradient 
		algortihms used to approximate the gradient of the function with respect to t and the 
		gradient of the objective function with respect to beta.


	Returns
	-------
	t : array-like of shape (n_covariates)
		The array of t values at the conclusion of the Adam optimisation algorithm.

	model : array-like of integers
		The final chosen model, in the form of an array of integers that correspond to the 
		indicies chosen after using the Adam optimiser.

	converge : Boolean 
		Boolean value that signifies if the gradient descent algorithm terminated by convergence 
		(converge = True), or if it exhausted its maximum iterations (converge = False).

	l+1 : int
		The number of gradient descent iterations executed by the algorithm. If the algorithm 
		reaches the maximum number of iterations provided into the function, l = gd_maxiter.
	"""
	
	(n, p) = X.shape
	
	## One time operation
	Xy = (X.T@y)/n

	
	## Initialization
	t = t_init.copy()
		
	w = t_to_w(t)
	
	t_trun = t.copy()
	t_prev = t.copy()
	active = p
	
	u = np.zeros(p)
	v = np.zeros(p)
	
	gamma_trun = np.zeros(p)  

	upsilon = np.zeros(p)
	g1 = np.zeros(n)
	g2 = np.zeros(n)
	
	
	count_to_term = 0
	
	
	for l in range(gd_maxiter):
		M = np.nonzero(t)[0] # Indices of t that correspond to elements greater than eta. 
		M_trun = np.nonzero(t_trun)[0] 
		active_new = M_trun.shape[0]
		
		if active_new != active:
			## Find the effective set by removing the columns and rows corresponds to zero t's
#             XX = XX[M_trun][:, M_trun]
#             Z = Z[M_trun][:, M_trun]
			X = X[:, M_trun]
			Xy = Xy[M_trun]
			active = active_new
			t_trun = t_trun[M_trun]
		
		## Compute gradient for the effective terms
		grad_trun, gamma_trun, upsilon, g1, g2 = f_grad_cg(t_trun, X, y, Xy, lam, delta_frac, gamma_trun[M_trun],  upsilon[M_trun], g1=g1, g2=g2)
		w_trun = w[M]
		grad_trun = grad_trun*(w_to_t(w_trun)*(1 - w_to_t(w_trun)))
		
		## ADAM Updates 
		u = xi1*u[M_trun] + (1 - xi1)*grad_trun
		v = xi2*v[M_trun] + (1 - xi2)*(grad_trun*grad_trun) 
	
		u_hat = u/(1 - xi1**(l+1))
		v_hat = v/(1 - xi2**(l+1))
		
		w_trun = w_trun - alpha*np.divide(u_hat, epsilon + np.sqrt(v_hat)) 
		w[M] = w_trun
		t[M] = w_to_t(w_trun)
		
		w[t <= eta] = -np.inf
		t[t <= eta] = 0.0

		t_trun = t[M] 
		
		if max_norm:
			norm_t = max(np.abs(t - t_prev))
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


def combss_dynamicV0(X, y, 
				   q = None,
				   nlam = None,
				   t_init= [],             # Initial t vector
				   tau=0.5,             # tau parameter
				   delta_frac=1,        # delta_frac = n/delta
				   fstage_frac = 0.5,   # Fraction lambda values explored in first stage of dynamic grid
				   eta=0.0,             # Truncation parameter
				   epoch=10,            # Epoch for termination 
				   gd_maxiter=1000,     # Maximum number of iterations allowed by GD
				   gd_tol=1e-5,         # Tolerance of GD
				   cg_maxiter=None,     # Maximum number of iterations allowed by CG
				   cg_tol=1e-5):        # Tolerance of CG
	
	""" Dynamically performs Adam for COMBSS over a grid of lambdas to retrieve model of the desired size.

	The dynamic grid of lambda is generated as follows: We are given maximum model size $q$ of interest. 
	
	First pass: We start with $\lambda = \lambda_{\max} = \mathbf{y}^\top \mathbf{y}/n$, 
				where an empty model is selected, and use $\lambda \leftarrow \lambda/2$ 
				until we find model of size larger than $q$. 
	
	Second pass: Then, suppose $\lambda_{grid}$ is (sorted) vector of $\lambda$ valued exploited in 
				 the first pass, we move from the smallest value to the large value on this grid, 
				 and run COMBSS at $\lambda = (\lambda_{grid}[k] + \lambda_{grid}[k+1])/2$ if $\lambda_{grid}[k]$ 
				 and $\lambda_{grid}[k+1]$ produced models with different sizes. 
				 We repeat this until the size of $\lambda_{grid}$ is larger than a fixed number $nlam$.

	Parameters
	----------
	X : array-like of shape (n_samples, n_covariates)
		The design matrix, where `n_samples` is the number of samples observed
		and `n_covariates` is the number of covariates measured in each sample.

	y : array-like of shape (n_samples)
		The response data, where `n_samples` is the number of response elements.
	
	q : int
		The maximum model size of interest. If q is not provided, it is taken to be n.
		Default value = None.

	nlam (Adam parameter) : float
		The number of lambdas explored in the dynamic grid.
		Default value = None.

	t_init : array-like of floats of shape (n_covariates, 1)
		The initial values of t passed into Adam.
		Default value = [].

	tau (Adam parameter) : float
		The cutoff value for t that signifies its selection in the model. 
		If t[i] > tau, the ith covariate is selected in the model. 
		If t[i] < tau, the ith covariate is not selected in the model.
		Default value = 0.5.

	delta_frac : float
		 The value of n/delta as found in the objective function for COMBSS.
		Default value = 1.

	fstage_frac : float
		The fraction of lambda values explored in first stage of dynamic grid.
		Default value = 0.5.

	eta : float
		The parameter that dictates the upper limit used for truncating matrices.
		If the value of t[i] is less than eta, t[i] will be approximated to zero,
		and the ith column of X will be ignored in calculations to improve algorithm perfomance.
		Default value = 0.

	epoch : int
		The integer that specifies how many consecutive times the termination conditon on the norm has 
		to be satisfied before the function terminates.
		Default value = 10.

	gd_maxiter (Gradient descent parameter) : int
		The maximum number of iterations for gradient descent before the algorithm terminates.
		Default value = 1000.

	gd_tol (Gradient descent parameter) : float
		The acceptable tolerance used for the termination condition in gradient descent.
		Default value = 1e-5.

	cg_maxiter (Conjugate gradient parameter) : int
		The maximum number of iterations provided to the conjugate gradient algorithm used 
		to approximate the gradient of the objective function with respect to t and the gradient 
		of the objective function with respect to beta. The conjugate gradient 
		algorithm terminates upon reaching 'cg_maxiter' iterations.
		Default value = None

	cg_tol (Conjugate gradient parameter) : float
		The acceptable tolerance used for the termination condition in the conjugate gradient 
		algortihms used to approximate the gradient of the objective function with respect to t and the 
		gradient of the objective function with respect to beta.
		Default value: 1e-5

		
	Returns
	-------
	model_list : array-like of array-like of integers. 
	Describe the indices chosen as the models for each lambda, e.g. [[1], [1, 6], [1, 11, 20], [12]]  

	lam_list : array-like of floats.
	Captures the sequence of lambda values explored in best subset selection.
	"""

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
	
	# Maximal value for lambda
	lam_max = y@y/n 

	# Lists to store the findings
	model_list = []
	
	lam_list = []
	lam_vs_size = []
	
	lam = lam_max
	count_lam = 0

	## First pass on the dynamic lambda grid
	stop = False
	while not stop:
		t_final, model, converge, _ = ADAM_combss(X, y, lam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, epoch=epoch, gd_maxiter=gd_maxiter,gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)

		len_model = model.shape[0]

		lam_list.append(lam)
		model_list.append(model)
		lam_vs_size.append(np.array((lam, len_model)))
		count_lam += 1
		print(len_model)
		if len_model >= q or count_lam > nlam*fstage_frac:
			stop = True
		lam = lam/2
		


	## Second pass on the dynamic lambda grid
	## Second pass on the dynamic lambda grid
	stop = False
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
				model_list.append(model)
				lam_vs_size.append(np.array((lam, len_model)))    
				count_lam += 1

			if count_lam > nlam:
				stop = True
				break

	temp = np.array(lam_vs_size)
	order = np.argsort(temp[:, 1])
	model_list = [model_list[i] for i in order]
	lam_list = [lam_list[i] for i in order]
	
	return  (model_list, lam_list)


def combssV0(X_train, y_train, X_test, y_test, 
			q = None,           # maximum model size
			nlam = 50,          # number of values in the lambda grid
			t_init= [],         # Initial t vector
			tau=0.5,            # tau parameter
			delta_frac=1,       # delta_frac = n/delta
			eta=0.001,          # Truncation parameter
			epoch=10,           # Epoch for termination 
			gd_maxiter=1000,    # Maximum number of iterations allowed by GD
			gd_tol=1e-5,        # Tolerance of GD
			cg_maxiter=None,    # Maximum number of iterations allowed by CG
			cg_tol=1e-5):
	
	""" Dynamically performs Adam for COMBSS with SubsetMapV1 as proposed in the original paper
		over a grid of lambdas to retrieve a model of the desired size.

		This is the first version of COMBSS available in the paper. In particular, we only look 
		at the final t obtained by the gradient descent algorithm (ADAM Optimiser) and consider 
		the model that corresponds to significant elements of t.
		
	Parameters
	----------
	X_train : array-like of shape (n_samples, n_covariates)
		The design matrix used for training, where `n_samples` is the number of samples 
		observed and `n_covariates` is the number of covariates measured in each sample.

	y_train : array-like of shape (n_samples)
		The response data used for training, where `n_samples` is the number of response elements.

	X_test : array-like of shape (n_samples, n_covariates)
		The design matrix used for testing, where `n_samples` is the number of samples 
		observed and `n_covariates` is the number of covariates measured in each sample.

	y_test : array-like of shape (n_samples)
	The response data used for testing, where `n_samples` is the number of response elements.    

	q : int
		The maximum model size of interest. If q is not provided, it is taken to be n.
		Default value = None.

	nlam : int
		The number of lambdas explored in the dynamic grid.
		Default value = None.

	t_init : array-like of integers
		The initial values of t passed into Adam.
		Default value = [].

	tau : float
		The cutoff value for t that signifies its selection in the model. 
		If t[i] > tau, the ith covariate is selected in the model. 
		If t[i] < tau, the ith covariate is not selected in the model.
		Default value = 0.5.

	delta_frac : float
		 The value of n/delta as found in the objective function for COMBSS.
		Default value = 1.

	eta : float
		The parameter that dictates the upper limit used for truncating matrices.
		If the value of t[i] is less than eta, t[i] will be approximated to zero,
		and the ith column of X will be removed to improve algorithm perfomance.
		Default value = 0.

	epoch : int
		The integer that specifies how many consecutive times the termination condiiton has to be satisfied
		before the function terminates.
		Default value = 10.

	gd_maxiter (Gradient descent parameter) : int
		The maximum number of iterations for gradient descent before the algorithm terminates.
		Default value = 1000.

	gd_tol (Gradient descent parameter) : float
		The acceptable tolerance used for the termination condition in gradient descent.
		Default value = 1e-5.

	cg_maxiter (Conjugate gradient parameter) : int
		The maximum number of iterations provided to the conjugate gradient algorithm used 
		to approximate the gradient of the objective function with respect to t and the gradient 
		of the objective function with respect to beta. The conjugate gradient 
		algorithm terminates upon reaching 'cg_maxiter' iterations.
		Default value = None

	cg_tol (Conjugate gradient parameter) : float
		The acceptable tolerance used for the termination condition in the conjugate gradient 
		algortihms used to approximate the gradient of the objective function with respect to t and the 
		gradient of the objective function with respect to beta.
		Default value: 1e-5


	Returns
	-------
	model_opt : array-like of array-like of integers
	The indices of the best subset predictors in the the optimal model chosen by COMBSS, 
	e.g. [[1], [1, 6], [1, 11, 20], [12]].

	mse_opt : float
		The mean squared error of the optimal model chosen by COMBSS.

	beta_opt : array-like of floats  
		Represents estimates of coefficients for linear regression for the optimal model as chosen by COMBSS.

	lam_opt : float
		The optimal value of lambda used in COMBSS to arrive at the optimal model chosen by COMBSS.

	time : float
		The time taken to execute COMBSS to perform best subset selection, given the data.

	"""

	# Data Normalisation
	mean_vector = np.mean(X_train, axis=0)
	sd_vector = np.std(X_train, axis=0, ddof=1)  # ddof=1 for sample standard deviation

	# Normalize each column by subtracting mean and dividing by standard deviation
	X_train = (X_train - mean_vector) / sd_vector

	X_train = X_train*sd_vector + mean_vector
	
	# Call COMBSS_dynamic with the Adam optimiser
	(n, p) = X_train.shape
	t_init = np.array(t_init) 
	if t_init.shape[0] == 0:
		t_init = np.ones(p)*0.5
		
	# If q is not given, take q = n
	if q == None:
		q = min(n, p)
	
	tic = time.process_time()
	(model_list, lam_list) = combss_dynamicV0(X_train, y_train, q = q, nlam = nlam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, epoch=epoch, gd_maxiter= gd_maxiter, gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)
	toc = time.process_time()
	
	"""
	Computing the MSE on the test data
	"""
	nlam = len(lam_list)
	mse_list = [] 
	beta_list = []

	X_train = X_train*sd_vector + mean_vector

	
	for i in range(nlam):
		model_final = model_list[i]

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

	ind_opt = np.argmin(mse_list)
	lam_opt = lam_list[ind_opt]
	model_opt = model_list[ind_opt]
	mse_opt = mse_list[ind_opt] 
	beta_opt = beta_list[ind_opt]
	
	time_taken = toc - tic

	return model_opt, mse_opt, beta_opt, lam_opt, time_taken
