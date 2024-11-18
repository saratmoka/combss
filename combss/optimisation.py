"""
combss._optimisation.py

This private module contains the optimisation-specific functions for combss.linear.py, 
ensuring seperable logic for future development.

Functions:
- sigmoid(t): Returns the sigmoid mapping between t and w.
- logit(w): Returns the logit mapping between w and t.
- h(u, delta_frac): A helper function for the refined gradient expression.
- f_grad_cg(): Computes the gradient expression, upsilon and numpy arrays to support the Woodbury matrix identity.
- adam(): Employs gradient descent with the Adam optimisers to minimise the novel objective function.
- dynamic_grid(): Conducts a search over a dynamic grid of lambda values.
- bss(): Manages the dynamic grid search, and conducts model evaluation.

"""
import numpy as np
from numpy.linalg import pinv, norm
import time
from scipy.sparse.linalg import cg, LinearOperator


def sigmoid(t):
	""" 
	Maps the vector t to the vector w using the sigmoid function. Used for interchanging between 
	functions that use t for framework interpretation, and w for unconstrained optimisation. 
	Consequently, this function converts box-constrained problems to an unconstrained problems.

	Parameters
	----------
	t :  array-like of float of length n_covariates.
		A numpy array of floats of range [0, 1], where values close to 0 for a particular t[i] support 
		the non-selection of the ith covariate in subset selection, and values close to 1 support 
		selection of the ith covariate in subset selection. In practice, the length of t may be well 
		less than n_covariates due to truncation.

		
	Returns
	-------
	w : array-like of float of length n_covariates
		A numpy array of floats, that derived from the sigmoid mapping from t to w. The mapping of vectors 
		t to w with the sigmoid mapping allows for continuous optimisation methods to be applied toward 
		unconstrained optimisation problems. In practice, the length of w may be well less than n_covariates 
		due to truncation.
	"""

	w = np.log(t/(1-t))
	return w


def logit(w):

	""" 
	Maps the vector w to the vector t using the logit function. Used for interchanging between 
	functions that use w for unconstrained optimisation and t for framework interpretation. 

	Parameters
	----------
	w : array-like of float of length n_covariates
		A numpy array of floats, that derived from the signoid mapping from t to w. The mapping of vectors 
		t to w with the sigmoid mapping allows for continuous optimisation methods to be applied toward 
		unconstrained optimisation problems. In practice, the length of w may be well less than n_covariates 
		due to truncation.

		
	Returns
	-------
	t :  array-like of float of length n_covariates.
		A numpy array of floats of range [0, 1], where values close to 0 for a particular t[i] support 
		the non-selection of the ith covariate in subset selection, and values close to 1 support 
		selection of the ith covariate in subset selection. In practice, the length of t may be well 
		less than n_covariates due to truncation.
	"""
	t = 1/(1+np.exp(-w))
	return t


def h(u, delta_frac):
	""" 
	An auxillary function utilised in the refined gradient expression for COMBSS.

	Parameters
	----------
	u : array-like of float of length n_covariates
		A numpy array of floats. In practice, this is only called taking u = t. In practice, the length 
		of w may be well less than n_covariates due to truncation.

	delta_frac : float
		The quantity delta/n as observed in the paper Moka et al. (2024).

		
	Returns
	-------
	h(u, delta_frac) :  array-like of float of length n_covariates.
		An array of floats used to compute the refined gradient expression. In practice, the length of 
		h(u, delta_frac) may be well less than n_covariates due to truncation.
	"""

	return (2*delta_frac)/(u**3)


def f_grad_cg(t, X, y, Xy, lam, delta_frac, gamma,  upsilon, g1, g2, 
			  cg_maxiter=None, 
			  cg_tol=1e-5):
	""" 
	Calculates the gradient of the objective function f_lambda(t) with respect to parameters t, as well as the 
	corresponding estimate of beta. Also returns components of the objective function used for recurrent 
	calls of this function.

	Parameters
	----------
	t : array-like of floats
		The a numpy array representing the t vector at which the gradient is calculated.

	X : array-like of shape (n, p)
		The design matrix, where `n` is the number of samples in the dataset
		and `p` is the number of covariates measured in each sample.

	y : array-like of shape (n, )
		The response data, where `n` is the number of samples in the dataset.

	Xy : array-like of shape (p, ).
		The matrix Xy is defined as (X.T@y)/n, as featured in the COMBSS method of Moka et al. (2024).

	lam : float
		The penalty parameter used within the objective function. Referred to as
		'lambda' in the paper Moka et al. (2024).

	delta_frac : float
		The value of delta/n, where delta is a tuning parameter as referenced in Moka et al. (2024). 

	gamma : array-like of floats of shape (p, )
		Element-wise product of the current values of beta and t, calculated from the X matrix, y vector and current 
		values of vector t.

	upsilon : array-like of floats of shape (p, ).
		The initial guess for solving the linear equation Mt upsilon = Xy using the conjugate gradient.
	
	g1 , g2 : each is array-like of floats of shape (n, )
		The vectors are used as initial guesses for solving the two linear equations using conjugate gradient 
		in high-dimensional case (n < p) using the Woodbury indetity.

	cg_maxiter : int
		The maximum number of iterations for the conjugate gradient algortihm.
		Default value = None.

	cg_tol : float
		The acceptable tolerance used for the termination condition in the conjugate gradient 
		algortihms.
		Default value = 1e-5.

	
	Returns
	-------
	grad : array-like of floats (p, ).
		The gradient of the objective function f_lambda(t) with respect to t.

	upsilon : array-like of floats of shape (p, ).
		The solution of the linear equation Mt upsilon = Xy. 
		This is useful as the intial guess for solving the same linear equation in the next iteration in Adam.

	g1, g2 : array-like of floats of shape (n, )
		The solutions of the two linear equations in high-dimensional case (n < p) using the Woodbury indetity.
		This is useful as the intial guess for solving the same linear equations in the next iteration in Adam.
		
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
		gamma, _ = cg(M, Xy, x0=gamma, maxiter=cg_maxiter, rtol=cg_tol)
		
		b = X.T@(X@gamma)/n 
		upsilon, _ = cg(M, b, x0=upsilon, maxiter=cg_maxiter, rtol=cg_tol) 
		
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
		
		g1, _ = cg(M, XD_invXy, x0=g1, maxiter=cg_maxiter, rtol=cg_tol)
	   
		## estimate gamma
		gamma = D_invXy - inv_dia*(X.T@g1)/n
		
		b = X.T@(X@gamma)/n 
		D_invb = inv_dia*b
		XD_invb = X@D_invb
		g2, _ = cg(M, XD_invb, x0=g2, maxiter=cg_maxiter, rtol=cg_tol)
		upsilon = D_invb - inv_dia*(X.T@g2)/n
		
		## Constructing gradient
		grad = 2*h(t, delta_frac)*gamma*(upsilon - gamma) + lam

	return grad, gamma, upsilon, g1, g2


#%%

def adam(X, y,  lam, t_init,
		delta_frac = 1,

		## Adam parameters
		xi1 = 0.9, 
		xi2 = 0.999,            
		alpha = 0.1, 
		epsilon = 10e-8,
	 
		## Parameters for Termination
		gd_maxiter = 100000,
		gd_tol = 1e-5,
		max_norm = True,    # By default, we use max norm as the termination condition.
		patience=10,
		
		## Truncation parameters
		tau = 0.5,
		eta = 0.0, 
		
		## Parameters for Conjugate Gradient method
		cg_maxiter = None,
		cg_tol = 1e-5):
	
	""" The Adam optimiser used within the COMBSS algorithm.

	Parameters
	----------
	X : array-like of shape (n, p)
		The design matrix, where `n` is the number of samples in the dataset
		and `p` is the number of covariates measured in each sample.
	y : array-like of shape (n, )
		The response data, where `n` is the number of samples in the dataset.

	lam : float
		The penalty parameter used within the objective function. Referred to as
		'lambda' in the paper Moka et al. (2024).

	t_init : array-like of floats of shape (p, )
		The initial values of t passed into Adam.
		Default value = [].

	delta_frac : float
		The value of n/delta as found in the objective function in the COMBSS algorithm.
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
		extremely small numbers within Adam.
		Default value = 1e-8.

	gd_maxiter : int
		The maximum number of iterations for Adam before the algorithm terminates.
		Default value = 100000.

	gd_tol : float
		The acceptable tolerance used for the termination condition in Adam.
		Default value = 1e-5.

	max_norm : Boolean
		Boolean value that signifies if the max norm is used for the termination condition in gradient descent.
		If max_norm = True, the termination condition is evaluated using the max norm. Otherwise, 
		the L2 norm will be used instead.
		Default value = True

	patience : int
		The integer that specifies how many consecutive times the termination condiiton has to be satisfied
		before the function terminates.
		Default value = 10.

	tau : float
		The cutoff value for t that signifies its selection of the covariates. 
		If t[i] > tau, the ith covariate is selected. 
		If t[i] < tau, the ith covariate is not selected.
		Default value = 0.5.

	eta : float
		The parameter that dictates the upper limit used for truncating matrices.
		If the value of t[i] is less than eta, t[i] will be approximated to zero,
		and the ith column of X will be ignored in calculations to improve algorithm perfomance.
		Default value = 0.

	cg_maxiter : int
		The maximum number of iterations for the conjugate gradient algortihm.
		Default value = None.

	cg_tol : float
		The acceptable tolerance used for the termination condition in the conjugate gradient 
		algortihms.
		Default value = 1e-5.


	Returns
	-------
	t : array-like of shape (p, )
		The array of t values at the conclusion of the Adam optimisation algorithm.

	subset : array-like of integers
		The final chosen subset, in the form of an array of integers that correspond to the 
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
		
	w = sigmoid(t)
	
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
			X = X[:, M_trun]
			Xy = Xy[M_trun]
			active = active_new
			t_trun = t_trun[M_trun]
		
		## Compute gradient for the effective terms
		grad_trun, gamma_trun, upsilon, g1, g2 = f_grad_cg(t_trun, X, y, Xy, lam, delta_frac, gamma_trun[M_trun],  upsilon[M_trun], g1=g1, g2=g2, cg_maxiter=cg_maxiter, cg_tol=cg_tol)
		w_trun = w[M]
		grad_trun = grad_trun*(logit(w_trun)*(1 - logit(w_trun)))
		
		## ADAM Updates 
		u = xi1*u[M_trun] + (1 - xi1)*grad_trun
		v = xi2*v[M_trun] + (1 - xi2)*(grad_trun*grad_trun) 
	
		u_hat = u/(1 - xi1**(l+1))
		v_hat = v/(1 - xi2**(l+1))
		
		w_trun = w_trun - alpha*np.divide(u_hat, epsilon + np.sqrt(v_hat)) 
		w[M] = w_trun
		t[M] = logit(w_trun)
		
		w[t <= eta] = -np.inf
		t[t <= eta] = 0.0

		t_trun = t[M] 
		
		if max_norm:
			norm_t = max(np.abs(t - t_prev))
			if norm_t <= gd_tol:
				count_to_term += 1
				if count_to_term >= patience:
					break
			else:
				count_to_term = 0
				
		else:
			norm_t = norm(t)
			if norm_t == 0:
				break
			
			elif norm(t_prev - t)/norm_t <= gd_tol:
				count_to_term += 1
				if count_to_term >= patience:
					break
			else:
				count_to_term = 0
		t_prev = t.copy()
	
	subset = np.where(t > tau)[0]

	if l+1 < gd_maxiter:
		converge = True
	else:
		converge = False
	return  t, subset, converge, l+1


def dynamic_grid(X, y, t_init,
				   q = None,
				   nlam = None,
				   tau=0.5,             # tau parameter
				   delta_frac=1,        # delta_frac = n/delta
				   fstage_frac = 0.5,   # Fraction lambda values explored in first stage of dynamic grid
				   eta=0.0,             # Truncation parameter
				   patience=10,         # Patience period for termination 
				   gd_maxiter=1000,     # Maximum number of iterations allowed by GD
				   gd_tol=1e-5,         # Tolerance of GD
				   cg_maxiter=None,     # Maximum number of iterations allowed by CG
				   cg_tol=1e-5):        # Tolerance of CG
	
	""" Executes the COMBSS algorithm over a dynamic grid of lambdas to provide a subset for each lambda on the grid.

	The dynamic grid of lambda is generated as follows: We are given maximum subset size q of interest. 
	
	First pass: We start with $\lambda = \lambda_{\max} = \mathbf{y}^\top \mathbf{y}/n$, 
				where an empty subset is guaranteed to be selected, and decrease iteratively $\lambda \leftarrow \lambda/2$ 
				until we find subset of size larger than $q$. 
	
	Second pass: Then, suppose $\lambda_{grid}$ is (sorted) vector of $\lambda$ valued exploited in 
				 the first pass, we move from the smallest value to the large value on this grid, 
				 and run COMBSS at $\lambda = (\lambda_{grid}[k] + \lambda_{grid}[k+1])/2$ if $\lambda_{grid}[k]$ 
				 and $\lambda_{grid}[k+1]$ produced subsets with different sizes. 
				 We repeat this until the size of $\lambda_{grid}$ is larger than a fixed number $nlam$.

	Parameters
	----------
	X : array-like of shape (n, p)
		The design matrix, where `n` is the number of samples in the dataset
		and `p` is the number of covariates measured in each sample.
	y : array-like of shape (n, )
		The response data, where `n` is the number of samples in the dataset.
		
	q : int
		The maximum subset size of interest. If q is not provided, it is taken to be n.
		Default value = None.

	nlam : float
		The number of lambdas explored in the dynamic grid. 
		Default value = None.

	t_init : array-like of floats of shape (p, )
		The initial values of t passed into Adam.
		Default value = [].

	tau : float
		The cutoff value for t that signifies its selection of covariates. 
		If t[i] > tau, the ith covariate is selected. 
		If t[i] < tau, the ith covariate is not selected.
		Default value = 0.5.

	delta_frac : float
		 The value of n/delta as found in the objective function for COMBSS.
		Default value = 1.

	fstage_frac : float
		The fraction of lambda values explored in first pass of dynamic grid.
		Default value = 0.5.

	eta : float
		The parameter that dictates the upper limit used for truncating matrices.
		If the value of t[i] is less than eta, t[i] will be approximated to zero,
		and the ith column of X will be ignored in calculations to improve algorithm perfomance.
		Default value = 0.

	patience : int
		The integer that specifies how many consecutive times the termination conditon on the norm has 
		to be satisfied before the function terminates.
		Default value = 10.

	gd_maxiter : int
		The maximum number of iterations for Adam before the algorithm terminates.
		Default value = 1000.

	gd_tol : float
		The acceptable tolerance used for the termination condition in Adam.
		Default value = 1e-5.

	cg_maxiter : int
		The maximum number of iterations for the conjugate gradient algortihm.
		Default value = None.

	cg_tol : float
		The acceptable tolerance used for the termination condition in the conjugate gradient 
		algortihms.
		Default value = 1e-5.

		
	Returns
	-------
	subset_list : array-like of array-like of integers. 
		Describe the indices chosen as the subsets for each lambda, e.g. [[1], [1, 6], [1, 11, 20], [12]]  

	lam_list : array-like of floats.
		Captures the sequence of lambda values explored in best subset selection.
	"""
	(n, p) = X.shape
	
	# If q is not given, take q = n.
	if q == None:
		q = min(n, p)
	
	# If number of lambda is not given, take it to be n.
	if nlam == None:
		nlam == 50

	# If t_init is not given, we take t_init to be the p-dimensional vector with 0.5 as every element.
	if t_init.shape[0] == 0:
		t_init = np.ones(p)*0.5
	
	if cg_maxiter == None:
		cg_maxiter = n
	
	# Maximal value for lambda
	lam_max = y@y/n 

	# Lists to store the findings
	subset_list = []
	
	lam_list = []
	lam_vs_size = []
	
	lam = lam_max
	count_lam = 0

	## First pass on the dynamic lambda grid
	stop = False
	non_empty_set = False
	while not stop:

		t_final, subset, converge, _ = adam(X, y, lam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, patience=patience, gd_maxiter=gd_maxiter, gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)

		len_subset = subset.shape[0]
		if len_subset != 0:
			non_empty_set = True
		
		lam_list.append(lam)
		subset_list.append(subset)
		lam_vs_size.append(np.array((lam, len_subset)))
		count_lam += 1
		
		if len_subset >= q: 
			stop = True
		
		if count_lam > nlam*fstage_frac and non_empty_set:
			stop = True
		lam = lam/2
		
		
	stop = False
	if count_lam >= nlam or not non_empty_set:
		stop = True

	## Second pass on the dynamic lambda grid if stop = False
	while not stop:

		temp = np.array(lam_vs_size)
		order = np.argsort(temp[:, 1])
		lam_vs_size_ordered = np.flip(temp[order], axis=0)        

		## Find the next index
		for i in range(order.shape[0]-1):

			if count_lam <= nlam and lam_vs_size_ordered[i+1][1] <= q and  (lam_vs_size_ordered[i+1][1] != lam_vs_size_ordered[i][1]):

				lam = (lam_vs_size_ordered[i][0] + lam_vs_size_ordered[i+1][0])/2

				t_final, subset, converge, _ = adam(X, y, lam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, patience=patience, gd_maxiter=gd_maxiter,gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)

				len_subset = subset.shape[0]

				lam_list.append(lam)
				subset_list.append(subset)
				lam_vs_size.append(np.array((lam, len_subset)))    
				count_lam += 1

			if count_lam > nlam:
				stop = True
				break

	temp = np.array(lam_vs_size)
	order = np.argsort(temp[:, 1])
	subset_list = [subset_list[i] for i in order]
	lam_list = [lam_list[i] for i in order]
	
	return  (subset_list, lam_list)


def bss(X_train, y_train, X_test, y_test, 
			q = None,           # Maximum subset size
			nlam = 50,          # Number of values in the lambda grid
			t_init= [],         # Initial t vector
			scaling = False,    # If True, the training data is scaled 
			tau=0.5,            # Threshold parameter
			delta_frac=1,       # delta_frac = n/delta
			eta=0.001,          # Truncation parameter
			patience=10,        # Patience period for termination 
			gd_maxiter=1000,    # Maximum number of iterations allowed by GD
			gd_tol=1e-5,        # Tolerance of GD
			cg_maxiter=None,    # Maximum number of iterations allowed by CG
			cg_tol=1e-5):
	
	""" Best subset selection from the list of subsets generated by COMBSS with 
		SubsetMapV1 as proposed in the original paper Moka et al. (2024)
		over a grid of dynamically generated lambdas.

		
	Parameters
	----------
	X_train : array-like of shape (n_train, n_covariates)
		The design matrix used for training, where `n_train` is the number of samples 
		in the training data and `n_covariates` is the number of covariates measured in each sample.

	y_train : array-like of shape (n_train)
		The response data used for training, where `n_train` is the number of samples in the training data.

	X_test : array-like of shape (n_test, n_covariates)
		The design matrix used for testing, where `n_test` is the number of samples 
		in the testing data and `n_covariates` is the number of covariates measured in each sample.

	y_test : array-like of shape (n_test)
		The response data used for testing, where `n_samples` is the number of samples in the testing data.    

	q : int
		The maximum subset size of interest. If q is not provided, it is taken to be n.
		Default value = None.

	nlam : int
		The number of lambdas explored in the dynamic grid.
		Default value = 50.

	t_init : array-like of integers
		The initial value of t passed into Adam optimizer.
		Default value = [].

	tau : float
		The cutoff value for t that signifies its selection of covariates. 
		If t[i] > tau, the ith covariate is selected. 
		If t[i] < tau, the ith covariate is not selected.
		Default value = 0.5.

	delta_frac : float
		The value of n_train/delta as found in the objective function for COMBSS.
		Default value = 1.

	eta : float
		The parameter that dictates the upper limit used for truncating t elements during the optimization.
		If the value of t[i] is less than eta, t[i] will be mapped to zero,
		and the ith column of X will be removed to improve algorithm perfomance.
		Default value = 0.

	patience : int
		The integer that specifies how many consecutive times the termination condiiton has to be satisfied
		before the Adam optimzer terminates.
		Default value = 10.

	gd_maxiter : int
		The maximum number of iterations for Adam before the algorithm terminates.
		Default value = 1000.

	gd_tol : float
		The acceptable tolerance used for the termination condition in Adam.
		Default value = 1e-5.

	cg_maxiter : int
		The maximum number of iterations for the conjugate gradient algortihm.
		Default value = n.

	cg_tol : float
		The acceptable tolerance used for the termination condition in the conjugate gradient 
		algortihms.
		Default value = 1e-5.


	Returns 
	-------
	A dictionary consisting of the following:
		
	subset_opt : array-like of integers
		The indices of the optimal subset of predictors chosen from all the subsets selected 
		by COMBSS over the dynamic grid of lambdas, 

	mse_opt : float
		The mean squared error on the test data corresponds to the subset_opt.

	beta_opt : array-like of floats  
		Represents estimates of coefficients for linear regression for the subset_opt.

	lam_opt : float
		The value of lambda corresponds to the subset_opt.

	time : float
		The time taken to execute COMBSS on the dynamic grid.
		
	lambda_list : list 
		The list of lambda values of the dynamic grid.
		
	subset_list : list
		The list subsets obtained by COMBSS on the dynamic grid of lambda values. For each i, 
		subset_list[i] corresponds to lambda_list[i].
	"""
	
	if scaling:
		column_norms = np.linalg.norm(X_train, axis=0)
		X_train = X_train / column_norms

	(n, p) = X_train.shape
	t_init = np.array(t_init) 
	
	if t_init.shape[0] == 0:
		t_init = np.ones(p)*0.5
		
	# If q is not given, take q = n
	if q == None:
		q = min(n, p)
	
	tic = time.process_time()
	(subset_list, lam_list) = dynamic_grid(X_train, y_train, q = q, nlam = nlam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, patience=patience, gd_maxiter= gd_maxiter, gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)
	toc = time.process_time()

	if scaling:
		X_train = X_train * column_norms

	"""
	Computing the MSE on the test data
	"""
	nlam = len(lam_list)
	mse_list = [] 
	beta_list = []
	
	for i in range(nlam):
		subset_final = subset_list[i]

		X_hat = X_train[:, subset_final]
		X_hatT = X_hat.T

		X_hatTy = X_hatT@y_train
		XX_hat = X_hatT@X_hat
	
		beta_hat = pinv(XX_hat)@X_hatTy 
		X_hat = X_test[:, subset_final]
		mse = np.square(y_test - X_hat@beta_hat).mean()
		mse_list.append(mse)
		beta_pred = np.zeros(p)
		beta_pred[subset_final] = beta_hat
		beta_list.append(beta_pred)

	ind_opt = np.argmin(mse_list)
	lam_opt = lam_list[ind_opt]
	subset_opt = subset_list[ind_opt]
	mse_opt = mse_list[ind_opt] 
	beta_opt = beta_list[ind_opt]
	
	time_taken = toc - tic
	
	result = {
		"subset" : subset_opt,
		"mse" : mse_opt,
		"coef" : beta_opt,
		"lambda" : lam_opt,
		"time" : time_taken,
		"subset_list" : subset_list,
		"lambda_list" : lam_list
		}

	return result #subset_opt, mse_opt, beta_opt, lam_opt, time_taken