import numpy as np
import pandas as pd
from numpy.linalg import pinv, norm
from scipy.sparse.linalg import cg
import time
import helpers


'''
COMBSS Variant 2 Functions, Blockwise Type 2b

In this type of blockwise coordinate descent, we optimise over beta, then t t as the objective function is convex 
w.r.t t when delta is taken to be greater than the maximal eigenvalue of X.T@T provided a fixed value of beta.
'''


def obj_fn(X, y, t, delta, lam):
	(n,p) = X.shape
	Xt = t*X
	bt = (pinv(Xt.T@Xt + delta*(1-t*t)*np.eye(p)))@(Xt.T@y)
	n = np.shape(X)[0]
	return 1/n*((y-Xt@bt).T@(y-Xt@bt)) + lam*np.sum(t)


""" The Adam optimiser for COMBSS.

	Parameters
	----------
	X : array-like of shape (n_samples, n_covariates)
		The design matrix, where `n_samples` is the number of samples observed
		and `n_covariates` is the number of covariates measured in each sample.

	y : array-like of shape (n_samples)
		The response data, where `n_samples` is the number of response elements.
	
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
		and the ith column of X will be removed to improve algorithm perfomance.
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
		The array of t values at the conclusion of Adam.

	model : array-like of integers
		The final chosen model, in the form of an array of integers that correspond to the 
		indicies chosen after performing Adam.

	converge : Boolean 
		Boolean value that signifies if the gradient descent algorithm converged by it's 
		termination conditions (converge = True), or if it exhausted its maximum iterations 
		(converge = False).

	l+1 : int
		The number of iterations of the gradient descent loop executed by the algorithm. 
		If the algorithm reaches the maximum number of iterations provided into the function, 
		l = gd_maxiter.
"""
def iterate_combss(X, y,  lam, t_init, delta_frac = 1):
	"""
	Implementation of the ADAM optimizer for combss. 
	"""    
	(n, p) = X.shape
	
	## One time operations
	delta = delta_frac*n
	# t_init doesnt work when t = t_init
	t = t_init.copy()
	t_curr = np.zeros_like(t)

	
	while not np.array_equal(t, t_curr):
		t_curr = t.copy()
		
		for i in range(np.shape(t)[0]):
			t_0 = np.copy(t)
			t_0[i] = 0

			t_1 = np.copy(t)
			t_1[i] = 1

			f_t0 = obj_fn(X, y, t_0, delta, lam)
			f_t1 = obj_fn(X, y, t_1, delta, lam)

			if (f_t0 < f_t1):
				t[i] = 0
			elif (f_t1 < f_t0):
				t[i] = 1				

	model = np.where(t != 0)[0]

	return t, model

""" Basic Gradient Descent for COMBSS.
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
	"""
	Implementation of the Basic Gradient Descent (BGD) for best model selection.  
	We do not use this for the simulations in the paper.
	"""

	(n, p) = X.shape
	
	## One time operations
	delta = delta_frac*n
	Xy = (X.T@y)/n
	XX = (X.T@X)/n
	Z = XX.copy()
	np.fill_diagonal(Z, np.diagonal(Z) - (delta/n))
	
	## Initialization
	t = t_init.copy()
		
	w = helpers.t_to_w(t)
	
	t_trun = t.copy()
	t_prev = t.copy()
	active = p
	
	beta_trun = np.zeros(p)  

	c = np.zeros(p)
	g1 = np.zeros(n)
	g2 = np.zeros(n)
	
	count_to_term = 0
	
	
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
		grad_trun, beta_trun, c, g1, g2 = helpers.f_grad_cg(t_trun, X, y, XX, Xy, Z, lam, delta, beta_trun[M_trun],  c[M_trun], g1, g2)
		w_trun = w[M]
		grad_trun = 2*grad_trun*(w_trun*np.exp(- w_trun*w_trun))
		
		## BGD Updates
		w_trun = w_trun - alpha*grad_trun 
		w[M] = w_trun
		t[M] = helpers.w_to_t(w_trun)
		
		w[t <= eta] = 0.0
		t[t <= eta] = 0.0
		
		beta = np.zeros(p)
		beta[M] = beta_trun

		t_trun = t[M] 
		
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

""" Dynamically performs Adam for COMBSS over a grid of lambdas to retrieve model of the desired size.

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

	t_init : array-like of integers
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
		and the ith column of X will be removed to improve algorithm perfomance.
		Default value = 0.

	epoch : int
		The integer that specifies how many consecutive times the termination condiiton has to be satisfied
		before the function terminates.
		Default value = 10.

	gd_maxiter (Gradient descent parameter) : int
		The maximum number of iterations for gradient descent before the algorithm terminates.
		Default value = 1e5.

	gd_tol (Gradient descent parameter) : float
		The acceptable tolerance used for the termination condition in gradient descent.
		Default value = 1e-5.

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
	model_list : array-like of integers, size q

	lam_list : array-like 

"""
def combss_dynamicV4(X, y, 
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
	i = 0
	while not stop:
		print(f'i: {i}')
		print(f't_init = {t_init}')
		t_final, model = iterate_combss(X, y, lam, t_init=t_init, delta_frac=delta_frac)
		print("a")
		print(f'lam: {lam}')
		len_model = model.shape[0]

		lam_list.append(lam)
		model_list.append(model)
		lam_vs_size.append(np.array((lam, len_model)))
		count_lam += 1
		print(len_model)
		if len_model >= q or count_lam > nlam*fstage_frac:
			stop = True
		lam = lam/2
		#print('lam = ', lam, 'len of model = ', len_model)
		i += 1


	## Second pass on the dynamic lambda grid
	stop = False
	#print('Second pass of lambda grid is running')
	while not stop:
		print("b")
		temp = np.array(lam_vs_size)
		order = np.argsort(temp[:, 1])
		lam_vs_size_ordered = np.flip(temp[order], axis=0)        

		## Find the next index
		for i in range(order.shape[0]-1):

			if count_lam <= nlam and lam_vs_size_ordered[i+1][1] <= q and  (lam_vs_size_ordered[i+1][1] != lam_vs_size_ordered[i][1]):

				lam = (lam_vs_size_ordered[i][0] + lam_vs_size_ordered[i+1][0])/2

				t_final, model = iterate_combss(X, y, lam, t_init=t_init, delta_frac=delta_frac)

				len_model = model.shape[0]

				lam_list.append(lam)
				# t_list.append(t_final)
				# beta_list.append(beta)
				model_list.append(model)
				lam_vs_size.append(np.array((lam, len_model)))    
				count_lam += 1

		stop = True
	
	return  (model_list, lam_list)

""" Dynamically performs Adam for COMBSS over a grid of lambdas to retrieve model of the desired size.

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

	nlam (Adam parameter) : float
		The number of lambdas explored in the dynamic grid.
		Default value = None.

	t_init : array-like of integers
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
		and the ith column of X will be removed to improve algorithm perfomance.
		Default value = 0.

	epoch : int
		The integer that specifies how many consecutive times the termination condiiton has to be satisfied
		before the function terminates.
		Default value = 10.

	gd_maxiter (Gradient descent parameter) : int
		The maximum number of iterations for gradient descent before the algorithm terminates.
		Default value = 1e5.

	gd_tol (Gradient descent parameter) : float
		The acceptable tolerance used for the termination condition in gradient descent.
		Default value = 1e-5.

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
	model_list : array-like of 

	lam_list : array-like

"""
def combssV4(X_train, y_train, X_test, y_test, 
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
	""" 
	COMBSS with SubsetMapV1
	
	This is the first version of COMBSS available in the paper. 
	In particular, we only look at the final t obtained by 
	the gradient descent algorithm (ADAM Optimizer) and consider the model corresponds 
	to significant elements of t.
	"""
	
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
	(model_list, lam_list) = combss_dynamicV4(X_train, y_train, q = q, nlam = nlam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, epoch=epoch, gd_maxiter= gd_maxiter, gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)
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
