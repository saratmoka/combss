import numpy as np
import pandas as pd
from numpy.linalg import pinv, norm
from scipy.sparse.linalg import cg
import time
import helpers


'''
COMBSS Variant 2 Functions
'''

def grad_fn(X, y, t, beta, delta):
	n = np.shape(X)[0]
	bt = beta*t
	return 1/n*y.T@y - (2/n)*(bt).T@(X.T@y) + (1/n)*(bt.T@((X.T@X)@bt))+(delta/n)*(beta.T@((1-t*t)*beta))

def obj_fn(X, y, t, delta, lam):
	n = np.shape(X)[0]
	Xt = X*t
	beta_tilde = (pinv(Xt.T@Xt + delta*(1-t*t)))@Xt.T@y
	return (1/n)*(norm(y-Xt@beta_tilde))**2 + lam*np.sum(t)


def adam_w(X, y, beta, w0, lam, gam1 = 0.9, gam2 = 0.999, alpha = 0.1, epsilon = 10e-8, maxiter = 1e5, tol = 1e-5):

	# Initialising data-related variables
	(n, p) = X.shape 
	delta = n
	
	# Initialising Adam-related variables
	i = 0
	v_w, u_w = np.zeros(p), np.zeros(p)
	
	stop = False
	converge = False
	
	w_new = w0.copy()
	t_new = helpers.w_to_t(w_new)

	while not stop:
		
		# Initialisation parameters
		w_curr = w_new.copy()
		t_curr = helpers.w_to_t(w_curr)

		# Perform updates for w
		gradw = helpers.grad_v1_w(X, t_curr, beta, delta, lam, y, w_curr)
		v_w = gam1*v_w + (1 - gam1)*gradw
		u_w = gam2*u_w + (1 - gam2)*(gradw*gradw)
		v_ws = v_w/(1-gam1**(i+1))
		u_ws = u_w/(1-gam2**(i+1))
		w_new = w_curr - alpha*np.divide(v_ws, (np.sqrt(u_ws) + epsilon))
		t_new = helpers.w_to_t(w_new)

		# Assess stopping conditions
		if (i > maxiter):
			stop = True
		else:
			diff_t = np.linalg.norm((t_new- t_curr), 2)
			if (diff_t < tol):
				gradt_new = helpers.grad_v1_t(X, t_new, beta, delta, y)
				gradt_curr = helpers.grad_v1_t(X, t_curr, beta, delta, y)

				diff_gradt = np.linalg.norm((gradt_new - gradt_curr),2)
				if (diff_gradt < tol):
					stop = True
		
		# Iterate through counter
		i = i + 1
	
	if i + 1 < maxiter:
		converge = True
	
	return w_new, t_new, converge, i+1

def BCD_COMBSS(X, y, lam, lam_max):
	 
	p = X.shape[1]
	
	## One time operations
	delta = lam_max
	s = np.ones(p)
	s_curr = np.zeros(p)
	j = 0
	
	while not np.array_equal(s, s_curr):

		s_curr = s.copy()
		N = np.where(s == 1)[0]
		Xs = X[:, N]

		beta_trun = (pinv(Xs.T@Xs))@(Xs.T@y)

	
		beta = np.zeros(p)
  
		beta[N] = beta_trun

		i = 0
		while i < np.shape(s)[0]:
			s_0 = np.copy(s)
			s_0[i] = 0

			s_1 = np.copy(s)
			s_1[i] = 1

			f_s0 = obj_fn(X, y, s_0, delta, lam)
			f_s1 = obj_fn(X, y, s_1, delta, lam)

			if (f_s0 < f_s1):
				s[i] = 0
			else:
				s[i] = 1

			i += 1
		j += 1
	model = s

	return model

def combss_dynamicV2(X, y, 
				   q = None,
				   nlam = None,
				   fstage_frac = 0.5): # fraction lambda values explored in first stage of dynamic grid     
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

	lam_max = helpers.gen_lam_max(X.T@X) # max value for lambda

	# Lists to store the findings
	model_list = []
	
	lam_list = []
	lam_vs_size = []
	

	lam = lam_max
	count_lam = 0

	## First pass on the dynamic lambda grid
	stop = False
	#print('First pass of lambda grid is running with fraction %s' %fstage_frac)
	while not stop:
		model = BCD_COMBSS(X, y, lam, lam_max)

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

				model = BCD_COMBSS(X, y, lam, lam_max)

				len_model = model.shape[0]

				lam_list.append(lam)
				model_list.append(model)

				lam_vs_size.append(np.array((lam, len_model)))    
				count_lam += 1

		stop = True
	
	return  (model_list, lam_list)

def combssV2(X_train, y_train, X_test, y_test, 
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
			cg_tol=1e-5,     # Tolerance of CG
            adam_maxiter=None, # Maximum number of iterations allowed by Adam
            adam_tol = 10e-5): # Tolerance of Adam  
	""" 
	COMBSSV1 with SubsetMapV1
	
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
	(model_list, lam_list) = combss_dynamicV2(X_train, y_train, q = q, nlam = nlam)
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
