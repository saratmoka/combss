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

def obj_fn(X, y, t, beta, delta, lam):
	n = np.shape(X)[0]
	bt = beta*t
	
	S = np.nonzero(t)[0]
	print(f'length of t: {len(t)}')
	print(f'nonzero elements: {len(S)}')

	if len(S) == 0:
		obj = 1/n*(y.T@y) + lam*np.sum(t)
	else:
		obj = 1/n*y.T@y - (2/n)*(bt).T@(X.T@y) + (1/n)*(bt.T@((X.T@X)@bt))+(delta/n)*(beta.T@((1-t*t)*beta)) + lam*np.sum(t)
	return obj


def BCD_COMBSS(X, y, lam):
	 
	(n, p) = X.shape
	
	if (n > p):
		A = X.T@X
	else:
		A = X@X.T
	
	delta = int(1.1*helpers.gen_eta_max(A))
	s = np.zeros(p)
	s_curr = np.ones(p)
	
	N = np.where(s == 1)[0]
	Xs = X[:, N]

	beta = np.zeros(p)
	
	if len(N) != 0:
		beta_trun = (pinv(Xs.T@Xs))@(Xs.T@y)
		beta[N] = beta_trun
	
	while not np.array_equal(s, s_curr):
		s_curr = s.copy()
		
		i = 0
		while i < np.shape(s)[0]:
			s_0 = np.copy(s)
			s_0[i] = 0

			s_1 = np.copy(s)
			s_1[i] = 1

			f_s0 = obj_fn(X, y, s_0, beta, delta, lam)
			f_s1 = obj_fn(X, y, s_1, beta, delta, lam)

			update = False

			if (f_s0 < f_s1 and s[i] != 0):
				update = True
				s[i] = 0
				print(f'obj function: {f_s0}, s[{i}] = 0')
			elif (f_s1 < f_s0 and s[i] != 1):
				update = True
				s[i] = 1
				print(f'obj function: {f_s1}, s[{i}] = 1')

			
			if update:
				N = np.where(s == 1)[0]
				Xs = X[:, N]
				beta = np.zeros(p)
				if len(N) != 0:
					beta_trun = (pinv(Xs.T@Xs))@(Xs.T@y)
					beta[N] = beta_trun
			i += 1

	model = np.where(s != 0)[0]

	return s, model


def BCD_COMBSS_CG(X, y, lam):
	 
	(n, p) = X.shape
	
	## One time operations
	if (n > p):
		A = X.T@X
	else:
		A = X@X.T
	
	delta = int(1.1*helpers.gen_eta_max(A))
	s = np.ones(p)
	s_curr = np.zeros(p)
	j = 0
	
	N = np.where(s == 1)[0]
	Xs = X[:, N]

	ps = Xs.shape[1]
	XsTXs = Xs.T@Xs
	cginv, _ = cg(XsTXs, np.ones(ps))

	beta_trun = (cginv)@(Xs.T@y)
	beta = np.zeros(p)
	beta[N] = beta_trun
	
	while not np.array_equal(s, s_curr):
		s_curr = s.copy()
		
		i = 0
		while i < np.shape(s)[0]:
			s_0 = np.copy(s)
			s_0[i] = 0

			s_1 = np.copy(s)
			s_1[i] = 1

			f_s0 = obj_fn(X, y, s_0, beta, delta, lam)
			f_s1 = obj_fn(X, y, s_1, beta, delta, lam)

			if (f_s0 < f_s1 and s[i] != 0):
				s[i] = 0

				N = np.where(s == 1)[0]
				Xs = X[:, N]

				ps = Xs.shape[1]
				XsTXs = Xs.T@Xs
				cginv, _ = cg(XsTXs, np.ones(ps))

				beta_trun = (cginv)@(Xs.T@y)
				beta = np.zeros(p)
				beta[N] = beta_trun

			elif (f_s1 < f_s0 and s[i] != 1):
				s[i] = 1
				N = np.where(s == 1)[0]
				Xs = X[:, N]

				ps = Xs.shape[1]
				XsTXs = Xs.T@Xs
				cginv, _ = cg(XsTXs, np.ones(ps))

				beta_trun = (cginv)@(Xs.T@y)
				beta = np.zeros(p)
				beta[N] = beta_trun
				
			i += 1
		j += 1

	model = np.where(s != 0)[0]

	return s, model


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

	# max value for lambda
	lam_max = y@y/n

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
		t, model = BCD_COMBSS(X, y, lam)

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

				t, model = BCD_COMBSS(X, y, lam)

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
