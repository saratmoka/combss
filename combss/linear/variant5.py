import numpy as np
import pandas as pd
from numpy.linalg import pinv, norm
from scipy.sparse.linalg import cg
import time
import helpers
import random
from functools import partial
import math


'''
Best Subset Selection: Heuristic inspired by Splitting Method
'''


def obj_fn(t, X, y, lam):
	(n,p) = X.shape
	S = np.nonzero(t)[0]

	if len(S) == 0:
		obj = 1/n*(y.T@y) + lam*np.sum(t)
	else:
		Xs = X[:,S]
		bt = pinv(Xs.T@Xs)@(Xs.T@y)
		n = np.shape(X)[0]
		obj = 1/n*((y-Xs@bt).T@(y-Xs@bt)) + lam*np.sum(t)

	return obj


def iterate_logic(i, s, f_s0, f_s1, level, splits, split_created):
	
	if (f_s0 < level):
		if (f_s1 < level):
			bin_seed = int(time.time())
			np.random.seed(bin_seed)
			s[i] = np.random.randint(0, 2)
			splits.append(s)
			split_created = True
		else:
			s[i] = 0
			splits.append(s)
			split_created = True
	elif (f_s1 < level):
		s[i] = 1
		splits.append(s)
		split_created = True
	elif (f_s0 < f_s1):
		s[i] = 0
	elif (f_s1 < f_s0):
		s[i] = 1
	
	return s, splits, split_created


""" The Adam optimiser for COMBSS.

	Parameters
	----------
	


	Returns
	-------
	
"""
def iterate_combss(X, y, lam, epsilon):

	
	(n, p) = X.shape
 	
	# Number of Candidates
	if p < 1000:
		C = p*20
	else:	  	
		C = int(p/8) 

	N = int(C/5) # Number of Candidates promoted to next level
	# print(f'C = {C}')
	# print(f'N = {N}')
	bin_prob = 0.5  # Probability of success on each trial
	candidates = []
	# f_lambda = partial(obj_fn, X = X, y = y, lam = lam)


	while len(candidates) < C:
		# Generate a random binomial vector
		random_model = np.random.binomial(1, bin_prob, p)
		candidates.append(random_model)
	
	# print(f'candidates: {candidates}')

	fs_values = np.array([obj_fn(s, X, y, lam) for s in candidates])
	pairs = list(zip(candidates, fs_values))

	sorted_pairs = sorted(pairs, key=lambda x: x[1])
	# print(f'sorted_pairs: {sorted_pairs}')

	sorted_candidates = [pair[0] for pair in sorted_pairs]
	# print(f'sorted_candidates: {sorted_candidates}')

	top_candidates = sorted_candidates[:N].copy()
	# print(f'top_candidates: {top_candidates}')

	# Step 3: Get sorted f(x, a, b) values using sorted indices
	fs_sorted = [pair[1] for pair in sorted_pairs]
	# print(f'fs_sorted: {fs_sorted}')

	# Step 4: Find the largest and 4th largest f(x, a, b) values
	optimal = fs_sorted[0]
	print(f'initial optimal: {optimal}')

	level = fs_sorted[N-1]
	# print(f'initial level: {level}')

	promoted_list = sorted_candidates[:N].copy()

	level_old = np.inf

	while abs(level_old - level)/level > epsilon:
		level_old = level
		candidates = top_candidates
		N = len(top_candidates)

		for j in range(N):
			s = candidates[j]

			if j < N-1:
				n = math.ceil((C-N)/N)
			else:
				n = C - N - (N-1)*math.ceil((C-N)/N)

			splits = []
			while len(splits) < n:
				split_created = False
				while split_created is False:

					time_seed = random.randint(1,2^30)

					np.random.seed(time_seed)
					indices = np.arange(p)  
					np.random.shuffle(indices)
				
					for i in indices:
						s_0 = np.copy(s)
						s_0[i] = 0

						s_1 = np.copy(s)
						s_1[i] = 1

						f_s0 = obj_fn(s_0, X, y, lam)
						f_s1 = obj_fn(s_1, X, y, lam)

						s, splits, split_created, = iterate_logic(i=1, s=s, f_s0 = f_s0, f_s1 = f_s1, 
													level = level, splits = splits, split_created = split_created)
						
						if split_created is True:
							if s in promoted_list:
								split_created = False
								continue
							else:
								promoted_list.append(s)
								break
			
			# promoted_list.extend(splits)
				

		fs_values = np.array([obj_fn(s, X, y, lam) for s in promoted_list])
		# print(f'ft_values after splitting: {fs_values}')
		pairs = list(zip(promoted_list, fs_values))

		sorted_pairs = sorted(pairs, key=lambda x: x[1])
		# print(f'sorted_pairs: {sorted_pairs}')

		sorted_candidates = [s[0] for s in sorted_pairs]
		# print(f'sorted_candidates after splitting: {sorted_candidates}')

		top_candidates = sorted_candidates[:N].copy()
		# print(f'top_candidates: {top_candidates}')

		fs_sorted = [pair[1] for pair in sorted_pairs]
		# print(f'fs_sorted: {fs_sorted}')

		top_fs = fs_sorted[:N]
		# print(f'top_fs: {top_fs}')

		optimal = fs_sorted[0]
		print(f'optimal: {optimal}')

		level = fs_sorted[N-1]
		# print(f'level: {level}')

		promoted_list = sorted_candidates[:N].copy()
		# print(f'len(promoted_list) = {len(promoted_list)}, should be 200')

		# print(f'promoted_list: {promoted_list}')
		# print(f'number promoted: {len(promoted_list)}')

	final_indices = top_candidates[0]
	model = np.where(final_indices != 0)[0]
	# print(f'model: {model}')

	return model


def combss_dynamicV5(X, y, 
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
		model = iterate_combss(X, y, lam, epsilon = 1e-3)
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
		temp = np.array(lam_vs_size)
		order = np.argsort(temp[:, 1])
		lam_vs_size_ordered = np.flip(temp[order], axis=0)        

		## Find the next index
		for i in range(order.shape[0]-1):

			if count_lam <= nlam and lam_vs_size_ordered[i+1][1] <= q and  (lam_vs_size_ordered[i+1][1] != lam_vs_size_ordered[i][1]):

				lam = (lam_vs_size_ordered[i][0] + lam_vs_size_ordered[i+1][0])/2

				model = iterate_combss(X, y, lam, epsilon = 1e-3)
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
def combssV5(X_train, y_train, X_test, y_test, 
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
	(model_list, lam_list) = combss_dynamicV5(X_train, y_train, q = q, nlam = nlam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, epoch=epoch, gd_maxiter= gd_maxiter, gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)
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
