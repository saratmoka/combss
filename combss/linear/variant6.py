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
Best Subset Selection: Heuristic inspired by Splitting Method, Subset Permutations
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


def iterate_logic(s_0, s_1, f_s0, f_s1, level, split_created):
	
	s = s_0

	if (f_s0 < level):
		if (f_s1 < level):
			bin_seed = int(time.time())
			np.random.seed(bin_seed)
			s = random.choice([s_0, s_1])
			split_created = True
		else:
			s = s_0
			split_created = True
	elif (f_s1 < level):
		s = s_1
		split_created = True
	
	return s, split_created


def iterate_combss(X, y, q, lam, epsilon):

	(n, p) = X.shape
 	
	# Number of Candidates
	if p < 1000:
		C = p*20
	else:	  	
		C = int(p/8) 

	N = int(C/5) # Number of Candidates promoted to next level
	candidates = []

	while len(candidates) < C:
		# Generate a random binomial vector
		random_model = np.zeros(p, dtype=int)
	
		# Randomly choose q indices to set to 1
		random_ones = np.random.choice(p, q, replace=False)
		random_model[random_ones] = 1
		candidates.append(random_model)
	
	fs_values = np.array([obj_fn(s, X, y, lam) for s in candidates])
	pairs = list(zip(candidates, fs_values))

	sorted_pairs = sorted(pairs, key=lambda x: x[1])

	sorted_candidates = [pair[0] for pair in sorted_pairs]

	top_candidates = sorted_candidates[:N].copy()

	# Get sorted f() values using sorted indices
	fs_sorted = [pair[1] for pair in sorted_pairs]

	# Find the largest and 4th largest f() values
	optimal = fs_sorted[0]
	print(f'initial optimal: {optimal}')

	level = fs_sorted[N-1]
	print(f'initial level: {level}')

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
				
					zeros_ind = np.where(s == 0)
					ones_ind = np.where(s != 0)

					random.seed(time.time())
					s_0 = np.copy(s)
					s_in = random.choice(zeros_ind)
					s_out = random.choice(ones_ind)
					s_0[s_in] = 1
					s_0[s_out] = 0

					random.seed(time.time())
					s_1 = np.copy(s)
					s_in = random.choice(zeros_ind)
					s_out = random.choice(ones_ind)
					s_1[s_in] = 1
					s_1[s_out] = 0

					f_s0 = obj_fn(s_0, X, y, lam)
					f_s1 = obj_fn(s_1, X, y, lam)

					s, split_created, = iterate_logic(s_0 = s_0, s_1 = s_1, f_s0 = f_s0, f_s1 = f_s1, 
												level = level, split_created = split_created)
					
					print(f'f_s0 = {f_s0}')
					print(f'f_s1 = {f_s1}')

					if split_created is True:
						duplicate = any(np.array_equal(s, arr) for arr in promoted_list)
						print(f'duplicate: {duplicate}')
						if duplicate:
							split_created = False
							continue
						else:
							splits.append(s)
							promoted_list.append(s)
							break	
					
					if len(splits) >= n:
						break
					print(f'len(promoted_list) = {len(promoted_list)}')
					print(f'len(splits) = {len(splits)}')

		fs_values = np.array([obj_fn(s, X, y, lam) for s in promoted_list])
		pairs = list(zip(promoted_list, fs_values))

		sorted_pairs = sorted(pairs, key=lambda x: x[1])

		sorted_candidates = [s[0] for s in sorted_pairs]

		top_candidates = sorted_candidates[:N].copy()

		fs_sorted = [pair[1] for pair in sorted_pairs]

		optimal = fs_sorted[0]
		print(f'optimal: {optimal}')

		level = fs_sorted[N-1]
		print(f'level: {level}')


		promoted_list = sorted_candidates[:N].copy()

	final_indices = top_candidates[0]
	model = np.where(final_indices != 0)[0]

	return model


def combss_dynamicV6(X, y, 
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
		model = iterate_combss(X, y, q, lam, epsilon = 1e-3)
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

				model = iterate_combss(X, y, q, lam, epsilon = 1e-3)
				len_model = model.shape[0]

				lam_list.append(lam)
				# t_list.append(t_final)
				# beta_list.append(beta)
				model_list.append(model)
				lam_vs_size.append(np.array((lam, len_model)))    
				count_lam += 1

		stop = True
	
	return  (model_list, lam_list)


def combssV6(X_train, y_train, X_test, y_test, 
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
	(model_list, lam_list) = combss_dynamicV6(X_train, y_train, q = q, nlam = nlam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, epoch=epoch, gd_maxiter= gd_maxiter, gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)
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
