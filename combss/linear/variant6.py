import numpy as np
import pandas as pd
from numpy.linalg import pinv, norm
from scipy.sparse.linalg import cg
import time
import helpers
import random
from functools import partial
import math
import secrets
from math import comb


'''
Best Subset Selection: Heuristic inspired by Splitting Method, Subset Permutations
'''

def gen_permutation(s):

	zeros_ind = np.where(s == 0)[0]
	ones_ind = np.where(s != 0)[0]
	
	s_0 = s.copy()
	random.seed(secrets.randbits(64))
	s_in = random.choice(zeros_ind)
	s_out = random.choice(ones_ind)
	s_0[s_in] = 1
	s_0[s_out] = 0

	s_1 = s.copy()
	random.seed(secrets.randbits(64))
	s_in = random.choice(zeros_ind)
	s_out = random.choice(ones_ind)
	s_1[s_in] = 1
	s_1[s_out] = 0
	
	while (s_0==s_1).all():
		s_1 = s.copy()
		random.seed(secrets.randbits(64))
		s_in = random.choice(zeros_ind)
		s_out = random.choice(ones_ind)
		s_1[s_in] = 1
		s_1[s_out] = 0

	return s_0, s_1

def obj_fn(t, X, y):
	(n,p) = X.shape
	S = np.nonzero(t)[0]

	if len(S) == 0:
		obj = 1/n*(y.T@y)
	else:
		Xs = X[:,S]
		bt = pinv(Xs.T@Xs)@(Xs.T@y)
		n = np.shape(X)[0]
		obj = 1/n*((y-Xs@bt).T@(y-Xs@bt))
	return obj


def iterate_logic(s_0, s_1, f_s0, f_s1, level, split_created):
	
	s = s_0

	if (f_s0 < level):
		if (f_s1 < level):
			bin_seed = int(time.time())
			np.random.seed(bin_seed)
			s = random.choice([s_0, s_1]).copy()
			split_created = True
		else:
			s = s_0.copy()
			split_created = True
	elif (f_s1 < level):
		s = s_1.copy()
		split_created = True
	
	return s, split_created


def iterate_combss(X, y, u, q, epsilon):

	(n, p) = X.shape

	if u == 0:
		return np.array([], dtype=np.int64)
	elif u == p:
		return np.arange(0, q, dtype=np.int64)
 	
	if p < 1000:
		C = p*20
	else:	  	
		C = int(p/8) 
	
	C = min(C, comb(p,u))

	N = int(C/5) # Number of Candidates promoted to next level
	print(f'For a model of size {u}, C = {C} candidates are explored, and N = {N} are promoted to the next level.')

	candidates = []
	seen = set()

	while len(candidates) < C:
		# Generate a random binomial vector
		random_model = np.zeros(p, dtype=int)
	
		# Randomly choose q indices to set to 1
		random_ones = np.random.choice(p, u, replace=False)
		random_model[random_ones] = 1
		candidates.append(random_model)
	
	fs_values = np.array([obj_fn(s, X, y) for s in candidates])
	pairs = list(zip(candidates, fs_values))

	sorted_pairs = sorted(pairs, key=lambda x: x[1])

	fs_sorted = [pair[1] for pair in sorted_pairs]

	sorted_candidates = [pair[0] for pair in sorted_pairs]
	top_candidates = sorted_candidates[:N].copy()

	# Find the largest and 4th largest f() values
	optimal = fs_sorted[0]
	print(f'initial optimal: {optimal}')

	level = fs_sorted[N-1]
	print(f'initial level: {level}')

	promoted_list = sorted_candidates[:N].copy()

	level_old = np.inf
	maxiter = 100000
	i = 0
	exhausted = False
	while abs(level_old - level)/level > epsilon and i < maxiter:
		level_old = level
		candidates = top_candidates
		N = len(top_candidates)
		for j in range(N):

			if i > maxiter or exhausted:
				exhausted = True
				break
			s = candidates[j]

			if j < N-1:
				n = math.ceil((C-N)/N)
			else:
				n = C - N - (N-1)*math.ceil((C-N)/N)

			# print(f'For {u}, C = {C} and N = {N}, n should be 4 and is {n}.')


			splits = []
			while len(splits) < n: 
				# print(f'check c = {i}')

				split_created = False
				if i > maxiter or exhausted:
					exhausted = True
					break

				while split_created is False:
					# print(f'check d = {i}')

					if i > maxiter or exhausted:
						exhausted = True
						break
					time_seed = random.randint(1,2^20)

					np.random.seed(time_seed)
					indices = np.arange(p)  
					np.random.shuffle(indices)
					
					# print(f'check e = {i}')

					s_0, s_1 = gen_permutation(s)

					# print(f'check f = {i}')

					f_s0 = obj_fn(s_0, X, y)
					f_s1 = obj_fn(s_1, X, y)
					
					# print(f'check f_s0 = {f_s0}')
					# print(f'check f_s1 = {f_s1}')

					s, split_created, = iterate_logic(s_0 = s_0, s_1 = s_1, f_s0 = f_s0, f_s1 = f_s1, 
												level = level, split_created = split_created)
					
					# print(f'f_s0 = {f_s0}')
					# print(f'f_s1 = {f_s1}')
					# print(f'split_created = {split_created}')

					# print(f'len(promoted_list) = {len(promoted_list)}')
					# print(f'len(splits) = {len(splits)}')

					if split_created is True:
						# BIG ERROR, never letting me get through this duplicate stage.
						# duplicate = any(tuple(s.flat) == tuple(arr.flat) for arr in promoted_list)
						duplicate = False
						# print(f'duplicate: {duplicate}')
						# print(f'step: b')
						if duplicate:
							split_created = False
							# print(f'step: c')
							continue
						else:
							splits.append(s)
							promoted_list.append(s)
							# print(f'step: d')
							break	
					if len(splits) >= n:
						# print(f'step: e')
						break
					# print(f'len(promoted_list) = {len(promoted_list)}')
					# print(f'len(splits) = {len(splits)}')
					i += 1
					if i > maxiter:
						exhausted = True
						break
		# print(f'step: f')
		fs_values = np.array([obj_fn(s, X, y) for s in promoted_list])
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
		i += 1

	final_indices = top_candidates[0]
	model = np.where(final_indices != 0)[0]

	return model


def combss_subsets(X, y, q = None): 

	(n, p) = X.shape
	
	# If q is not given, take q = n.
	if q == None:
		q = min(n, p)

	# Lists to store the findings
	model_list = []
	model_sizes = []

	for i in range(q+1):
		model = iterate_combss(X, y, i, q, epsilon = 1e-3)

		model_list.append(model)
		model_sizes.append(i)
	
	return (model_list, model_sizes)


def combssV6(X_train, y_train, X_test, y_test, q = None):
	
	(n, p) = X_train.shape

	# If q is not given, take q = n
	if q == None:
		q = min(n, p)
	
	tic = time.process_time()
	(model_list, model_sizes) = combss_subsets(X_train, y_train, q = q)
	toc = time.process_time()

	mse_list = [] 
	beta_list = []
	print(f'model_list = {model_list}')
	
	for i in range(q+1):
		model_final = model_list[i]
		print(f'model_final = {model_final}')

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
	q_opt = model_sizes[ind_opt]
	model_opt = model_sizes[ind_opt]
	mse_opt = mse_list[ind_opt] 
	beta_opt = beta_list[ind_opt]
	
	return model_opt, mse_opt, beta_opt, q_opt, toc - tic
