import numpy as np
import pandas as pd
from numpy.linalg import pinv, norm
import time
import helpers

def ADAM_combssV1(X, y, lam, gam1 = 0.9, gam2 = 0.999, alpha = 0.1, epsilon = 10e-8, maxiter = 1e3, tol = 1e-8, tau = 0.5):
	# To compensate for data types fed into Adam, otherwise the output will be of incorrect dimension.
	#y = np.reshape(y, (-1,1))

	# Initialising data-related variables
	(n, p) = X.shape 
	delta = n
	
	# Initialising Adam-related variables
	i = 0
	v_beta, v_w, u_beta, u_w = np.zeros(p), np.zeros(p), np.zeros(p), np.zeros(p)
	
	stop = False
	converge = False
	beta_seq = []
	t_seq = []
	
	beta_new = np.random.randn(p)
	w_new = np.zeros(p)

	while not stop:
		
		# Initialisation parameters
		beta_curr = beta_new.copy()
		w_curr = w_new.copy()
		t_curr = helpers.w_to_t(w_curr)
		
		beta_seq.append(beta_curr.copy())
		t_seq.append(t_curr.copy())

		# Perform updates for beta
		gradbeta = helpers.grad_v1_beta(X, t_curr, beta_curr, delta, y)
		v_beta = gam1*v_beta + (1 - gam1)*gradbeta
		u_beta = gam2*u_beta + (1 - gam2)*(gradbeta*gradbeta)
		v_betas = v_beta/(1-gam1**(i+1))
		u_betas = u_beta/(1-gam2**(i+1))
		beta_new = beta_curr - alpha*np.divide(v_betas, (np.sqrt(u_betas) + epsilon))

		# Perform updates for w
		gradw = helpers.grad_v1_w(X, t_curr, beta_curr, delta, lam, y, w_curr)
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
			diff_beta = np.linalg.norm((beta_new - beta_curr), 2)
			diff_t =  np.linalg.norm((t_new - t_curr), 2)
			if ((diff_beta + diff_t) < tol):
				gradbeta_new = helpers.grad_v1_beta(X, t_curr, beta_new, delta, y)
				gradbeta_curr = helpers.grad_v1_beta(X, t_curr, beta_curr, delta, y)

				gradw_new = helpers.grad_v1_w(X, lam, t_new, beta_curr, delta, y, w_new)
				gradw_curr = helpers.grad_v1_w(X, lam, t_curr, beta_curr, delta, y, w_curr)

				diff_gradbeta = np.linalg.norm((gradbeta_new - gradbeta_curr),2)
				diff_gradt = np.linalg.norm((gradw_new - gradw_curr),2)
				if ((diff_gradbeta + diff_gradt) < tol):
					stop = True
		
		# Iterate through counter
		i = i + 1
	
	model = np.where(t_new > tau)[0]

	if i + 1 < maxiter:
		converge = True
	
	
	return t_new, model, converge, i+1, beta_seq, t_seq

def combss_dynamicV1(X, y, 
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
	while not stop:
		t_final, model, converge, _, _, _ = ADAM_combssV1(X, y, lam, gam1 = 0.9, gam2 = 0.999, alpha = 0.1, epsilon = 10e-8, maxiter = 1e3, tol = 1e-8, tau = 0.5)

		len_model = model.shape[0]

		lam_list.append(lam)
		# t_list.append(t_final)
		# beta_list.append(beta)
		# t_seq_list.append(t_seq)
		# beta_seq_list.append(beta_seq)
		model_list.append(model)
		# converge_list.append(converge)
		lam_vs_size.append(np.array((lam, len_model)))
		count_lam += 1
		print(len_model)
		if len_model >= q or count_lam > nlam*fstage_frac:
			stop = True
		lam = lam/2
		#print('lam = ', lam, 'len of model = ', len_model)
		


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

				t_final, model, converge,_ ,_ ,_= ADAM_combssV1(X, y, lam, gam1 = 0.9, gam2 = 0.999, alpha = 0.1, epsilon = 10e-8, maxiter = 1e3, tol = 1e-8, tau = 0.5)

				len_model = model.shape[0]

				lam_list.append(lam)
				# t_list.append(t_final)
				# beta_list.append(beta)
				# t_seq_list.append(t_seq)
				# beta_seq_list.append(beta_seq)
				model_list.append(model)
				# converge_list.append(converge)
				lam_vs_size.append(np.array((lam, len_model)))    
				count_lam += 1

		stop = True
	
	return  (model_list, lam_list)

def combssV1(X_train, y_train, X_test, y_test, 
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
	(model_list, lam_list) = combss_dynamicV1(X_train, y_train, q = q, nlam = nlam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, epoch=epoch, gd_maxiter= gd_maxiter, gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)
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
