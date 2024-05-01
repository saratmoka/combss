import numpy as np
import pandas as pd
from numpy.linalg import pinv, norm
from scipy.sparse.linalg import cg
import time

'''
Helper functions for COMBSS
'''

def t_to_w(t):
	"""
	Function to convert t to w, and it is used in converting box-constraint problem to an unconstraint one.
	"""
	w = np.log(t/(1-t))
	return w

def w_to_t(w):
	"""
	Function to convert w to t, and it is used in converting solution of the unconstraint problem to the constrained case.
	"""  
	t = 1/(1+np.exp(-w))
	return t

def f_grad_cg(t, X, y, XX, Xy, Z, lam, delta, beta,  c,  g1, g2,
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



'''
COMBSS Variant 0 Functions
'''

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
		max_norm = True, # default we use max norm as the termination condition.
		epoch=10,
		
		## Truncation parameters
		tau = 0.5,
		eta = 0.0, 
		
		## Parameters for Conjugate Gradient method
		cg_maxiter = None,
		cg_tol = 1e-5):
	"""
	Implementation of the ADAM optimizer for combss. 
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
		
	w = t_to_w(t)
	
	t_trun = t.copy()
	t_prev = t.copy()
	active = p
	
	u = np.zeros(p)
	v = np.zeros(p)
	
	beta_trun = np.zeros(p)  

	c = np.zeros(p)
	g1 = np.zeros(n)
	g2 = np.zeros(n)
	
	
	count_to_term = 0
	
	# t_seq = []
	# beta_seq = []
	
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
		grad_trun, beta_trun, c, g1, g2 = f_grad_cg(t_trun, X, y, XX, Xy, Z, lam, delta, beta_trun[M_trun],  c[M_trun], g1, g2)
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
		
		beta = np.zeros(p)
		beta[M] = beta_trun

		t_trun = t[M] 
		
		# t_seq.append(t.copy())
		# beta_seq.append(beta)

		if max_norm:
			norm_t = max(np.abs(t - t_prev))
			if l > 10000:
				print('l', l)
				
				if l%100 == 0:
					print('\t norm diff', norm_t)
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
	print(f'conjugate gradient iteration: {l}')

	if l+1 < gd_maxiter:
		converge = True
	else:
		converge = False
	return  t, model, converge, l+1

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
		
	w = t_to_w(t)
	
	t_trun = t.copy()
	t_prev = t.copy()
	active = p
	
	beta_trun = np.zeros(p)  

	c = np.zeros(p)
	g1 = np.zeros(n)
	g2 = np.zeros(n)
	
	count_to_term = 0
	
	# t_seq = []
	# beta_seq = []
	
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
		grad_trun, beta_trun, c, g1, g2 = f_grad_cg(t_trun, X, y, XX, Xy, Z, lam, delta, beta_trun[M_trun],  c[M_trun], g1, g2)
		w_trun = w[M]
		grad_trun = 2*grad_trun*(w_trun*np.exp(- w_trun*w_trun))
		
		## BGD Updates
		w_trun = w_trun - alpha*grad_trun 
		w[M] = w_trun
		t[M] = w_to_t(w_trun)
		
		w[t <= eta] = 0.0
		t[t <= eta] = 0.0
		
		beta = np.zeros(p)
		beta[M] = beta_trun

		t_trun = t[M] 
		
		# t_seq.append(t.copy())
		# beta_seq.append(beta)
		
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

def combss_dynamicV0(X, y, 
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
		t_final, model, converge, _ = ADAM_combss(X, y, lam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, epoch=epoch, gd_maxiter=gd_maxiter,gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)

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

				t_final, model, converge, _ = ADAM_combss(X, y, lam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, epoch=epoch, gd_maxiter=gd_maxiter,gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)

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

def combssV0(X_train, y_train, X_test, y_test, 
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
	(model_list, lam_list) = combss_dynamicV0(X_train, y_train, q = q, nlam = nlam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, epoch=epoch, gd_maxiter= gd_maxiter, gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)
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



'''
COMBSS Variant 1 Functions
'''

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
		t_curr = w_to_t(w_curr)
		
		beta_seq.append(beta_curr.copy())
		t_seq.append(t_curr.copy())

		# Perform updates for beta
		gradbeta = grad_v1_beta(X, t_curr, beta_curr, delta, y)
		v_beta = gam1*v_beta + (1 - gam1)*gradbeta
		u_beta = gam2*u_beta + (1 - gam2)*(gradbeta*gradbeta)
		v_betas = v_beta/(1-gam1**(i+1))
		u_betas = u_beta/(1-gam2**(i+1))
		beta_new = beta_curr - alpha*np.divide(v_betas, (np.sqrt(u_betas) + epsilon))

		# Perform updates for w
		gradw = grad_v1_w(X, t_curr, beta_curr, delta, lam, y, w_curr)
		v_w = gam1*v_w + (1 - gam1)*gradw
		u_w = gam2*u_w + (1 - gam2)*(gradw*gradw)
		v_ws = v_w/(1-gam1**(i+1))
		u_ws = u_w/(1-gam2**(i+1))
		w_new = w_curr - alpha*np.divide(v_ws, (np.sqrt(u_ws) + epsilon))
		t_new = w_to_t(w_new)

		# Assess stopping conditions
		if (i > maxiter):
			stop = True
		else:
			diff_beta = np.linalg.norm((beta_new - beta_curr), 2)
			diff_t =  np.linalg.norm((t_new - t_curr), 2)
			if ((diff_beta + diff_t) < tol):
				gradbeta_new = grad_v1_beta(X, t_curr, beta_new, delta, y)
				gradbeta_curr = grad_v1_beta(X, t_curr, beta_curr, delta, y)

				gradw_new = grad_v1_w(X, lam, t_new, beta_curr, delta, y, w_new)
				gradw_curr = grad_v1_w(X, lam, t_curr, beta_curr, delta, y, w_curr)

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



'''
COMBSS Variant 2 Functions
'''
def adam_w(X, y, beta, lam, gam1 = 0.9, gam2 = 0.999, alpha = 0.1, epsilon = 10e-8, maxiter = 1e3, tol = 1e-8):

	# Initialising data-related variables
	(n, p) = X.shape 
	delta = n
	
	# Initialising Adam-related variables
	i = 0
	v_w, u_w = np.zeros(p), np.zeros(p)
	
	stop = False
	converge = False
	
	w_new = np.zeros(p)
	t_new = w_to_t(w_new)

	while not stop:
		
		# Initialisation parameters
		w_curr = w_new.copy()
		t_curr = w_to_t(w_curr)

		# Perform updates for w
		gradw = grad_v1_w(X, t_curr, beta, delta, lam, y, w_curr)
		v_w = gam1*v_w + (1 - gam1)*gradw
		u_w = gam2*u_w + (1 - gam2)*(gradw*gradw)
		v_ws = v_w/(1-gam1**(i+1))
		u_ws = u_w/(1-gam2**(i+1))
		w_new = w_curr - alpha*np.divide(v_ws, (np.sqrt(u_ws) + epsilon))
		t_new = w_to_t(w_new)

		# Assess stopping conditions
		if (i > maxiter):
			stop = True
		else:
			diff_t = np.linalg.norm((t_new- t_curr), 2)
			if (diff_t < tol):
				gradt_new = grad_v1_t(X, t_new, beta, delta, y)
				gradt_curr = grad_v1_t(X, t_curr, beta, delta, y)

				diff_gradt = np.linalg.norm((gradt_new - gradt_curr),2)
				if (diff_gradt < tol):
					stop = True
		
		# Iterate through counter
		i = i + 1
	
	if i + 1 < maxiter:
		converge = True
	
	return w_new, t_new, converge, i+1

def controller_combssV2(X, y, lam, 
		# Adam parameters				
		gam1 = 0.9, 
		gam2 = 0.999, 
		alpha = 0.1, 
		epsilon = 10e-8, 
		maxiter = 1e5, 
		tol = 1e-5, 
		tau = 0.5,
		eta = 0.0,
		
		## Parameters for Conjugate Gradient method
		cg_maxiter = 10,
		cg_tol = 1e-5,
		controller_maxiter = np.inf):

	# Initialising data-related variables
	(n, p) = X.shape 
	delta = n
	Xy = (X.T@y)/n
	XX = (X.T@X)/n
	Z = XX.copy()
	np.fill_diagonal(Z, np.diagonal(Z) - (delta/n))
	
	w = np.zeros(p)
	t = w_to_t(w)

	beta_trun = np.zeros(p)
	beta_ref = beta_trun.copy()
	t_trun = t.copy()
	active = p

	c = np.zeros(p)
	g1 = np.zeros(n)
	g2 = np.zeros(n)
	
	stop = False
	converge = False

	i = 0
	while not stop:
		M = np.nonzero(t)[0] ## Indices of t correponds to elements greater than eta. 
		M_trun = np.nonzero(t_trun)[0] 
		active_new = M_trun.shape[0]
		beta_ref = beta_trun.copy()
		
		if active_new != active:
			## Find the effective set by removing the columns and rows corresponds to zero t's
			XX = XX[M_trun][:, M_trun]
			Z = Z[M_trun][:, M_trun]
			X = X[:, M_trun]
			Xy = Xy[M_trun]
			active = active_new
			t_trun = t_trun[M_trun]
		
		## Compute gradient for the effective terms
		_, beta_trun, c, g1, g2 = f_grad_cg(t_trun, X, y, XX, Xy, Z, lam, delta, beta_trun[M_trun],  c[M_trun], g1, g2, cg_maxiter=cg_maxiter, cg_tol=cg_tol)

		# Perform updates for w
		w_trun, t, converge, _ = adam_w(X, y, beta_trun, lam, gam1 = gam1, gam2 = gam2, alpha = alpha, epsilon = epsilon, maxiter = maxiter, tol = tol)

		w[M] = w_trun
		t[M] = w_to_t(w_trun)
		
		w[t <= eta] = -np.inf
		t[t <= eta] = 0.0
		
		beta = np.zeros(p)
		beta[M] = beta_trun

		t_trun = t[M] 

		# Assess stopping conditions
		diff_beta = np.linalg.norm((beta[M] - beta_ref[M]), 2)
		if (diff_beta < tol):
			converge = True
			break

			
	model = np.where(t > tau)[0]
	i = i + 1

	if i + 1 > controller_maxiter:
		converge = False
		stop = True
	
	return t, model, converge, i+1

def combss_dynamicV2(X, y, 
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
		t_final, model, converge, _ = controller_combssV2(X, y, lam, gam1 = 0.9, gam2 = 0.999, alpha = 0.1, epsilon = 10e-8, maxiter = 1e3, tol = 1e-8, tau = 0.5, cg_maxiter = cg_maxiter)

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

				t_final, model, converge,_ = controller_combssV2(X, y, lam, gam1 = 0.9, gam2 = 0.999, alpha = 0.1, epsilon = 10e-8, maxiter = 1e3, tol = 1e-8, tau = 0.5, cg_maxiter=cg_maxiter)

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
	(model_list, lam_list) = combss_dynamicV2(X_train, y_train, q = q, nlam = nlam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, epoch=epoch, gd_maxiter= gd_maxiter, gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)
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
