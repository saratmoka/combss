import numpy as np
from numpy.linalg import pinv, norm
from scipy.sparse.linalg import cg
from scipy.optimize import minimize, LinearConstraint, Bounds
import time, scipy

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

def grad_v1_t(X, t, beta, delta, y):
	n = y.shape[0]
	b_mult_t = np.multiply(beta, t)
	bracket_term = X.T@(X@b_mult_t) - delta*b_mult_t - X.T@y
	return (2/n)*np.multiply(beta,bracket_term)

def grad_v1_w(X, t, beta, delta, y, w):
	grad_t = grad_v1_t(X, t, beta, delta, y)
	sigmoid_w = w_to_t(w)
	second_term = sigmoid_w*(1-sigmoid_w)
	return np.multiply(grad_t, second_term)

def grad_v1_beta(X, t, beta, delta, y):
	n = y.shape[0]
	b_mult_t = np.multiply(beta, t)
	XTy = X.T@y
	bracket_term = X.T@(X@b_mult_t) - np.multiply(t,XTy) + delta*beta - delta*np.multiply(t,b_mult_t)
	return (2/n)*np.multiply(t, bracket_term)

def adam_v1(X, y, gam1 = 0.9, gam2 = 0.999, alpha = 0.1, epsilon = 10e-8, maxiter = 1e3, tol = 1e-8, tau = 0.5):
	# To compensate for data types fed into Adam, otherwise the output will be of incorrect dimension.
	y = np.reshape(y,(-1,1))

	# Initialising data-related variables
	delta = X.shape[0]
	p = X.shape[1]

	# Initialising Adam-related variables
	i = 0
	v_beta, v_w, u_beta, u_w = np.zeros((p,1)), np.zeros((p,1)), np.zeros((p,1)), np.zeros((p,1))
	v_betas, v_ws, u_betas, u_ws = np.zeros((p,1)), np.zeros((p,1)), np.zeros((p,1)), np.zeros((p,1))

	stop = False
	converge = False

	# beta_new = np.random.randn(p,1)
	# beta_new = np.ones((p,1))
	beta_new = 1/2*np.ones((p,1))
	w_new = np.zeros((p,1))

	while not stop:
		# Initialisation parameters
		beta_curr = beta_new.copy()
		w_curr = w_new.copy()
		t_curr = w_to_t(w_curr)

		# Perform updates for beta
		gradbeta = grad_v1_beta(X, t_curr, beta_curr, delta, y)
		v_beta = gam1*v_beta + (1 - gam1)*gradbeta
		u_beta = gam2*u_beta + (1 - gam2)*np.multiply(gradbeta, gradbeta)
		v_betas = v_beta/(1-gam1**(i+1))
		u_betas = u_beta/(1-gam2**(i+1))
		beta_new = beta_curr - alpha*np.divide(v_betas,(np.sqrt(u_betas) + epsilon))

		# Perform updates for w
		gradw = grad_v1_w(X, t_curr, beta_curr, delta, y, w_curr)
		v_w = gam1*v_w + (1 - gam1)*gradw
		u_w = gam2*u_w + (1 - gam2)*np.multiply(gradw, gradw)
		v_ws = v_w/(1-gam1**(i+1))
		u_ws = u_w/(1-gam2**(i+1))
		w_new = w_curr - alpha*np.divide(v_ws,(np.sqrt(u_ws) + epsilon))
		t_new = w_to_t(w_new)

		'''
		print(f"v beta: {v_beta}")
		print(f"u beta: {u_beta}")
		print(f"v betas: {v_betas}")
		print(f"u betas: {u_betas}")
		print(f"Delta beta: {beta_new - beta_curr}")
		print(f"Delta w: {w_new - w_curr}")
		'''

		# Assess stopping conditions
		if (i > maxiter):
			stop = True
		else:
			delta_beta = np.linalg.norm((beta_new - beta_curr),2)
			delta_t =  np.linalg.norm((t_new - t_curr),2)
			if ((delta_beta + delta_t) < tol):
				gradbeta_new = grad_v1_beta(X, t_curr, beta_new, delta, y)
				gradbeta_curr = grad_v1_beta(X, t_curr, beta_curr, delta, y)

				gradw_new = grad_v1_w(X, t_new, beta_curr, delta, y, w_new)
				gradw_curr = grad_v1_w(X, t_curr, beta_curr, delta, y, w_curr)

				delta_gradbeta = np.linalg.norm((gradbeta_new - gradbeta_curr),2)
				delta_gradt = np.linalg.norm((gradw_new - gradw_curr),2)
				if ((delta_gradbeta + delta_gradt) < tol):
					stop = True
		
		# Iterate through counter
		i = i + 1
	
	model = np.where(t_new > tau)[0]

	if i + 1 < maxiter:
		converge = True

	print(f"Final Adam Beta: {beta_new}")
	print(f"Final Adam t: {t_new}")
	print(f"Final Adam model: {model}")
	print(f"Final Adam convergence: {converge}")
	print(f"Final iterations: {i}")
	
	return beta_new, t_new, model, converge, i+1