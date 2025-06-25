"""
GCMMA-MMA-Python

This file is part of GCMMA-MMA-Python. GCMMA-MMA-Python is licensed under the terms of GNU 
General Public License as published by the Free Software Foundation. For more information and 
the LICENSE file, see <https://github.com/arjendeetman/GCMMA-MMA-Python>. 

The orginal work is written by Krister Svanberg in MATLAB. This is the Python implementation 
of the code written by Arjen Deetman.
"""

# Loading modules
from __future__ import division
from mmapy import mmasub, kktcheck
import numpy as np
import time

def runMMA(nVariables,nConstraints,optimizationFunction,X0,lowerBound,
			 upperBound, fTolerance = 1e-4,gTolerance = 1e-4,
			 maxIterations = 250,timeLimitSecs = 3600,move_limit = 0.2,kktTol = 1e-6,verbose = False):
	'''
	 Input
		nVariables: scalar (N)
		nConstraints: scalar (M)
		optimizationFunction must return a tuple consisting of
			f0val: a scalar
			df0dx: (N,1) array
			gval: (M,1) array
			dgdx: (M,N) array, i.e., the Jacobian
		X0: (N,1) array
		lowerBound: (N,1) array
		upperBound: (N,1) array
		useGCMMAsub: If True, use gccmmasub else use mmasub
		maxIterations (optional): int
		kktTol (optional):  float
		verbos (optional): Boolean to print
	'''
	# Set numpy print options
	np.set_printoptions(precision=4, formatter={'float': '{:0.4f}'.format})
	
	# Initial settings
	n = nVariables #Number of variables
	m = nConstraints #Number of constraints
	xval = X0 # initial values
	eeem = np.ones((m, 1)) # a convenience array
	zerom = np.zeros((m, 1)) # a convenience array
	xmin = lowerBound #lower bound
	xmax = upperBound #upper bound
	maxoutit = maxIterations #  maximum iterations
	kkttol = kktTol
	 
	# Other arrays and params
	low = xmin.copy()
	upp = xmax.copy()
	
	xold1 = xval.copy() 
	xold2 = xval.copy()
	move = move_limit
	c = 100 * eeem
	d = eeem.copy()
	a0 = 1
	a = zerom.copy()
	outeriter = 0
	# Calculate function values and gradients of the objective and constraints functions
	if outeriter == 0:
		f0val, df0dx, gval, dgdx = optimizationFunction(xval)

	# The iterations start
	kktnorm = kkttol + 10
	outit = 0
	timeMMA = 0.0
	timeFuncEval = 0.0
	f0Scaling = f0val if abs(f0val) >1e-6 else 1
	f0valPrev = f0val/f0Scaling
	fErr = 1
	gErr = 1
	tStart = time.time()
	while (kktnorm > kkttol and outit < maxoutit) :
		outit += 1
		outeriter += 1
		startTime = time.time()
		xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
				m, n, outeriter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, gval, dgdx, low, upp, a0, a, c, d, move)
		timeMMA += time.time() - startTime
		# Some vectors are updated:
		xold2 = xold1.copy()
		xold1 = xval.copy()
		xval = xmma.copy()
		
		# Re-calculate function values and gradients of the objective and constraints functions
		startTime = time.time()
		f0val, df0dx, gval, dgdx = optimizationFunction(xval)
		f0val = f0val / f0Scaling
		df0dx = df0dx / f0Scaling # scale the gradient of the objective
		timeFuncEval += time.time() - startTime
		
		
		# The residual vector of the KKT conditions is calculated
		startTime = time.time()
		residu, kktnorm, residumax = kktcheck(
			m, n, xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, xmin, xmax, df0dx, gval, dgdx, a0, a, c, d)
		timeMMA += time.time() - startTime
		fErr = np.abs(f0val - f0valPrev) / (1e-10 + np.abs(f0val))
		gErr = np.max(gval)
		if(verbose):
			print('iter: {}, f: {:.3e}, max(g): {:.3e}, fErr: {:.3e}'.format(
				outeriter, f0val if np.isscalar(f0val) else float(f0val), 
				gErr, fErr))
		if (fErr < fTolerance and gErr < gTolerance):
			if(verbose):
				print(f'Convergence reached with fEerr: {fErr:.3e} and max(gval): {gErr:.3e}')
			break
	
		f0valPrev = f0val.copy()
		if (time.time() - tStart > timeLimitSecs):
			print(f"Time limit of {timeLimitSecs:0.2f} reached, exiting...")
			break	
	if(verbose):
		print(f"MMA (secs):  {timeMMA:0.2f}")
		print(f"FuncEval (secs): {timeFuncEval:0.2f}")

	return [xval,f0val, df0dx, gval, dgdx,outit]
