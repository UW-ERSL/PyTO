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
from mmapy import mmasub, gcmmasub,kktcheck
from typing import Tuple
import numpy as np
import time

def runMMA(nVariables,nConstraints,optimizationFunction,X0,lowerBound,
			 upperBound, fErrAcceptable = 1e-4,gErrAcceptable = 1e-4,maxIterations = 250,kktTol = 1e-6,verbose = False):
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
	move = 1.0
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
		if (outit > 5 and fErr < fErrAcceptable and gErr < gErrAcceptable):
			if(verbose):
				print('Convergence reached with fEerr: ', fErr, ' and max(gval): ', max(gval))
			break
		if(verbose):
			print('iter: {}, f: {:.3e}, max(g): {:.3e}, fErr: {:.3e}'.format(
				outeriter, f0val if np.isscalar(f0val) else float(f0val), 
				gErr, fErr))
		f0valPrev = f0val.copy()
	if(verbose):
		print("time MMA: ", timeMMA, "time FuncEval: ", timeFuncEval)

	return [xval,f0val, df0dx, gval, dgdx,outit]

def sampleFunction1(xval: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
	'''
	Minimize:
		 x(1)^2 + x(2)^2 + x(3)^2
	
	Subject to: 
		(x(1)-5)^2 + (x(2)-2)^2 + (x(3)-1)^2 <= 9
		(x(1)-3)^2 + (x(2)-4)^2 + (x(3)-3)^2 <= 9
		0 <= x(j) <= 5, for j=1,2,3.
		
	Note that:
		f0val: a scalar
		df0dx: (N,1) array, where N is the number of variables
		gval: (M,1) array, where M is the number of constraints
		dgdx: (M,N) array
	'''
	f0val = xval[0][0]**2 + xval[1][0]**2 + xval[2][0]**2 # objective
	df0dx = 2 * xval # gradient of objective
	
	gval1 = ((xval.T - np.array([[5, 2, 1]]))**2).sum() - 9
	gval2 = ((xval.T - np.array([[3, 4, 3]]))**2).sum() - 9
	gval = np.array([[gval1, gval2]]).T # inequality constraints
	dgdx1 = 2 * (xval.T - np.array([[5, 2, 1]]))
	dgdx2 = 2 * (xval.T - np.array([[3, 4, 3]]))
	dgdx = np.concatenate((dgdx1, dgdx2)) # gradient of inequality constraints
	return f0val, df0dx, gval, dgdx

def sampleFunction2(xval: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
	'''
	Minimize:
		sin(x+y) + (x-y)*(x-y) - 1.5*x + 2.5*y +1
	
	Subject to: 
		-1 <= x0 <= 2
		-2 <= x1 <= 2
		
	'''
	x = xval[0]
	y = xval[1]
	f0val = np.sin(x+y) + (x-y)*(x-y) - 1.5*x + 2.5*y +1 # objective
	df0dx = np.array([np.cos(x+y) + 2*(x-y)- 1.5 ])
	df0dy = np.array([np.cos(x+y) - 2*(x-y) + 2.5])
	df0dx = np.concatenate((df0dx, df0dy)) # gradient of objective
	
	gval = np.array([[-1]]) # dummy inequality constraints
	dgdx = np.array([[0,0]]) # gradient of inequality constraints
	
	return f0val, df0dx, gval, dgdx

def twoSpringSystem(xval: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
	'''
	Minimize:
		potential energy of a 2 spring system
	
	Subject to: 
		v - u**3 <=0
		
	'''
	u = xval[0]
	v = xval[1]
	L12 = np.sqrt(u*u+(1+v)*(1+v))
	L13 = np.sqrt(u*u+(1-v)*(1-v))
	f0val = 0.5*(100*(L12-1)**2 + 50*(L13-1)**2) - (10*u+8*v)
	df0du = np.array([100.0*u*(np.sqrt(u**2 + (v + 1)**2) - 1)/np.sqrt(u**2 + (v + 1)**2) + 50.0*u*(np.sqrt(u**2 + (1 - v)**2) - 1)/np.sqrt(u**2 + (1 - v)**2) - 10])
	df0dv = np.array([-8 + 100.0*(v + 1)*(np.sqrt(u**2 + (v + 1)**2) - 1)/np.sqrt(u**2 + (v + 1)**2) + 50.0*(v - 1)*(np.sqrt(u**2 + (1 - v)**2) - 1)/np.sqrt(u**2 + (1 - v)**2)])
	df0dx = np.concatenate((df0du, df0dv)) # gradient of objective
	
	gval = np.array([u**3 -v]) #  inequality constraints
	dgdx = np.array([[3*u[0]**2,-1]]) # gradient of inequality constraints

	
	return f0val, df0dx, gval, dgdx

def  thompsonProblem(xval: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
	nPoints = int(len(xval)/3)
	pts = np.reshape(xval,(3,nPoints))
	f = 0
	for i in range(nPoints):
		pt_i = pts[:,i]
		for j in range(i+1,nPoints):
			pt_j = pts[:,j]
			dist = np.sqrt((pt_i[0] - pt_j[0])**2 + \
					(pt_i[1] - pt_j[1])**2 + \
					(pt_i[2] - pt_j[2])**2 ) +1e-12# avoid divide by zero
			f = f + 1/dist

	f0val = [f]

	df0dx = np.zeros((nPoints*3,1))
	for i in range(nPoints):
		pt_i = pts[:,i]
		for j in range(nPoints):
			if i == j:
				continue
			pt_j = pts[:,j]
			dist = np.sqrt((pt_i[0] - pt_j[0])**2 + (pt_i[1] - pt_j[1])**2 + (pt_i[2] - pt_j[2])**2) + 1e-12
			grad = -(pt_i - pt_j) / (dist**3)
			df0dx[3*i:3*i+3,0] += grad


	gval = np.zeros((nPoints,1))
	for i in range(nPoints):
		gval[i,0] = pts[0,i]*pts[0,i] + pts[1,i]*pts[1,i] + pts[2,i]*pts[2,i] -1
		
	dgdx = np.zeros((nPoints, nPoints*3))
	for i in range(nPoints):
		dgdx[i,3*i:3*i+3] = 2 * pts[:,i]

	return f0val, df0dx, gval, dgdx

