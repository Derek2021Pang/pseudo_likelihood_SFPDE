# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:22:59 2021

@author: pangg
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma
from Solvers import FDM_time__stochastic_fractional_diffusion as FDM_solver
import matplotlib.pyplot as plt
import scipy
import time


def nll_fun(args, X, U, Nx, Nt, paras_est_flag, paras_true):
    """
    
    

    Parameters
    ----------
    args : list
        Initial guesses for parameters to be estimated
    X : 1d array of size Nx+1
        Space grid
    U : 2D array of size (Nx+1)*(Nt+1)
        Observation (Full or partial)
    Nx : Int
        Number of space grid points: N0x/rx
    Nt : Int
        Number of time grid points: N0t/rt
    paras_est_flag : list
        A list of boolean variables; if the i-th element of the list is True, 
        then the corresponding parameter needs to be estimated. The correspondence of 
        the index of the list and the parameters is [Fractional order, Diffusion coefficient, Noise magnitude].
    paras_true : List
        A list of [true fractional order, true diffusion coefficient, true noise magnitude].

    Returns
    -------
    Float
        Negative log-likelihood

    """
    
    

    def f_fun(x, t, alpha):
        return 2*t*x**2*(1-x)**2 - 2 * t**(1+alpha)/gamma(2+alpha) * (2-12*x+12*x**2)

    alpha, c, eps = paras_true
    alpha_true = paras_true[0]
    
    count = 0
    if paras_est_flag[0] == True:
        alpha = args[count]
        count += 1
    if paras_est_flag[1] == True:
        c = args[count]
        count += 1
    if paras_est_flag[2] == True:
        eps = args[count]
        
          
     
        
        
    nll = 0.0
    dx = 1./Nx
    dt = 1./Nt
    A = np.diag(np.ones(Nx-2), -1) + np.diag(np.ones(Nx-2), 1) - \
        2 * np.diag(np.ones(Nx-1))
    A = A/dx**2
    if np.abs(alpha-1.0) < 1e-5:
        if eps < 1e-4:
            M = np.eye(Nx-1) - dt * c * A
            inv_M = np.linalg.solve(M, np.eye(Nx-1))
            for n in np.arange(1, Nt+1):
                V = U[1:-1, n:(n+1)] - inv_M @ (U[1:-1, (n-1):n] +
                                                f_fun(X[1:-1].reshape((-1, 1)), n*dt, alpha=alpha_true)*dt)
                nll += V.T @ V
            return nll
        else:
            M = np.eye(Nx-1) - dt * c * A
            L = np.linalg.cholesky(M)
            inv_M = np.linalg.solve(M, np.eye(Nx-1))
            for n in np.arange(1, Nt+1):
                V = U[1:-1, n:(n+1)] - inv_M @ (U[1:-1, (n-1):n] +
                                                f_fun(X[1:-1].reshape((-1, 1)), n*dt, alpha=alpha_true)*dt)
                nll += -2*np.sum(np.log(np.diag(L))) + 0.5 * \
                    np.sum((M @ V)**2) / (Nx*dt * eps**2) \
                  + (Nx-1)/2* (np.log(eps**2*dt*Nx) +np.log(2*np.pi))
            return nll

    else:
        b_coeff = np.zeros(Nt+1)
        b_coeff[0] = 1.0
        for i in range(Nt):
            b_coeff[i+1] = (1-(1-alpha+1)/(i+1)) * b_coeff[i]

        if eps < 1e-4:
            M = np.eye(Nx-1) - dt**alpha * b_coeff[0] * c * A
            inv_M = np.linalg.solve(M, np.eye(Nx-1))
            for n in np.arange(1, Nt+1):
                temp = 0.0
                for j in range(n):
                    temp += b_coeff[n-j] * c * A @ U[1:-1, j:(j+1)]
                V = U[1:-1, n:(n+1)] - inv_M @ (U[1:-1, (n-1):n] + f_fun(
                    X[1:-1].reshape((-1, 1)), n*dt, alpha=alpha_true)*dt + dt**alpha * temp)
                nll += V.T @ V
            return nll

        else:
            M = np.eye(Nx-1) - dt**alpha * b_coeff[0] * c * A
            inv_M = np.linalg.solve(M, np.eye(Nx-1))
            L = np.linalg.cholesky(M)
            for n in np.arange(1, Nt+1):
                temp = 0.0
                for j in range(n):
                    temp += b_coeff[n-j] * c * A @ U[1:-1, j:(j+1)]
                V = U[1:-1, n:(n+1)] - inv_M @ (U[1:-1, (n-1):n] + f_fun(
                    X[1:-1].reshape((-1, 1)), n*dt, alpha=alpha_true)*dt + dt**alpha * temp)
                nll += -2*np.sum(np.log(np.diag(L))) + 0.5 * \
                    np.sum((M @ V)**2) / (Nx*dt * eps**2) \
                     + (Nx-1)/2* (np.log(eps**2*dt*Nx) +np.log(2*np.pi))
            return nll


if __name__ == '__main__':
    
    # alpha_true = 0.1
    alpha_true = 0.5
    # alpha_true = 0.9

    c_true = 1.0
    eps_true = 0.1
    T_final = 1.0
    I = 1
    Nx = 128
    Nt = 128
    X = np.linspace(0, 1, Nx+1)
    dt = T_final/Nt
    U_single_obs = np.loadtxt('U_FDM_alpha_'+str(alpha_true)+'_I_'+str(I)+'_Nx_'+str(Nx)+'_Nt_'+str(Nt)+'_eps_'+str(eps_true)+'_T_final_'+str(T_final)+'.txt')


    rx = 1
    rt = 1
    U = U_single_obs.reshape((Nx+1,Nt+1))
    
    
######### One parameter estimation    

    tt0 = time.time()
    
    paras_est_flag = [False, False, True]   ###  Estiamte the third parameter \epsilon (eps)
    paras_true = [alpha_true, c_true, eps_true]
    
    # temp = minimize(nll_fun, [0.5], method='L-BFGS-B', args = (X[::rx], U[::rx,::rt], int(Nx/rx), int(Nt/rt), paras_est_flag, paras_true), bounds= [(0.01,0.99)])
    temp = minimize(nll_fun, [0.5], method='L-BFGS-B',  args = (X[::rx], U[::rx,::rt], int(Nx/rx), int(Nt/rt), paras_est_flag, paras_true), bounds= [(0.001,1)],options={'ftol': 1e-12, 'gtol': 1e-9},)
    tt1 = time.time()

    print("Estimate one parameter\n")
    print(f"Estimated paras: {temp.x[0]}\n")
    print(f"Negative log-likelihood: {temp.fun}\n")
    print(f"Number of function evaluations: {temp.nfev}\n")
    print(f"CPU time for parameter estimation: {np.round(tt1-tt0,4)} seconds \n") 
    print("############################################")
    
######### Two parameter estimation    

    tt0 = time.time()
    
    paras_est_flag = [True,False,True] ### Estimate the first and third parameters \alpha and \epsilon
    paras_true = [alpha_true, c_true, eps_true]
    
    # temp = minimize(nll_fun, [0.8, 0.5], method='L-BFGS-B', args = (X[::rx], U[::rx,::rt], int(Nx/rx), int(Nt/rt), paras_est_flag, paras_true), bounds= [(0.01,0.99), (0.001,1)])
    temp = minimize(nll_fun, [0.8,0.5], method='L-BFGS-B', args = ( X[::rx], U[::rx,::rt], int(Nx/rx), int(Nt/rt),paras_est_flag, paras_true), bounds= [(0.01,0.99),(0.001,1.0)],options={'ftol': 1e-12, 'gtol': 1e-9},)
    tt1 = time.time()

    print("Estimate one parameter\n")
    print(f"Estimated paras: {temp.x}\n")
    print(f"Negative log-likelihood: {temp.fun}\n")
    print(f"Number of function evaluations: {temp.nfev}\n")
    print(f"CPU time for parameter estimation: {np.round(tt1-tt0,2)} seconds \n")  
    print("############################################")
    
    
######### Three parameter estimation    

    tt0 = time.time()
    
    paras_est_flag = [True,True, True]  ### Estimate all the three parameters
    paras_true = [alpha_true, c_true, eps_true]
    
    # temp = minimize(nll_fun, [0.8,0.5,0.5], method='L-BFGS-B', args = (X[::rx], U[::rx,::rt], int(Nx/rx), int(Nt/rt), paras_est_flag, paras_true), bounds= [(0.01,0.99), (0.01,2), (0.001,1)])
    temp = minimize(nll_fun, [0.8,0.5,0.5], method='L-BFGS-B',  args = (X[::rx], U[::rx,::rt], int(Nx/rx), int(Nt/rt), paras_est_flag, paras_true), bounds= [(0.01,0.99),(0.01,2),(0.001,1.0)],options={'ftol': 1e-12, 'gtol': 1e-9},)
    tt1 = time.time()

    print("Estimate one parameter\n")
    print(f"Estimated paras: alpha = {temp.x}\n")
    print(f"Negative log-likelihood: {temp.fun}\n")
    print(f"Number of function evaluations: {temp.nfev}\n")
    print(f"CPU time for parameter estimation: {np.round(tt1-tt0,4)} seconds \n")     
    print("############################################")
    