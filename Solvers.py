# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 09:20:02 2021

@author: pangg
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import gamma
import datetime

"""
Solve 1D stochastic time-fractional diffusion equation with homogeneous Dirichlet
BCs

\frac{\partial u(x,t, \omega)}{\partial t} 
= D^{1-\alpha}\Delta u(x, t,\omega) + f(x, t) + \epsilon\sum_{k=1}^M \phi_k(x) Z_k(t)/dt

where x, t \in (0,1)\times (0,T_final], \alpha \in (0,1), {Z_k(t)} is a sequence of independent Wiener processes,
and \phi_k(x) = \sqrt(2)\sin(k\pi x) is normalized eigen-function of $-\Delta$.

The finite difference method was proposed in 
"MAX GUNZBURGER, BUYANG LI, AND JILU WANG
Abstract.Sharp convergence rates of time discretization for
 stochastic time-fractional PDEs subject to additive space-time white noise, 2010"

"""


def FDM_time__stochastic_fractional_diffusion(Nx, Nt, T_final, I, alpha, c, eps):
   
    
    
    """

    Parameters
    ----------
    Nx : Int
        Number of space grid points; space grid points:  0, 1/Nx, 2/Nx, ..., 1.
    Nt : Int
        Number of time grid points; Time grid points: 0, T_final/Nt, 2T_final/Nt, ..., T_final
    T_final : Float
        Final time for evaluation or observation
    I : Int
        Number of realizations of white noise
    alpha : float
        Order of fractional-time derivative
    c : Float
        Diffusion coefficient
    eps : Float
        Magnitude of white noise

    Returns
    -------
    3d array
       a collection of stochastic solutions for I different realizations of white noise.
       The first axis of the 3d array represents the index of realization;
       the second axis of the array represents the space direction;
       the thrid axis of the array represents the time direction.

    """
    
    def f_fun(x,t,alpha):
        return 2*t*x**2*(1-x)**2 - 2* t**(1+alpha)/gamma(2+alpha) * (2-12*x+12*x**2)    
    
    
    # def u_true(x,t):
    #     return t**2*x**2*(1-x)**2


    def g_noise(x, M, I, Nt, dt):
        Phi = np.sqrt(2.0)*np.sin(x[1:-1].reshape((-1,1)) * np.linspace(1,M,M).reshape((1,-1))*np.pi) 
        np.random.seed(10000) ### random seed is fixed for sake of replication
        W = np.random.randn(I, M,Nt)*np.sqrt(dt)
        noise = np.matmul(Phi, W/dt)
        return np.transpose(noise,(0,2,1)).reshape((I,-1))


    tt0 = time.time()
    
    X = np.linspace(0, 1, Nx+1)

    dt = T_final/Nt
    dx = 1/Nx
    
    A = np.diag(np.ones(Nx-2),-1) + np.diag(np.ones(Nx-2),1) - 2* np.diag(np.ones(Nx-1))
    A = A/dx**2
    
    
    noise = eps * g_noise(X, Nx, I, Nt, dt)   
        
        
    U = np.zeros((I, Nx+1, Nt+1))
    
    if np.abs(alpha - 1.0)<1e-4:
        M = np.eye(Nx-1) - dt  * c * A  
        inv_M = np.linalg.solve(M, np.eye(Nx-1))       
        
        for i in range(I):
            for n in np.arange(1,Nt+1):               
                U[i, 1:-1,n:(n+1)] = inv_M @ (U[i, 1:-1,(n-1):n] + f_fun(X[1:-1].reshape((-1,1)),n*dt,alpha)*dt  + noise[i,:].reshape((Nt,Nx-1)).T[:,(n-1):n] * dt)
                    
        
    else:        
        
        b_coeff = np.zeros(Nt+1)
        b_coeff[0] = 1.0
        for i in range(Nt):
            b_coeff[i+1] = (1-(1-alpha+1)/(i+1)) * b_coeff[i] 
        M = np.eye(Nx-1) - dt**alpha * b_coeff[0] * c * A
        
        inv_M = np.linalg.solve(M, np.eye(Nx-1))
        
        U = np.zeros((I, Nx+1, Nt+1))
        
        for i in range(I):
            for n in np.arange(1,Nt+1):
                temp = 0.0
                for j in range(n):
                    temp += b_coeff[n-j] * c * A @ U[i, 1:-1,j:(j+1)]
                temp *= dt**alpha
                U[i, 1:-1,n:(n+1)] = inv_M @ (U[i, 1:-1,(n-1):n] + f_fun(X[1:-1].reshape((-1,1)),n*dt,alpha)*dt + temp + noise[i,:].reshape((Nt,Nx-1)).T[:,(n-1):n] * dt)
            

    
    
    
    np.savetxt('U_FDM_alpha_'+str(alpha)+'_I_'+str(I)+'_Nx_'+str(Nx)+'_Nt_'+str(Nt)+'_eps_'+str(eps)+'_T_final_'+str(T_final)+'.txt', U.reshape((I,-1)))
    
    tt1 = time.time()
    
    with open('Paras_I_'+str(I)+'_Nx_'+str(Nx)+'_Nt_'+str(Nt)+'_eps_'+str(eps)+'.txt', 'a') as f:
        f.write('alpha= '+str(alpha) + ' c= ' + str(c) +', eps= ' + str(eps) + '\n')
        f.write('CPU time: ' + str(tt1-tt0))
    return U    



if __name__ == '__main__':
    
    
    print(f"Code starts at {str(datetime.datetime.now())}\n")
    alpha_true = 0.5
    c_true = 1.0
    I = 1
    Nx = 128
    Nt = 128
    T_final = 1.0
    eps_true = 0.1
    dt = T_final/Nt
    M = Nx
    X = np.linspace(0, 1, Nx+1)
    
   
    
    
    tt0 = time.time() 
    U_obs = FDM_time__stochastic_fractional_diffusion(Nx, Nt, T_final, I, alpha_true,c_true, eps_true)
    tt1 = time.time()
    print(f"Nx = {Nx}, Nt={Nt}, I={I}\n")
    print(f"alpha={alpha_true}, c={c_true}, eps={eps_true}, T_final={T_final}\n")
    print(f"CPU time for one run: {tt1-tt0} seconds\n")
    
    
    print(f"Code ends at {str(datetime.datetime.now())}\n")
    
    
    
    
    
  
    
    
    
    
    