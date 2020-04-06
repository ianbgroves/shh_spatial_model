# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:10:22 2020

@author: Ian_b
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:40:24 2020

@author: Ian Groves, Alexander Fletcher

Finite difference solver for a non-dimensionalised form of the Schnakenberg 
reaction-diffusion PDE system, which exhibits a Turing instability.

This code uses a Crank-Nicolson scheme to solve the PDE system. 

The PDEs are: ##change these to incorporate something similar to Menshykau et al. plos comp biol

    u_t = d_u*u_xx + gamma*(a - u + u*u*v),
    v_t = d_v*v_xx + gamma*(b - u*u*v).
    
We solve this system on a 1D spatial domain 0 < x < L with no-flux boundary 
conditions, using initial conditions that are a small random perturbation 
away from the spatially uniform steady state (u, v) = (a+b, b/((a+b)**2).

For the chosen parameter values, the system evolves to a stable spatially 
non-uniform steady state (i.e. a pattern).
    


##                                          
"""

import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt

# Parameter values
a = 0.2
b = 2.0
gamma = 0.2
d_s = 1
d_f = 10
L = 1

# Reaction terms
def reaction_term_s(s, f):
    return gamma*(a - s + s*s*f)

def reaction_term_f(s, f):
    return gamma*(b - s*s*f)
    
def pde_solver_with_no_flux_bcs(Nx, T):
    x = np.linspace(0, L, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    nu = 0.5                      # nu = D*dt/(dx**2) <= 0.5 (to avoid spurious oscillations)
    dt = nu*dx*dx/max(d_s, d_f)
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, T, Nt+1)   # mesh points in time

    s = np.zeros(Nx+1)            # solution array for u at t[n+1]
    f = np.zeros(Nx+1)            # solution array for v at t[n+1]
    
    s_old = np.zeros(Nx+1)        # solution array for u at t[n]
    f_old = np.zeros(Nx+1)        # solution array for v at t[n]

    # Precompute sparse matrices (scipy format)
    diagonal = (1 + d_s*dt/(dx**2))*np.ones(Nx+1)
    upper = -0.5*(d_s*dt/(dx**2))*np.ones(Nx)
    upper[0] = -d_s*dt/(dx**2)
    lower = -0.5*(d_s*dt/(dx**2))*np.ones(Nx)
    lower[-1] = -d_s*dt/(dx**2)
    s_matrix_lhs = scipy.sparse.diags(diagonals=[diagonal, lower, upper],
                           offsets=[0, -1, 1], shape=(Nx+1, Nx+1),
                           format='csr')
    
    diagonal = (1 - d_s*dt/(dx**2))*np.ones(Nx+1)
    upper = -upper
    lower = -lower
    s_matrix_rhs = scipy.sparse.diags(diagonals=[diagonal, lower, upper],
                           offsets=[0, -1, 1], shape=(Nx+1, Nx+1),
                           format='csr')

    diagonal = (1 + d_f*dt/(dx**2))*np.ones(Nx+1)
    upper = -0.5*(d_f*dt/(dx**2))*np.ones(Nx)
    upper[0] = -d_f*dt/(dx**2)
    lower = -0.5*(d_f*dt/(dx**2))*np.ones(Nx)
    lower[-1] = -d_f*dt/(dx**2)
    f_matrix_lhs = scipy.sparse.diags(diagonals=[diagonal, lower, upper],
                           offsets=[0, -1, 1], shape=(Nx+1, Nx+1),
                           format='csr')
    
    diagonal = (1 - d_f*dt/(dx**2))*np.ones(Nx+1)
    upper = -upper
    lower = -lower
    f_matrix_rhs = scipy.sparse.diags(diagonals=[diagonal, lower, upper],
                           offsets=[0, -1, 1], shape=(Nx+1, Nx+1),
                           format='csr')
        
    # Representation of right-hand sides
    rhs_s = np.zeros(Nx+1)
    rhs_f = np.zeros(Nx+1)
    
    # Set initial condition
    s_old = (a + b)*np.ones(Nx+1) + 0.01*np.random.rand(Nx+1)
    f_old = (b/((a+b)**2))*np.zeros(Nx+1) + 0.01*np.random.rand(Nx+1)

    # Time loop
    for n in range(0, Nt):
        rhs_s = s_matrix_rhs*s_old + dt*reaction_term_s(s_old, f_old)
        rhs_f = f_matrix_rhs*f_old + dt*reaction_term_f(s_old, f_old)
        
        s = scipy.sparse.linalg.spsolve(s_matrix_lhs, rhs_s)
        f = scipy.sparse.linalg.spsolve(f_matrix_lhs, rhs_f)
        
        s_old, s = s, s_old
        f_old, f = f, f_old

    return s, f, x, t


s, f, x, t = pde_solver_with_no_flux_bcs(Nx=50, T=1.0)


# Call the PDE solver and plot u(x,1.0) and v(x,1.0)
plt.figure(0)
plt.plot(x, s)
plt.xlabel('$x$')
plt.ylabel('$s(x,t)$')

plt.figure(1)
plt.plot(x, f)
plt.xlabel('$x$')
plt.ylabel('$f(x,t)$')