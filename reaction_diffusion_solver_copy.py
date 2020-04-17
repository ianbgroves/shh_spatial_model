# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:10:22 2020

@author: Ian_b
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:40:24 2020

@author: Ian Groves, Alexander Fletcher

Finite difference solver for a non-dimensionalised form of some equations from Menshykau et al. 2012 Plos Comp Biol paper, which is a Schnakenberg type 
reaction-diffusion PDE system, which exhibits a Turing instability. Currently, this implementation is just 2/3 of the compartments of their model. (Ptch not modelled)

This code uses a Crank-Nicolson scheme to solve the PDE system. 

The PDEs are: ##change these to incorporate something similar to Menshykau et al. plos comp biol

    s_t = d_s*s_xx + gamma*(a - s + s*s*f),
    f_t = d_f*f_xx + gamma*(b - s*s*f).
    
We solve this system on a 1D spatial domain 0 < x < L with no-flux boundary 
conditions, using initial conditions that are a small random perturbation 
away from the spatially uniform steady state (s, f) = (a+b, b/((a+b)**2).

For the chosen parameter values, the system evolves to a stable spatially 
non-uniform steady state (i.e. a pattern).
    


##                                          
"""

import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt


#parameters in a simple model of cross repression
##update documentation to report what eqs were solving.
rhoS = 1
rhoF = 2
n = 1
degs = 1
degf = 1
d_s = 1
d_f = 1
L = 1



# Reaction terms
def reaction_term_s(s, f):
    return rhoS/(f**n+1) - degs*s

def reaction_term_f(s, f):
    return rhoF/(s**n + 1) - degf*f
    
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
    s_old = (1+np.tanh((0.25-x)*10))*3 
    f_old = 1+np.tanh((0.25-x)*10) 

    # Time loop
    for n in range(0, Nt):
        rhs_s = s_matrix_rhs*s_old + dt*reaction_term_s(s_old, f_old)
        rhs_f = f_matrix_rhs*f_old + dt*reaction_term_f(s_old, f_old)
        
        s = scipy.sparse.linalg.spsolve(s_matrix_lhs, rhs_s)
        f = scipy.sparse.linalg.spsolve(f_matrix_lhs, rhs_f)
        
        s_old, s = s, s_old
        f_old, f = f, f_old

    return s, f, x, t




# Timecourse = 1.0
# s, f, x, t = pde_solver_with_no_flux_bcs(Nx=50, T=Timecourse)

# # Call the PDE solver and plot u(x,1.0) and v(x,1.0)

# fig, ax = plt.subplots()
# plt.title("Timecourse solution for S with T = {}".format(Timecourse))
# ax.plot(x, s)
# ax.yaxis.offsetText.set_visible(False)
# plt.xlabel('$x$')
# plt.ylabel('$s(x,t)$')

# # plt.figure()
# # plt.title("Timecourse solution for F with T = {}".format(Timecourse))
# # plt.plot(x, f)
# # plt.xlabel('$x$')
# # plt.ylabel('$f(x,t)$')

# plt.show()


T_array = np.array([1.,5.,10.,20.])

for T in T_array:
    s, f, x, t = pde_solver_with_no_flux_bcs(Nx=50, T=T)
    
    # Call the PDE solver and plot u(x,1.0) and v(x,1.0)
   
    fig, ax = plt.subplots()
    plt.title("Timecourse solution for S with T = {}".format(T))
    ax.plot(x, s)
    ax.yaxis.offsetText.set_visible(False)
    plt.xlabel('$x$')
    plt.ylabel('$s(x,t)$')
    #plt.savefig('Timecourse_soln_for_s_t_{}.png'.format(T))

    fig,ax = plt.subplots()
    plt.title("Timecourse solution for F with T = {}".format(T))
    ax.plot(x, f)
    ax.yaxis.offsetText.set_visible(False)
    plt.xlabel('$x$')
    plt.ylabel('$f(x,t)$')
    #plt.savefig('Timecourse_soln_for_f_t_{}.png'.format(T))
    
plt.show()