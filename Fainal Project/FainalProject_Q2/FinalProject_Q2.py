"""
Author: Eliyahu Cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Description:

This script is question 2 in the final project for the course intro to numerical analysis
in Tel Aviv University.

The script solves an non-homogeneous Heat Equation with implicit finite difference method.

de domain is a square with the following boundaries:
    x = 0, x = 1.5, y = 0, y = 1.5
    origin at (0,0) is at the left upper corner, while the positive x-axis is to the right and the positive y-axis is down

the equation is:
    dT/dt = κ *(d^2u/dx^2 + d^2u/dy^2) + f(x,y,t)

    where f(x,y,t) = -10^4*exp(-((x-1)^2/2*σ_x^2)*exp(-((y-0.5)^2/2*σ_y^2)*exp(-0.1*t)

    while σ_x = σa_y = 0.00625 meters

    initial condition:
             κ = 1.786*10^-3 m^2/s
             T(x,y,0) = 10 [C]

    boundary conditions (for the first 60 seconds):
            T(X,0,t) = 100 [C]
            T(X,1.5,t) = 10 [C]
            T(1.5,Y,t) = 100 -60y [C]

                        { 100 -112.5y [C] if 0<= y <= 0.8
            T(0,Y,t) = <
                        { 10 [C] if 0.8 < y <= 1.5


    we will solve the wave equation in the implicit method, using the ADI method.
    we will use Finite Difference 2nd order for the spatial derivatives, and 1st order for the time derivative.
    we will use the following discretization:
        ∂^2T/∂x^2 ≈ (T(i+1,j) - 2T(i,j) + T(i-1,j))/Δx^2
        ∂^2T/∂y^2 ≈ (T(i,j+1) - 2T(i,j) + T(i,j-1))/Δy^2
        ∂T/∂t ≈ (T(i,j,n+1) - T(i,j,n))/dt
        where Δx = Δy = 0.05 [m], Δt = 0.1 [s]

missions:

1. solve the equation for 0<= t <= 60 [s]
2. Show snapshots of the temperature distribution at t = 15, 30, 60 [s]
3. 4) Create a complete animation of the temperature distribution for 0<= t <= 60 [s] while we show for every 10 time-steps.

---------------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Constants
κ = 1.786*10**-3
σ_x = 0.00625
σ_y = 0.00625
Δx = 0.05
Δy = 0.05
Δt = 0.1
T0 = 10
Lx = 1.5
Ly = 1.5
Nx = int(Lx/Δx)
Ny = int(Ly/Δy)
Nt = 601
T = np.zeros((Nx,Ny,Nt))

# Initial condition
T[:,:,0] = T0

# Boundary conditions
T[:,0,:] = 100
T[:,-1,:] = 10
T[0,:int(0.8*Ny),:] = (100 - 112.5*np.linspace(0, 0.8, int(0.8*Ny))).reshape(int(0.8*Ny), 1)
T[0,:int(0.8*Ny),:] = (100 - 112.5*np.linspace(0, 0.8, int(0.8*Ny))).reshape(int(0.8*Ny), 1)
T[0,int(0.8*Ny):,:] = 10

f(x,y,t) = lambda x,y,t: -10**4*np.exp(-((x-1)**2/(2*σ_x**2)))*np.exp(-((y-0.5)**2/(2*σ_y**2)))*np.exp(-0.1*t)