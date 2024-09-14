"""
Author: Eliyahu Cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is question 1 in the final project for the course intro to numerical analysis

the objective of this script is to solve a problem of the wave equation in 2D space:
    ∂^2u/∂t^2 = c^2(∂^2u/∂x^2 + ∂^2u/∂z^2) + F(t)

        where:
        u(x,z,t) is the wave function
        c is the wave speed
        F(x,z,t) is the source function
        x,z are the spatial coordinates
        t is the time coordinate

        the source function is given as:
        F(t) = t*exp(2*pi*t)*sin(2*pi*t) for  0<=t<=0.05
        F(t) = 0 for t>0.05




---------------------------------------------------------------------------------
"""
