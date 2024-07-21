"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the Question 1 in HW_5 for the course intro to numerical analysis

the objective of this script is to solve the following ODE:

dm/dx = s

    where m is the momentum, s is the shear force and x is the position
    s = 10 - 2x
    m(x=0) = 0
    s(x=0) = 10
    x_max = 10 meters

we will solve the ODE using the following methods:
1) Euler's method with step size of: 0.05 meters, 0.25 meters
2) Runge-Kutta second order method (RK2)

we will compare the results of the methods and with the analytical solution
also we will plot the results

will use the file numerical_analysis_methods_tools.py for use functions from the previous assignments
---------------------------------------------------------------------------------
"""

# Libraries in use
import numpy as np
import scipy.special as sp
from scipy import integrate
import matplotlib.pyplot as plt
import numerical_analysis_methods_tools as na_tools

# Given parameters
