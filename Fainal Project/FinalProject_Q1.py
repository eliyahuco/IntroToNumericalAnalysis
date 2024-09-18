"""
Author: Eliyahu Cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Description:

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
            the source location is at x=3000 meters, z=2800 meters

there is a layer that defines the speed of the wave, the layer given by series of points: (x,z) = (0,2600), (1000,4000), (2600,3200), (4600,3600), (6000,2400)
above the layer, the speed of the wave is c1 = 2000 m/s, and below the layer, the speed of the wave is c2 = 3000 m/s
we will find the layer using cubic spline interpolation
then we will solve the wave equation using the finite difference 4th order in the spatial domain and 2nd order in the time domain
the spatial step dx = dz = 100 meters
the time step dt = 0.01 seconds, 0.03 seconds
we will show the wave field (snapshot):
for dt = 0.01 seconds at t = 0.15,0.4,0.7,1.0 seconds
for dt = 0.03 seconds at t = 0.15,0.3,0.6,0.9 seconds

at the end we will plot animation of the wave field with dt = 0.01 seconds

we will solve the wave equation in the explicit method

---------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numerical_analysis_methods_tools as na_tools

x_values = np.array([0, 1000, 2600, 4600, 6000])
z_values = np.array([2600, 4000, 3200, 3600, 2400])
n_splines = len(x_values)

spline_coefficients = na_tools.natural_cubic_spline(x_values, z_values)

x_segment = np.arange(0, 6000, 100)
z_segment = [na_tools.evaluate_spline(x, x_values, spline_coefficients) for x in x_segment]
#plot the cubic spline interpolation and the points
# add to the plot the position of the source
plt.plot(x_values, z_values, 'ro', label='Layer Points')
plt.plot(x_segment, z_segment, label='Cubic Spline Interpolation')
plt.xlabel('x [m]')
plt.ylabel('z [m]')
plt.title('Layer of the Wave Speed')
plt.legend()
plt.grid()
plt.show()
