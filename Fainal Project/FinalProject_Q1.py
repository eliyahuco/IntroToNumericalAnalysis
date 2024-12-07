"""
Author: Eliyahu Cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Description:

This script is question 1 in the final project for the course intro to numerical analysis

given an acoustic model with a uniform density and a source (that produces a waves) in at x = 3000 meters, z = 2800 meters

the domain is defined as:
0<=x<=6000 meters
0<=z<=6000 meters

there is a layer that defines the speed of the wave:
the layer given by series of points:
(x,z) = (0,2600), (1000,4000), (2600,3200), (4600,3600), (6000,2400)
the speed of the wave depends on the location of the wave with respect to the layer:
above the layer, the speed of the wave is c1 = 2000 m/s, and below the layer, the speed of the wave is c2 = 3000 m/s

the objective of this script is to solve a problem of the wave equation in 2D space:
    ∂^2u/∂t^2 = c^2(∂^2u/∂x^2 + ∂^2u/∂z^2) + F(x,z,t)

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

we will solve the wave equation in the explicit method


missions:

1) To represent the separation layer using cubic spline interpolation.
   *** Ensure that the velocity model is correct before solving the wave equation ***
2) Calculate the progression of the wave field in the medium using 4th-order finite difference for spatial steps and 2nd-order for time.
    *** Space step: Δx = Δz = 100 m, Time steps: Δt = 0.01 s and Δt = 0.03 s ***
    how the solution behaves with the time step Δt?
3) Show snapshots of the wave field at times
for Δt = 0.01 s at t = 0.15 s, 0.4 s, 0.7 s, 1 s
and
for Δt = 0.03 s at t = 0.15 s, 0.3 s, 0.6 s, 0.9 s .
4) Create a complete animation of the wave field for Δt = 0.01 s.

---------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def rk4_method_2nd_order(t0, y0, v0, h, tmax, m=None, k=None):
    t_values = np.arange(t0, tmax + h, h)
    y_values = [y0]
    v_values = [v0]

    y = y0
    v = v0

    for t in t_values[:]:
        f1_y = v
        f1_v = -0.5*v -7 * y

        f2_y = v + h * f1_v / 2
        f2_v = -0.5*(v + h * f1_v / 2) -7 * (y + h * f1_y / 2)

        f3_y = v + h * f2_v / 2
        f3_v = -0.5*(v + h * f2_v / 2) -7 * (y + h * f2_y / 2)

        f4_y = v + h * f3_v
        f4_v = -0.5*(v + h * f3_v) -7 * (y + h * f3_y)

        print('*' * 20, '\n')
        print('for t =', t, ', y =', round(y,6),', v =', round(v,6))
        print('\n','*' * 20, '\n','*'* 20)
        # printing the values of the slopes with round function to 6 decimal points
        print('f1_y = ',round(v,6) ,f'\nf1_v = -0.5*{round(v,6)} -7*{round(y,6)} = ', round(f1_v, 6))
        print(f'f2_y = {round(v,6)} + {h/2}*{round(f1_v,6)} = ', round(f2_y, 6), f'\nf2_v = -0.5*({round(v,6)}+{(h/2)}*{round(f1_v, 6)}) -7*({round(y,6)} + {(h/2)}*{round(f1_y, 6)}) = ', round(f2_v, 6))
        print(f'f3_y = {round(v,6)} + {h/2}*{round(f2_v,6)} = ', round(f3_y, 6), f'\nf3_v = -0.5*({round(v,6)}+{(h/2)}*{round(f2_v, 6)}) -7*({round(y,6)} + {(h/2)}*{round(f2_y, 6)}) = ', round(f3_v, 6))
        print(f'f4_y = {round(v,6)} + {h}*{round(f3_v,6)} = ', round(f4_y, 6), f'\nf4_v = -0.5*({round(v,6)}+{h}*{round(f3_v, 6)}) -7*({round(y,6)} + {h}*{round(f3_y, 6)}) = ', round(f4_v, 6))

        y += h * (f1_y + 2 * f2_y + 2 * f3_y + f4_y) / 6
        v += h * (f1_v + 2 * f2_v + 2 * f3_v + f4_v) / 6

        y_values.append(y)
        v_values.append(v)

    print('\n')
    print('*' * 20, '\n')
    print("summary of the results:")
    print(f'for x =0 to x = 2 with h = {h}')
    print(f'y(0) = {y0}, v(0) = {v0}')
    print(f'y(0.5) = {y_values[1]}, v(0.5) = {v_values[1]}')
    print(f'y(1) = {y_values[2]}, v(1) = {v_values[2]}')
    print(f'y(1.5) = {y_values[3]}, v(1.5) = {v_values[3]}')
    print(f'y(2) = {y_values[4]}, v(2) = {v_values[4]}')


    return t_values, np.array(y_values), np.array(v_values)

rk4_method_2nd_order(0,4,0,0.5,2)


import numpy as np

def rk4_method_2nd_order(x0, y0, z0, h, xmax):
    x_values = np.arange(x0, xmax + h, h)
    y_values = [y0]
    z_values = [z0]

    y = y0
    z = z0

    for x in x_values[:]:
        k1 = z
        l1 = -0.5 * z - 7 * y

        k2 = z + h * l1 / 2
        l2 = -0.5 * (z + h * l1 / 2) - 7 * (y + h * k1 / 2)

        k3 = z + h * l2 / 2
        l3 = -0.5 * (z + h * l2 / 2) - 7 * (y + h * k2 / 2)

        k4 = z + h * l3
        l4 = -0.5 * (z + h * l3) - 7 * (y + h * k3)

        print('*' * 20, '\n')

        print('For x =', x, ', y =', round(y, 6), ', z =', round(z, 6))
        print('\n', '*' * 20, '\n', '*' * 20)
        # Printing the values of the slopes rounded to 6 decimal points
        print('k1 = ', round(k1, 6), f'\nl1 = -0.5*{round(z, 6)} - 7*{round(y, 6)} = ', round(l1, 6))
        print(f'k2 = {round(z, 6)} + {h/2}*{round(l1, 6)} = ', round(k2, 6),
              f'\nl2 = -0.5*({round(z, 6)} + {h/2}*{round(l1, 6)}) - 7*({round(y, 6)} + {h/2}*{round(k1, 6)}) = ',
              round(l2, 6))
        print(f'k3 = {round(z, 6)} + {h/2}*{round(l2, 6)} = ', round(k3, 6),
              f'\nl3 = -0.5*({round(z, 6)} + {h/2}*{round(l2, 6)}) - 7*({round(y, 6)} + {h/2}*{round(k2, 6)}) = ',
              round(l3, 6))
        print(f'k4 = {round(z, 6)} + {h}*{round(l3, 6)} = ', round(k4, 6),
              f'\nl4 = -0.5*({round(z, 6)} + {h}*{round(l3, 6)}) - 7*({round(y, 6)} + {h}*{round(k3, 6)}) = ',
              round(l4, 6))

        y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        z += h * (l1 + 2 * l2 + 2 * l3 + l4) / 6
        print('\n')
        print('-' * 20)
        print(f'y({x+h}) = y({x}) + h * ({round(k1,4 )} + 2 * {round(k2, 4)} + 2 * {round(k3, 4)} + {round(k4, 4)}) / 6 = ', round(y, 6), f'\nz({x+h}) = z({x}) + h * ({round(l1,4 )} + 2 * {round(l2, 4)} + 2 * {round(l3, 4)} + {round(l4, 4)}) / 6 = ', round(z, 6))
        print('-' * 20)
        print('\n')
        y_values.append(y)
        z_values.append(z)

    print('\n')
    print('*' * 20, '\n')
    print("Summary of the results:")
    print(f'For x = 0 to x = 2 with h = {h}')
    print(f'y(0) = {y0}, z(0) = {z0}')
    print(f'y(0.5) = {y_values[1]}, z(0.5) = {z_values[1]}')
    print(f'y(1) = {y_values[2]}, z(1) = {z_values[2]}')
    print(f'y(1.5) = {y_values[3]}, z(1.5) = {z_values[3]}')
    print(f'y(2) = {y_values[4]}, z(2) = {z_values[4]}')

    return x_values, np.array(y_values), np.array(z_values)

# Example usage
rk4_method_2nd_order(0, 4, 0, 0.5, 2)

import numpy as np

# Parameters
k = 0.49  # Thermal conductivity (cal/cm/s/C)
q = 1.0  # Heat flux (cal/cm^2/s)
dx = 1.0  # Grid spacing (cm)
dy = 1.0

# Initialize the grid
T = np.zeros((4, 5))  # 4 rows (y), 5 columns (x)

# Boundary conditions
T[0, :] = 100  # Top boundary
T[:, 0] = 75  # Left boundary
T[:, -1] = 0  # Right boundary initially (updated later with flux)
# Bottom boundary: flux boundary condition will be applied iteratively

# Iterative solver parameters
tolerance = 1e-6  # Convergence tolerance
max_iterations = 1000  # Maximum number of iterations

# Iterate using Gauss-Seidel method
for iteration in range(max_iterations):
    T_old = T.copy()

    # Update interior points
    for i in range(1, 3):  # Only rows 1 and 2 (interior rows)
        for j in range(1, 4):  # Only columns 1 to 3 (interior columns)
            T[i, j] = 0.25 * (T[i + 1, j] + T[i - 1, j] + T[i, j + 1] + T[i, j - 1])

    # Update bottom boundary (flux condition)
    T[3, 1:4] = T[2, 1:4] - q / k  # Bottom boundary

    # Update right boundary (flux condition)
    T[1:3, 4] = T[1:3, 3] - q / k  # Right boundary

    # Check for convergence
    max_diff = np.max(np.abs(T - T_old))
    if max_diff < tolerance:
        break

# Results
print('\n', '*' * 20, '\n')
print(f"Converged after {iteration + 1} iterations")
print("Temperature distribution:")
print(T)
