import numpy as np
from math import sqrt, cos, sin, pi

##Globals - Function definition
g = f = np.vectorize(lambda x, y: (x ** 2) + (y ** 2))
dz_f = lambda x, y: [2 * x, 2 * y]
hessian = np.array([[2, 0], [0, 2]])
z_desired = 16
plot_desired = 16


#f = np.vectorize(lambda x, y: sqrt((x ** 2) + (3 * (y ** 2)) + (4 * y) - x - (2 * x * y)))
#g = lambda x, y: (x ** 2) + (3 * (y ** 2)) + (4 * y) - x - (2 * x * y)
#dz_f = lambda x, y: [((2 * x) - 1 - (2 * y))/f(x, y), ((6 * y) + 4 - (2 * x))/f(x,y)]
#hessian = np.array([[2, 0], [0, 0.5]])
#z_desired = 4
#plot_desired = 16

##Parameters - Formation Control
numSensors = 4
a = 0.6
b = 0.6
K2 = 6000
K3 = 20
dt = 0.01

##Parameters - Formation Center
mu_f = 10
K4 = 40

##Utilities: Formation Control
phi = np.array(
    [[1 / 4, 1 / 4, 1 / 4, 1 / 4],
     [-1 / sqrt(2), 1 / sqrt(2), 0, 0],
     [0, 0, 1 / sqrt(2), -1 / sqrt(2)],
     [-1 / 2, -1 / 2, 1 / 2, 1 / 2]
     ])

phi_inv = np.linalg.inv(phi)

##Utilities: Formation Center
rotateRight = np.array([[cos(pi / 2), -sin(pi / 2)],
                        [sin(pi / 2), cos(pi / 2)]])

rotateLeft = np.array([[cos(-pi / 2), -sin(-pi / 2)],
                       [sin(-pi / 2), cos(-pi / 2)]])

##Initialial variables
r_c = np.array([-2, -6])

r = np.array(
    [r_c + [1, 0],
     r_c + [-1, 0],
     r_c + [0, -1],
     r_c + [0, 1]])

# Jacobi vectors - formation control
q = phi @ r
dq = np.zeros((numSensors, 2))
u_r = np.zeros((numSensors, 2))
vel_q = np.zeros((numSensors, 2))

print("This is a test")
