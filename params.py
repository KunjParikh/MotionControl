import numpy as np
from math import sqrt, cos, sin, pi

class Params:
  def __init__(self):
    ##Globals - Function definition

    # self.f_list = [
    #   np.vectorize(lambda x, y: (x ** 4) + (y ** 4)),
    #   np.vectorize(lambda x, y: sqrt((x ** 2) + (3 * (y ** 2)) + (4 * y) - x - (2 * x * y))),
    # ]

    # self.dz_f_list = [
    #   lambda x, y: [4 * (x ** 3), 4 * (y ** 3)],
    #   lambda x, y: [((2 * x) - 1 - (2 * y))/self.f(x, y), ((6 * y) + 4 - (2 * x))/self.f(x,y)]
    # ]

    # self.hessian_list = [
    #   np.array([[2, 0], [0, 2]]),
    #   np.array([[2, 0], [0, 0.5]])        
    # ]

    # self.z_desired_list = [
    #   32,
    #   4
    # ]

    # self.g_list = [
    #   np.vectorize(lambda x, y: (x ** 4) + (y ** 4)),
    #   lambda x, y: (x ** 2) + (3 * (y ** 2)) + (4 * y) - x - (2 * x * y)
    # ]

    # self.plot_desired_list = [
    #   32,
    #   16
    # ]

    # self.function_name = [
    #   "circle_square",
    #   "elipse"                     
    # ]

    # self.g_list = self.f_list = [np.vectorize(lambda x, y: (x ** 2) + (y ** 2))]
    # self.dz_f_list = [lambda x, y: [2 * (x ** 1), 2 * (y ** 1)]]
    # self.hessian_list = [lambda x, y: [[2 , 0], [0, 2]]]
    # self.z_desired_list = [16]
    # self.plot_desired_list = [16]

    # self.g_list = self.f_list = [np.vectorize(lambda x, y: (x ** 4) + (y ** 4))]
    # self.dz_f_list = [lambda x, y: [4 * (x ** 3), 4 * (y ** 3)]]
    # self.hessian_list = [lambda x, y: [[12 * (x ** 2), 0], [0, 12 * (y ** 2)]]]
    # self.z_desired_list = [32]
    # self.plot_desired_list = [32]

    self.f_list = [np.vectorize(lambda x, y: sqrt((x ** 2) + (3 * (y ** 2)) + (4 * y) - x - (2 * x * y)))]
    self.g_list = [lambda x, y: (x ** 2) + (3 * (y ** 2)) + (4 * y) - x - (2 * x * y)]
    self.dz_f_list = [lambda x, y: [((2 * x) - 1 - (2 * y))/sqrt((x ** 2) + (3 * (y ** 2)) + (4 * y) - x - (2 * x * y)), ((6 * y) + 4 - (2 * x))/sqrt((x ** 2) + (3 * (y ** 2)) + (4 * y) - x - (2 * x * y))]]
    self.hessian_list = [lambda x, y: np.array([[2, 0], [0, 0.5]])]
    self.z_desired_list = [4]
    self.plot_desired_list = [16]

    # self.f_list = [np.vectorize(lambda x, y: (x ** 2) + (3 * (y ** 2)) + (4 * y) - x - (2 * x * y))]
    # self.g_list = [lambda x, y: (x ** 2) + (3 * (y ** 2)) + (4 * y) - x - (2 * x * y)]
    # self.dz_f_list = [lambda x, y: [((2 * x) - 1 - (2 * y)), ((6 * y) + 4 - (2 * x))]]
    # self.hessian_list = [lambda x, y: np.array([[2, 0], [0, 0.5]])]
    # self.z_desired_list = [16]
    # self.plot_desired_list = [16]

    # self.g_list = self.f_list = [np.vectorize(lambda x, y: (x ** 6) + (y ** 6))]
    # self.dz_f_list = [lambda x, y: [6 * (x ** 5), 6 * (y ** 5)]]
    # self.hessian_list = [lambda x, y: [[30 * (x ** 4), 0], [0, 30 * (y ** 4)]]]
    # self.z_desired_list = [100]
    # self.plot_desired_list = [100]

    ##Parameters - Formation Control
    self.numSensors = 4
    self.a = 0.6
    self.b = 0.6
    self.K2 = 6000
    self.K3 = 20
    self.dt = 0.01

    ##Parameters - Formation Center
    self.mu_f = 10
    self.K4 = 40

    ##Utilities: Formation Control
    self.phi = np.array(
        [[1 / 4, 1 / 4, 1 / 4, 1 / 4],
        [-1 / sqrt(2), 1 / sqrt(2), 0, 0],
        [0, 0, 1 / sqrt(2), -1 / sqrt(2)],
        [-1 / 2, -1 / 2, 1 / 2, 1 / 2]
        ])

    self.phi_inv = np.linalg.inv(self.phi)

    ##Utilities: Formation Center
    self.rotateRight = np.array([[cos(pi / 2), -sin(pi / 2)],
                            [sin(pi / 2), cos(pi / 2)]])

    self.rotateLeft = np.array([[cos(-pi / 2), -sin(-pi / 2)],
                          [sin(-pi / 2), cos(-pi / 2)]])

    ##Initialial variables
    self.r_c = np.array([-2, -6])

    self.r = np.array(
        [self.r_c + [1, 0],
        self.r_c + [-1, 0],
        self.r_c + [0, -1],
        self.r_c + [0, 1]])

    # Jacobi vectors - formation control
    self.q = self.phi @ self.r
    self.dq = np.zeros((self.numSensors, 2))
    self.u_r = np.zeros((self.numSensors, 2))
    self.vel_q = np.zeros((self.numSensors, 2))


