import numpy as np
from math import cos, sin, pi
import random


class Function:
    def __init__(self, name, f, dz_f, hessian_f, z_desired, mu_f):
        self.name = name
        self.f = f  # np.vectorize(lambda x, y: (x ** 2) + (y ** 2))
        self.dz_f = dz_f  # lambda x, y: [2 * (x ** 1), 2 * (y ** 1)]
        self.hessian_f = hessian_f  # lambda x, y: [[2 , 0], [0, 2]]
        self.z_desired = z_desired  # 16
        self.mu_f = mu_f  # 10

class FunctionGenerator:
    def generate(self, num):
        flist = [*2*['circle'], 'circle_4', 'circle_6', *3*['elipse'], *2*['irregular1'], 'irregular2']
        randlist = [p for p in range(2, 11)]
        # rand = random.random() * 10

        functions = []
        functionsUsed = []

        while len(functions) < num:
            type = random.choice(flist)
            rand = random.choice(randlist)
            method = getattr(self, type)
            if "{}_{}".format(type, rand) not in functionsUsed:
                functions.append(method(rand))
                functionsUsed.append("{}_{}".format(type, rand))

        return functions

    def getFunction(self, name):
        nameParts = name.split("_")
        type = '_'.join(nameParts[:-1])
        rand = float(nameParts[-1])
        method = getattr(self, type)
        return method(rand)

    def circle(self, rand):
        return Function("circle_{}".format(rand),
                     np.vectorize(lambda x, y: rand * (x ** 2) + (y ** 2)),
                     lambda x, y: [2 * rand * (x ** 1), 2 * (y ** 1)],
                     lambda x, y: [[2 * rand, 0], [0, 2]],
                     16,
                     10
                     )

    def circle_4(self, rand):
        return Function("circle_4_{}".format(rand),
                     np.vectorize(lambda x, y: (x ** 4) + rand * (y ** 4)),
                     lambda x, y: [4 * (x ** 3), 4 * rand * (y ** 3)],
                     lambda x, y: [[12 * (x ** 2), 0], [0, 12 * rand * (y ** 2)]],
                     32,
                     0.1
                     )

    def circle_6(self, rand):
        return Function("circle_6_{}".format(rand),
                     np.vectorize(lambda x, y: rand * (x ** 6) + (y ** 6)),
                     lambda x, y: [6 * rand * (x ** 5), 6 * (y ** 5)],
                     lambda x, y: [[30 * rand * (x ** 4), 0], [0, 30 * (y ** 4)]],
                     64,
                     0.01
                     )

    def elipse(self, rand):
        return Function("elipse_{}".format(rand),
                     np.vectorize(lambda x, y: rand * (x ** 2) + (3 * (y ** 2)) + (4 * y) - x - (2 * x * y)),
                     lambda x, y: [(2 * rand * x) - 1 - (2 * y), (6 * y) + 4 - (2 * x)],
                     lambda x, y: np.array([[2 * rand, -2], [-2, 6]]),
                     16,
                     10)

    def irregular1(self, rand):
        return Function("irregular1_{}".format(rand),
                        np.vectorize(lambda x, y: (x ** 2 + rand * y - 11) ** 2 + (x + y ** 2 - 7) ** 2),
                        lambda x, y: [2 * (x ** 2 + rand * y - 11) * 2 * x + 2 * (x + y ** 2 - 7),
                                      2 * rand * (x ** 2 + rand * y - 11) + 2 * (x + y ** 2 - 7) * 2 * y],
                        lambda x, y: np.array([[12 * (x ** 2) + 4 * rand * y - 42, 4 * rand * x + 4 * y],
                                               [4 * rand * x + 4 * y, 4 * x + (2 * rand**2 - 28) + 12 * (y ** 2)]]),
                        144,
                        0.1
                     )

    def irregular2(self, rand):
        return Function("irregular2_{}".format(rand),
                        np.vectorize(lambda x, y: (x ** 2 + 2 * y - 12) ** 2 + (rand * x + y ** 2 - 17) ** 2),
                        lambda x, y: [(4 * x ** 3 + (2 * rand ** 2 - 48 ) * x + 8 * x * y + 2 * rand * y ** 2 - 34 * rand),
                                      (4 * x ** 2 + 4 * rand * x * y - 60 * y + 4 * y ** 3 - 48)],
                        lambda x, y: np.array([[12 * (x ** 2) + 8 * y + (2 * rand ** 2 - 48), 8 * x + 4 * rand * y],
                                               [8 * x + 4 * rand * y, 4 * rand * x + 12 * (y ** 2) - 60]]),
                        400,
                        0.1
                     )

    def rhombus(self, rand):
        return Function("rhombus_{}".format(rand),
                        np.vectorize(lambda x, y: abs(x) + rand * abs(y)),
                        lambda x, y: [np.sign(x), rand * np.sign(y)],  # Square cone
                        lambda x, y: [[1, 0], [0, rand]],
                        6,
                        100
                        )

class Params:
    def __init__(self):
        # Globals - Function definition

        fg = FunctionGenerator()
        self.functions = [
            fg.circle(1),
            fg.circle_4(1),
            fg.circle_6(1),
            fg.elipse(1),
            fg.irregular1(1),
            fg.irregular2(1)
            # fg.rhombus(1)
        ]
        # self.functions.extend(fg.generate(20))

        # Parameters - Formation Control
        self.numSensors = 2 # 4
        self.a = 0.6
        self.b = 0.6
        self.K2 = 6000
        self.K3 = 20
        self.dt = 0.01

        # Parameters - Formation Center
        # self.mu_f = 10  #Convereted to mu_f_list because different for shapes
        self.K4 = 40

        # Utilities: Formation Center
        self.rotateRight = np.array([[cos(pi / 2), -sin(pi / 2)],
                                     [sin(pi / 2), cos(pi / 2)]])

        self.rotateLeft = np.array([[cos(-pi / 2), -sin(-pi / 2)],
                                    [sin(-pi / 2), cos(-pi / 2)]])

        # Initial variables
        self.r_c = np.array([-2, -6])

        # Utilities: Formation Control
        if self.numSensors == 4:
            self.phi = np.array(
                [[1 / 4, 1 / 4, 1 / 4, 1 / 4],
                 [-1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0],
                 [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
                 [-1 / 2, -1 / 2, 1 / 2, 1 / 2]
                 ])

            self.phi_inv = np.linalg.inv(self.phi)

            self.r = np.array(
                [self.r_c + [1, 0],
                 self.r_c + [-1, 0],
                 self.r_c + [0, -1],
                 self.r_c + [0, 1]])

        if self.numSensors == 2:
            self.phi = np.array(
                [[1 / 2, 1 / 2],
                 [1 / np.sqrt(2), - 1 / np.sqrt(2)],
                 ])

            self.phi_inv = np.linalg.inv(self.phi)

            self.r = np.array(
                [self.r_c + [1, 0],
                 self.r_c + [-1, 0]])


        # Jacobi vectors - formation control
        self.q = self.phi @ self.r
        self.dq = np.zeros((self.numSensors, 2))
        self.u_r = np.zeros((self.numSensors, 2))
        self.vel_q = np.zeros((self.numSensors, 2))

        #LSTM training
        self.batch_size = 25
