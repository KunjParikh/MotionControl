import numpy as np
from numpy.linalg import norm
from math import cos, sin, atan2
import matplotlib.pyplot as plt
# from kalmanFilter import kalmanFilter
import params
import utils as utils
pltVar = utils.PlotVariable("testVar")

def formationCenter(r_c, z_c, dz_c, hessian, x_2, y_2, i, rotateRight, rotateLeft, mu_f, z_desired, K4, dt):
    # hessian = np.array([[2, 0],[2, 0]]) * np.random.rand(1)
    # z_c, dz_c, p = kalmanFilter(z_c, dz_c, r, z_r, r_c, r_c_old, p, hessian, numSensors)

    y_1 = dz_c / norm(dz_c)
    x_1 = rotateRight @ y_1

    if i == 0:
        x_2 = x_1
        y_2 = y_1

    theta = atan2(x_2[1], x_2[0]) - atan2(x_1[1], x_1[0])
    kappa_1 = (x_1.T @ hessian @ x_1) / norm(dz_c)
    kappa_2 = (x_1.T @ hessian @ y_1) / norm(dz_c)
    f_z = mu_f * (1 - (z_desired / z_c) ** 2)
    u_c = kappa_1 * cos(theta) + \
          kappa_2 * sin(theta) - \
          (2 * f_z * norm(dz_c) * (cos(theta / 2) ** 2)) + \
          K4 * sin(theta / 2)
    # print([x_2, y_2])

    x_2 = x_2 + dt * u_c * y_2
    x_2 = x_2 / norm(x_2)
    y_2 = rotateLeft @ x_2
    pltVar.push(u_c)
    r_c = r_c + dt * x_2
    return r_c, x_2, y_2, hessian


if __name__ == "__main__":
    params = params.Params()
    for function in params.test_functions:
        r_c, r_c_old, x_2, y_2 = params.r_c, params.r_c, 0, 0
        r_c_plot = [r_c]
        for i in range(10000):
            # Decoupled: Ideal test.
            z_c = function.f(r_c[0], r_c[1])
            dz_c = function.dz_f(r_c[0], r_c[1])
            hessian = function.hessian_f(r_c[0], r_c[1])
            # hessian = [[0, 0], [0, 0]]
            r_c, x_2, y_2, tmp = formationCenter(r_c, z_c, dz_c, hessian, x_2, y_2, i,
                                                 params.rotateRight, params.rotateLeft,
                                                 function.mu_f, function.z_desired,
                                                 params.K4, params.dt)
            r_c_plot.append(r_c)

        x = np.linspace(-10, 10, 200)
        y = np.linspace(-10, 10, 200)
        X, Y = np.meshgrid(x, y)
        Z = function.f(X, Y)

        plt.contour(x, y, Z, [function.z_desired])
        plt.plot(*zip(*r_c_plot), 'b')
        plt.title(function.name)
        plt.show()
    pltVar.plot()