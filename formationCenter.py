import numpy as np
from numpy.linalg import norm
from math import cos, sin, atan2
import matplotlib.pyplot as plt
from kalmanFilter import kalmanFilter
import params

def formationCenter(r_c, z_c, dz_c, hessian, x_2, y_2, i,rotateRight, rotateLeft, mu_f, z_desired, K4, dt):

    #hessian = np.array([[2, 0],[2, 0]]) * np.random.rand(1)
    #z_c, dz_c, p = kalmanFilter(z_c, dz_c, r, z_r, r_c, r_c_old, p, hessian, numSensors)

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
    #print([x_2, y_2])

    x_2 = x_2 + dt * u_c * y_2
    x_2 = x_2/norm(x_2)
    y_2 = rotateLeft @ x_2

    r_c = r_c + dt * x_2
    return r_c, x_2, y_2, hessian


if __name__ == "__main__":
    r_c, r_c_old, f, dz_f, x_2, y_2 = params.r_c, params.r_c, params.f, params.dz_f, 0, 0
    hessian = params.hessian
    r_c_plot = [r_c]
    for i in range(10000):
        #Decoupled: Ideal test.
        z_c = f(r_c[0], r_c[1])
        dz_c = dz_f(r_c[0], r_c[1])

        r_c, x_2, y_2, hessian = formationCenter(r_c, z_c, dz_c, hessian,  x_2, y_2, i,
                          params.rotateRight, params.rotateLeft,
                          params.mu_f, params.z_desired,
                          params.K4, params.dt)
        r_c_plot.append(r_c)

    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    z = params.g(x[:,None], y[None,:])

    plt.contour(x, y, z, [16])
    plt.plot(*zip(*r_c_plot), 'b')
    plt.show()


    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.plot([r[0,0], r[1,0]], [r[0,1], r[1,1]], 'yo-')
    # plt.plot([r[2,0], r[3,0]], [r[2,1], r[3,1]], 'go-')
    # plt.plot(*zip(*r_c_plot), 'b')
