import params as params
import numpy as np
from math import sqrt, floor, pi
from numpy import sin, cos, array
from numpy.linalg import norm

def formationControl(r_c, r, dz_c, q, dq, u_r, vel_q, t):
    
    p = params.Params()
    a = p.a
    b = p.b
    dt = p.dt
    K2 = p.K2
    K3 = p.K3
    phi_inv = p.phi_inv

    # Moving opposite to gradient direction : Source seeking
    # velocity_center = - dz_c / norm(dz_c)
    # Moving perpendiular to gradient direction : Boundary tracking
    velocity_center = p.rotateLeft @ (dz_c / norm(dz_c))

    y1 = velocity_center
    x1 = p.rotateRight @ y1

    # e_1 = r[1] - r[0]
    # e_1 = e_1 / np.linalg.norm(e_1)
    # e_2 = r[2] - r[3]
    # e_2 = e_2 / np.linalg.norm(e_2)
    # q_0 = np.array([
    #     [0, 0],
    #     (a / sqrt(2)) * e_1,
    #     (b / sqrt(2)) * e_2,
    #     [0, 0]
    # ])

    if p.numSensors == 4:
        q_0 = np.array([
            [0, 0],
            (a / sqrt(2)) * y1,
            (b / sqrt(2)) * x1,
            [0, 0]
        ])

    if p.numSensors == 2:
        sigma = 10.0
        q_0 = np.array([
            [0, 0],
            (a / sqrt(2)) * x1,
        ])
        if floor( t / 4 ) % 2 == 0:
            q_0[1] = array([[sin(sigma/pi), cos(sigma/pi)],
                               [-cos(sigma/pi), sin(sigma/pi)]]) @ q_0[1]
        else:
            q_0[1] = array([[sin(sigma), -cos(sigma)],
                            [cos(sigma), sin(sigma)]]) @ q_0[1]

    # dq[1:] = dq[1:] + dt * u_r[1:]
    # u_r[1:] = -K2 * (q[1:] - q_0[1:]) - K3 * dq[1:]
    # vel_q[1:] = vel_q[1:] + dt * u_r[1:]
    vel_q[1:] = -10 * (q[1:] - q_0[1:])
    q[1:] = q[1:] + dt * vel_q[1:]
    q_N = np.append([r_c], q[1:], axis=0)
    r = phi_inv @ q_N

    return r, q, dq, u_r, vel_q
#

# if __name__ == "__main__":
#     p = params.Params()
#     r_c, r, q, dq, u_r, vel_q = p.r_c, p.r, p.q, p.dq, p.u_r, p.vel_q
#     dz_c = p.function.dz_f(r_c[0], r_c[1])
#
#     for i in range(1000):
#         print(r)
#         r, q, dq, u_r, vel_q= formationControl(r_c, r, dz_c, q, dq, u_r, vel_q, i)
