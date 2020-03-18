import params
import numpy as np
from math import sqrt

def formationControl(r_c, r, q, dq, u_r, vel_q, a, b, dt, K2, K3, phi_inv):
    e_1 = r[1] - r[0]
    e_1 = e_1 / np.linalg.norm(e_1)
    e_2 = r[2] - r[3]
    e_2 = e_2 / np.linalg.norm(e_2)

    q_0 = np.array([
        [0, 0],
        (a / sqrt(2)) * e_1,
        (b / sqrt(2)) * e_2,
        [0, 0]
    ])

    dq[1:] = dq[1:] + dt * u_r[1:]
    u_r[1:] = -K2 * (q[1:] - q_0[1:]) - K3 * dq[1:]

    vel_q[1:] = vel_q[1:] + dt * u_r[1:]
    q[1:] = q[1:] + dt * vel_q[1:]

    q_N = np.append([r_c], q[1:], axis=0)

    r = phi_inv @ q_N
    return r, q, dq, u_r, vel_q
#

if __name__ == "__main__":
    params = params.Params()
    r_c, r, q, dq, u_r, vel_q = params.r_c, params.r, params.q, \
                                params.dq, params.u_r, params.vel_q

    for i in range(1000):
        print(r)
        r, q, dq, u_r, vel_q = formationControl(r_c, r, q, dq, u_r, vel_q, params.a, params.b,
                            params.dt, params.K2, params.K3, params.phi_inv)
