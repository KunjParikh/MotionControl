import numpy as np

def kalmanFilter(z_c, dz, r, z_r, r_c, r_c_old, p, hessian, numSensors):
    s = np.array([z_c, *dz])  # 3x1
    m = 0.001 * np.identity(3)  # 3x1
    R = 0.001 * np.identity(numSensors)  # 4x4
    u = 0.001 * np.identity(4)  # 4x4
    a = np.array([[1, *(r_c - r_c_old)],
                  [0, 1, 0],
                  [0, 0, 1]])  # 3x3
    h = np.array([
        0,
        *(hessian @ (r_c - r_c_old))  # 2x2 . 2x1 = 2x1
    ])  # 3x1
    hc = np.array([hessian[0, 0], hessian[1, 0], hessian[0, 1], hessian[1, 1]])  # 4x1
    c = np.hstack((np.ones((numSensors, 1)), r - r_c))  # 4x3
    d = 0.5 * np.vstack([np.kron(pt - r_c, pt - r_c) for pt in r])  # 4x4


    s_e = a @ s + h  # 3x1
    p_e = a @ p @ a.T + m  # 3x3 @ 3x1 @ 3x3 = 3x3

    # c @ p_e @ c.T = 4x3 @ 3x3 @ 3x4 = 4x4
    # d @ u @ d.T = 4x4 @ 4x4 @ 4x4 = 4x4
    # p_e @ c.T @ inv... = 3x3 @ 3x4 @ 4x4 = 3x4

    K = p_e @ c.T @ np.linalg.inv((c @ p_e @ c.T) + (d @ u @ d.T) + R)  # 3x4

    # c @ s_e = 4x3 @ 3x1 = 4x1
    # d @ hc = 4x4 @ 4x1 = 4x1
    # K @ ... = 3x4 @ 4x1 = 3x1
    s = s_e + K @ (z_r - (c @ s_e) - (d @ hc))  # 3x1

    # d @ u @ d.T = 4x4 @ 4x4 @ 4x4 = 4x4
    # c.T @ .. @ c = 3x4 @ 4x4 @ 4x3 = 3x3
    p = np.linalg.inv(np.linalg.inv(p_e) + c.T @ np.linalg.inv(d @ u @ d.T + R) @ c)  # 3x3

    z_c = s[0]
    dz = s[1:]
    return z_c, dz, p
