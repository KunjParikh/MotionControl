import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import params
from formationCenter import formationCenter
from formationControl import formationControl
# from kalmanFilter import kalmanFilter
import numpy as np
import pandas as pd
import pickle
from pandas import Series

# !ls '/content/drive/My Drive/SJSU/Final Project/model_state.h5'

if True or __name__ == "__main__":
    params = params.Params()
    series_data = pd.Series(dtype='object')
    series_error: Series = pd.Series(dtype='object')
    for function in params.functions:
        r_c, r_c_old, r, = params.r_c, params.r_c, params.r
        x_2, y_2 = 0, 0  # Ignored in formationCenter for i == 0
        q, dq, u_r, vel_q = params.q, params.dq, params.u_r, params.vel_q
        # Kalman Filter initialization
        numSensors = params.numSensors
        p = np.zeros((3, 3))
        # hessian = params.hessian
        z_c = function.f(r_c[0], r_c[1])
        dz_c = function.dz_f(r_c[0], r_c[1])
        hessian = function.hessian_f(r_c[0], r_c[1])
        z_r = np.array([function.f(*pt) for pt in r])

        r_c_plot = [r_c]
        r_plot = [r]
        p_det = []

        keys = list(range(18))
        df_state = pd.DataFrame()
        df_error = pd.DataFrame()

        for i in range(10000):
            z_r = np.array([function.f(*pt) for pt in r])
            # hessian = np.random.rand(2, 2) * 20
            hessian = function.hessian_f(r_c[0], r_c[1])

            # Kalman filter: z_c, dz_c, p = kalmanFilter(z_c, dz_c, r, z_r, r_c, r_c_old, p,
            # hessian, numSensors, df_state, df_error)
            # def kalmanFilter(z_c, dz, r, z_r, r_c, r_c_old, p, hessian, numSensors):

            s = np.array([z_c, *dz_c])  # 3x1
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
            hc = np.array([hessian[0][0], hessian[1][0], hessian[0][1], hessian[1][1]])  # 4x1
            c = np.hstack((np.ones((numSensors, 1)), r - r_c))  # 4x3
            d = 0.5 * np.vstack([np.kron(pt - r_c, pt - r_c) for pt in r])  # 4x4

            s_e = a @ s + h  # 3x1

            df_state = df_state.append(pd.Series(np.append(s, s_e)), ignore_index=True)

            p_e = a @ p @ a.T + m  # 3x3 @ 3x1 @ 3x3 = 3x3
            df_error = df_error.append(pd.Series(np.append(p.flatten(), p_e.flatten())), ignore_index=True)

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
            dz_c = s[1:]
            # return z_c, dz, p
            # End of Kalman filter

            p_det.append(np.linalg.det(p))
            # z_c = f(r_c[0], r_c[1])
            # dz_c = dz_f(r_c[0], r_c[1])
            # print(dz_c)
            r_c_old = r_c
            r_c, x_2, y_2, _ = formationCenter(r_c, z_c, dz_c, hessian, x_2, y_2, i,
                                               params.rotateRight, params.rotateLeft,
                                               function.mu_f, function.z_desired,
                                               params.K4, params.dt)

            r, q, dq, u_r, vel_q = formationControl(r_c, r, q, dq, u_r, vel_q, params.a, params.b,
                                                    params.dt, params.K2, params.K3, params.phi_inv)
            r_c_plot.append(r_c)
            if i % 150 == 0:
                r_plot.append(r)

        x = np.linspace(-10, 10, 200)
        y = np.linspace(-10, 10, 200)
        X, Y = np.meshgrid(x, y)
        Z = function.f(X, Y)

        plt.contour(x, y, Z, [function.z_desired])
        plt.plot(*zip(*r_c_plot), 'b')
        for r in r_plot:
            plt.plot([r[0, 0], r[1, 0]], [r[0, 1], r[1, 1]], 'yo-')
            plt.plot([r[2, 0], r[3, 0]], [r[2, 1], r[3, 1]], 'go-')
        plt.title(function.name)
        plt.savefig("{}_kalmanFilter.pdf".format(function.name), bbox_inches='tight')
        plt.close()

        # plt.plot(p_det)
        # plt.show()

        series_data = series_data.append(pd.Series({function.name: df_state}))
        series_error = series_error.append(pd.Series({function.name: df_error}))

    pickle.dump(series_data, open("df_state.p", "wb"))
    pickle.dump(series_error, open("df_error.p", "wb"))

    # Sample code explaining the format of df_state data-structure
    # import pickle as pk
    # x = pk.load(open("C:\\Users\\KunjJParikh\\PycharmProjects\\MotionControl\\df_state.p", 'rb'))
    # for name, value in x.items():
    #     print(name)
    # #print(x['rhombus'])