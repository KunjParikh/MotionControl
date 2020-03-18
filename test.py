import params
from formationCenter import formationCenter
from formationControl import formationControl
#from kalmanFilter import kalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
import pickle
from sklearn.metrics import mean_squared_error
import math

if True or __name__ == "__main__":
  params = params.Params()
  for f, dz_f, hessian_f, z_desired, g, plot_desired in zip(params.f_list,
                                                            params.dz_f_list,
                                                            params.hessian_list,
                                                            params.z_desired_list,
                                                            params.g_list,
                                                            params.plot_desired_list):     
    r_c, r_c_old, r, = params.r_c, params.r_c, params.r
    x_2, y_2 = 0, 0  # Ignored in formationCenter for i == 0
    q, dq, u_r, vel_q = params.q, params.dq, params.u_r, params.vel_q

    ##Kalman Filter initialization
    numSensors = params.numSensors
    p = np.zeros((3, 3))
    z_c = f(r_c[0], r_c[1])
    dz_c = dz_f(r_c[0], r_c[1])
    hessian = params.hessian_f(r_c[0], r_c[1])
    z_r = np.array([f(*pt) for pt in r])

    r_c_plot = [r_c]
    r_plot = [r]
    p_det = []

    keys = list(range(18))
    #Loaded in above cells
    #model_state = load_model("/content/drive/My Drive/SJSU/Final Project/model_state.h5")
    #model_error = load_model("/content/drive/My Drive/SJSU/Final Project/model_error.h5")
    #scaler_state = pickle.load( open( "/content/drive/My Drive/SJSU/Final Project/scaler_state.p", "rb" ) )
    #scaler_error = pickle.load( open( "/content/drive/My Drive/SJSU/Final Project/scaler_error.p", "rb" ) )
    model_state = load_model("model_state.h5")
    model_error = load_model("model_error.h5")
    scaler_state = pickle.load( open( "scaler_state.p", "rb" ) )
    scaler_error = pickle.load( open( "scaler_error.p", "rb" ) )
    p_e_kalman_list = []
    p_e_lstm_list = []

    print(model_state.summary())
    print(model_error.summary())

    stateX = df_state.iloc[:,0:3].to_numpy()
    
    state = scaler_state.transform(np.hstack((stateX, np.zeros((10000,3)) )) )[:,0:3]
    state_predict = model_state.predict(np.reshape(state, (state.shape[0], 1, state.shape[1])))
    #state_predict = scaler_state.inverse_transform(np.hstack((state_predict, np.zeros((state_predict.shape)) )))[:,0:3]
    stateY = df_state.iloc[:,3:].to_numpy()
    stateY = scaler_state.transform(np.hstack((np.zeros((10000,3)), stateY )) )[:,3:]
    
    print(math.sqrt(mean_squared_error(stateY, state_predict)))

    errorX = df_error.iloc[:,0:9].to_numpy()

    error = scaler_error.transform(np.hstack((errorX, np.zeros((10000,9)) )) )[:,0:9]
    error_predict = model_error.predict(np.reshape(error, (error.shape[0], 1, error.shape[1])))
    #error_predict = scaler_error.inverse_transform(np.hstack((error_predict, np.zeros((error_predict.shape)) )))[:,0:9]
    errorY = df_error.iloc[:,9:].to_numpy()
    errorY = scaler_error.transform(np.hstack((np.zeros((10000,9)), errorY )) )[:,9:]
    
    print(math.sqrt(mean_squared_error(errorY, error_predict)))

    for i in range(10000):
        z_r = np.array([f(*pt) for pt in r])
        hessian = params.hessian_f(r_c[0], r_c[1])
        #hessian = np.random.rand(2, 2) * 20

        ##Kalman filter: z_c, dz_c, p = kalmanFilter(z_c, dz_c, r, z_r, r_c, r_c_old, p, hessian, numSensors, df_state, df_error)
        #def kalmanFilter(z_c, dz, r, z_r, r_c, r_c_old, p, hessian, numSensors):

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
        hc = np.array([hessian[0, 0], hessian[1, 0], hessian[0, 1], hessian[1, 1]])  # 4x1
        c = np.hstack((np.ones((numSensors, 1)), r - r_c))  # 4x3
        d = 0.5 * np.vstack([np.kron(pt - r_c, pt - r_c) for pt in r])  # 4x4

        #######USING LSTM
        #s_e = a @ s + h  # 3x1
        state = scaler_state.transform(np.hstack((s, np.zeros(3))).reshape(1,-1))[:,0:3]
        state_predict = model_state.predict(np.reshape(state, (state.shape[0], 1, state.shape[1])))
        state_predict = scaler_state.inverse_transform(np.hstack((np.zeros(state_predict.shape), state_predict )))[:,3:]
        s_e = state_predict[0]

        p_e_kalman = a @ p @ a.T + m  # 3x3 @ 3x1 @ 3x3 = 3x3
        error = p.flatten()
        error = scaler_error.transform(np.hstack((error, np.zeros(9))).reshape(1,-1))[:,0:9]
        error_predict = model_error.predict(np.reshape(error, (error.shape[0], 1, error.shape[1])))
        error_predict = scaler_error.inverse_transform(np.hstack((np.zeros(error_predict.shape), error_predict)))[:,9:]
        #error_predict = scaler_error.inverse_transform(np.hstack((error_predict, np.zeros(error_predict.shape))))[:,0:9]
        p_e_lstm = np.reshape(error_predict[0], (3,3))
        p_e = p_e_lstm
        p_e_kalman_list.append(p_e_kalman)
        p_e_lstm_list.append(p_e_lstm)


        ######END OF LSTM

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
        #return z_c, dz, p
        ##End of Kalman filter


        p_det.append(np.linalg.det(p))
        # z_c = f(r_c[0], r_c[1])
        # dz_c = dz_f(r_c[0], r_c[1])
        # print(dz_c)
        r_c_old = r_c
        r_c, x_2, y_2, hessian = formationCenter(r_c, z_c, dz_c, hessian, x_2, y_2, i,
                                                 params.rotateRight, params.rotateLeft,
                                                 params.mu_f, params.z_desired,
                                                 params.K4, params.dt)

        r, q, dq, u_r, vel_q = formationControl(r_c, r, q, dq, u_r, vel_q, params.a, params.b,
                                                params.dt, params.K2, params.K3, params.phi_inv)
        r_c_plot.append(r_c)
        if i % 150 == 0:
            r_plot.append(r)

    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    z = params.g(x[:, None], y[None, :])

    plt.contour(x, y, z, [params.plot_desired])
    plt.plot(*zip(*r_c_plot), 'b')
    for r in r_plot:
        plt.plot([r[0, 0], r[1, 0]], [r[0, 1], r[1, 1]], 'yo-')
        plt.plot([r[2, 0], r[3, 0]], [r[2, 1], r[3, 1]], 'go-')
    plt.show()

    plt.plot(p_det)
    plt.show()
