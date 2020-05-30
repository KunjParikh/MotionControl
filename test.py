import matplotlib
matplotlib.use('Agg')
import pickle
# from kalmanFilter import kalmanFilter
import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import params
from formationCenter import formationCenter
from formationControl import formationControl
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy.linalg import norm

# Comment this to use GPU. But looks like for me CPU (~10s) is faster than GPU (~80s).
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if True or __name__ == "__main__":
    params = params.Params()
    model_state_orig = load_model("model_state.h5")
    model_error_orig = load_model("model_error.h5")

    scaler_state = pickle.load(open("scaler_state.p", "rb"))
    scaler_error = pickle.load(open("scaler_error.p", "rb"))

    df_error = pickle.load(open('df_error.p', 'rb'))
    df_state = pickle.load(open('df_state.p', 'rb'))



    # re-define the batch size
    batch_size = 1
    # re-define model
    model_state = Sequential()
    model_state.add(LSTM(4, batch_input_shape=(batch_size, 1, 3), stateful=True))
    model_state.add(Dense(3))
    # copy weights
    old_weights = model_state_orig.get_weights()
    model_state.set_weights(old_weights)
    # compile model
    model_state.compile(loss='mean_squared_error', optimizer='adam')

    # re-define model
    model_error = Sequential()
    model_error.add(LSTM(10, batch_input_shape=(batch_size, 1, 9), stateful=True))
    model_error.add(Dense(9))
    # copy weights
    old_weights = model_error_orig.get_weights()
    model_error.set_weights(old_weights)
    # compile model
    model_error.compile(loss='mean_squared_error', optimizer='adam')

    model_hessian_orig = load_model("hessian.h5")
    scaler_hessian = pickle.load(open("hessian_scaler.p", "rb"))
    # re-define model
    model_hessian = Sequential()
    model_hessian.add(LSTM(22, batch_input_shape=(batch_size, 1, 21), stateful=True))
    model_hessian.add(Dense(4))
    # copy weights
    old_weights = model_hessian_orig.get_weights()
    model_hessian.set_weights(old_weights)
    # compile model
    model_hessian.compile(loss='mean_squared_error', optimizer='adam')

    print(model_state.summary())
    print(model_error.summary())
    print(model_hessian.summary())

    # testFunctions = [x for x in params.functions if x.name in ['rhombus']]
    testFunctions = params.functions

    for function in testFunctions:
        r_c, r_c_old, r, = params.r_c, params.r_c, params.r
        q, dq, u_r, vel_q = params.q, params.dq, params.u_r, params.vel_q

        # Kalman Filter initialization
        numSensors = params.numSensors
        p = np.zeros((3, 3))
        z_c = function.f(r_c[0], r_c[1])
        dz_c = function.dz_f(r_c[0], r_c[1])
        hessian = function.hessian_f(r_c[0], r_c[1])
        z_r = np.array([function.f(*pt) for pt in r])

        y_2 = dz_c / norm(dz_c)
        x_2 = params.rotateRight @ y_2

        r_c_plot = [r_c]
        r_plot = [r]
        p_det = []

        keys = list(range(18))

        s_e_kalman_list = []
        s_e_lstm_list = []
        p_e_kalman_list = []
        p_e_lstm_list = []

        #Rhombus' result improves on rescaling state... a hack.
        scaler_state.fit(df_state[function.name])
        scaler_error.fit(df_error[function.name])

        stateX = df_state[function.name].iloc[:, 0:3].to_numpy()

        state = scaler_state.transform(np.hstack((stateX, np.zeros((10000, 3)))))[:, 0:3]
        state_predict = model_state.predict(np.reshape(state, (state.shape[0], 1, state.shape[1])), batch_size=batch_size)
        # stateY = df_state[function.name].iloc[:, 3:].to_numpy()
        # stateY = scaler_state[function.name].transform(np.hstack((np.zeros((10000, 3)), stateY)))[:, 3:]

        errorX = df_error[function.name].iloc[:, 0:9].to_numpy()

        error = scaler_error.transform(np.hstack((errorX, np.zeros((10000, 9)))))[:, 0:9]
        error_predict = model_error.predict(np.reshape(error, (error.shape[0], 1, error.shape[1])), batch_size=batch_size)
        # errorY = df_error[function.name].iloc[:, 9:].to_numpy()
        # errorY = scaler_error[function.name].transform(np.hstack((np.zeros((10000, 9)), errorY)))[:, 9:]

        for i in range(10000):
            z_r = np.array([function.f(*pt) for pt in r])
            #hessian = function.hessian_f(r_c[0], r_c[1])
            #hessian = np.random.rand(2, 2) * 20

            # Kalman filter: z_c, dz_c, p = kalmanFilter(z_c, dz_c, r, z_r, r_c, r_c_old,
            #   p, hessian, numSensors, df_state, df_error)
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

            # USING LSTM
            s_e_kalman = a @ s + h  # 3x1
            state = scaler_state.transform(np.hstack((s, np.zeros(3))).reshape(1, -1))[:, 0:3]
            state_predict = model_state.predict(np.reshape(state, (state.shape[0], 1, state.shape[1])), batch_size=batch_size)
            state_predict = scaler_state.inverse_transform(
                np.hstack((np.zeros(state_predict.shape), state_predict)))[:, 3:]
            s_e_lstm = state_predict[0]
            s_e = s_e_lstm
            s_e_kalman_list.append(s_e_kalman)
            s_e_lstm_list.append(s_e_lstm)

            p_e_kalman = a @ p @ a.T + m  # 3x3 @ 3x1 @ 3x3 = 3x3
            error = p.flatten()
            error = scaler_error.transform(np.hstack((error, np.zeros(9))).reshape(1, -1))[:, 0:9]
            error_predict = model_error.predict(np.reshape(error, (error.shape[0], 1, error.shape[1])), batch_size=batch_size)
            error_predict = scaler_error.inverse_transform(
                np.hstack((np.zeros(error_predict.shape), error_predict)))[:, 9:]
            p_e_lstm = np.reshape(error_predict[0], (3, 3))
            p_e = p_e_lstm
            p_e_kalman_list.append(p_e_kalman)
            p_e_lstm_list.append(p_e_lstm)

            # END OF LSTM

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
            r_c, x_2, y_2 = formationCenter(r_c, z_c, dz_c, hessian, x_2, y_2, function.mu_f, function.z_desired)

            r, q, dq, u_r, vel_q = formationControl(r_c, r, q, dq, u_r, vel_q)

            hessian_state = np.concatenate([r_c, [z_c], dz_c, *r, z_r, *hessian])
            hessian_state = scaler_hessian.transform(hessian_state.reshape(1, -1))
            hessian = model_hessian.predict(np.reshape(hessian_state, (hessian_state.shape[0], 1, hessian_state.shape[1])),
                                    batch_size=1)[0].reshape(2, 2)
            r_c_plot.append(r_c)
            if i % 150 == 0:
                r_plot.append(r)

        x = np.linspace(-10, 10, 200)
        y = np.linspace(-10, 10, 200)
        X, Y = np.meshgrid(x, y)
        Z = function.f(X, Y)

        cs = plt.contour(x, y, Z, [function.z_desired])
        plt.plot(*zip(*r_c_plot), 'b')
        for r in r_plot:
            plt.plot([r[0, 0], r[1, 0]], [r[0, 1], r[1, 1]], 'yo-')
            plt.plot([r[2, 0], r[3, 0]], [r[2, 1], r[3, 1]], 'go-')
        plt.title(function.name)
        plt.savefig("{}.pdf".format(function.name), bbox_inches='tight')
        plt.close()

        # Estimate error by measuring field value on predicted trajectory, and comparing to desired field value
        tracedValues = []
        # Need to filter out values where we are still approaching the curve.
        onCurve = False
        for value in map(lambda pt: function.f(pt[0], pt[1]), r_c_plot):
            if onCurve:
                tracedValues.append(value)
            elif value < 1.5 * function.z_desired:
                onCurve = True
            else:
                pass
        desiredValue = np.full(len(tracedValues), function.z_desired)
        if len(tracedValues)>0:
            print("Traced value function {}: Error = {}, NumPoints close to curve = {}".format(function.name,
                                                               mean_squared_error(desiredValue, tracedValues), len(tracedValues)))
        else:
            print("Traced value function {}: No Convergence".format(function.name))
