import params
from formationCenter import formationCenter
from formationControl import formationControl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from sklearn.metrics import mean_squared_error
import math as math
import os as os
from keras.models import load_model
import utils as utils
pltVar = utils.PlotVariable("testVar")

#We cant model hessian as just function of r_c, we need z_r to account for different shapes.
#Matlab code uses: r_c, z_c, dz_c, r, z_r, hessian.


class Hessian:
    def __init__(self):
        pass

    def stepFunction(self, function, plot=False, collect=False, test=False):
        p = params.Params()
        r_c, r_c_old, x_2, y_2, r = p.r_c, p.r_c, 0, 0, p.r
        r_c_plot = [r_c]
        r_plot = [r]
        q, dq, u_r, vel_q = p.q, p.dq, p.u_r, p.vel_q
        trainData, model = False, False
        if collect:
            trainData = pd.DataFrame()
        if test:
            model_orig = load_model("hessian.h5")
            scaler = pkl.load(open("hessian_scaler.p", "rb"))
            # re-define the batch size
            batch_size = 1
            # re-define model
            model = Sequential()
            model.add(LSTM(22, batch_input_shape=(batch_size, 1, 21), stateful=True))
            model.add(Dense(4))
            # copy weights
            weights = model_orig.get_weights()
            model.set_weights(weights)
            # compile model
            model.compile(loss='mean_squared_error', optimizer='adam')
            print(model.summary())

        hessian = function.hessian_f(r_c[0], r_c[1])
        z_c = function.f(r_c[0], r_c[1])
        y_2 = dz_c / norm(dz_c)
        x_2 = p.rotateRight @ y_2

        for i in range(10000):
            # Decoupled: Ideal test.
            z_c = function.f(r_c[0], r_c[1])
            dz_c = function.dz_f(r_c[0], r_c[1])
            realHessian = []
            predictedHessians = []
            if collect:
                hessian = function.hessian_f(r_c[0], r_c[1])
            if test:
                realHessian.append(np.linalg.det(function.hessian_f(r_c[0], r_c[1])))
                predictedHessians.append(np.linalg.det(hessian))

            z_r = np.array([function.f(*pt) for pt in r])
            # hessian = [[0, 0], [0, 0]]
            r_c, x_2, y_2 = formationCenter(r_c, z_c, dz_c, hessian, x_2, y_2, function.mu_f, function.z_desired)
            r, q, dq, u_r, vel_q = formationControl(r_c, r, q, dq, u_r, vel_q)
            state = np.concatenate([r_c, [z_c], dz_c, *r, z_r, *hessian])
            if collect:
                trainData = trainData.append(pd.Series(state, ignore_index=True))
            if test:
                state = scaler.transform(state.reshape(1,-1))
                hessian = model.predict(np.reshape(state, (state.shape[0], 1, state.shape[1])),
                                                    batch_size=1)[0].reshape(2,2)
            pltVar.push(hessian[0,0])
            if plot:
                r_c_plot.append(r_c)
                if i % 150 == 0:
                    r_plot.append(r)

        if plot:
            x = np.linspace(-10, 10, 200)
            y = np.linspace(-10, 10, 200)
            xx, yy = np.meshgrid(x, y)
            z = function.f(xx, yy)

            plt.contour(x, y, z, [function.z_desired])
            plt.plot(*zip(*r_c_plot), 'b')
            for r in r_plot:
                plt.plot([r[0, 0], r[1, 0]], [r[0, 1], r[1, 1]], 'yo-')
                plt.plot([r[2, 0], r[3, 0]], [r[2, 1], r[3, 1]], 'go-')
            plt.title(function.name)
            plt.show()
        pltVar.plot()

        print("Finished stepping through function: {}\n".format(function.name))
        if test:
            print("Hessian error is: {}".format(math.sqrt(mean_squared_error(realHessian, predictedHessians))))
        return trainData

    def collect(self):
        p = params.Params()
        dataAll = pd.Series(dtype='object')
        for function in p.functions:
            data = self.stepFunction(function, collect=True)
            dataAll = dataAll.append(pd.Series({function.name: data}))

        pkl.dump(dataAll, open("hessian_data.p", "wb"))
        return dataAll

    def train(self):
        p = params.Params()
        data = pkl.load(open('hessian_data.p', 'rb'))
        nfeatures = 21
        rootpath = ""
        scaler = MinMaxScaler(feature_range=(-1, 1))
        model = Sequential()
        model.add(LSTM(nfeatures+1, batch_input_shape=(p.batch_size, 1, nfeatures), stateful=True))
        model.add(Dense(4))
        model.compile(loss='mean_squared_error', optimizer='adam')

        for function in p.functions:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            train = scaler.fit_transform(data[function.name])
            pkl.dump(scaler, open("hessian_scaler.p", "wb"))
            #Skip last element, and upshift hessian for training next step
            trainX, trainY = train[:-1,:], train[1:, -4:]
            #Align to batch_size of 25
            trainX, trainY = trainX[24:,:], trainY[24:,:]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

            es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
            rs = LambdaCallback(on_epoch_end=lambda epoch, logs: model.reset_states())
            mc = ModelCheckpoint(os.path.join(rootpath, "hessian.h5"),
                                 monitor='loss', mode='min', verbose=1, save_best_only=True)
            #Reuse model in loops - incremental training.
            history = model.fit(trainX, trainY, epochs=200, batch_size=p.batch_size, verbose=2, callbacks=[es, mc, rs],
                                shuffle=False)

            trainPredict = model.predict(trainX, batch_size=p.batch_size)
            # calculate root mean squared error
            trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
            print("Train Score: {} RMSE".format(trainScore))

    def test(self):
        p = params.Params()
        for function in p.test_functions:
            self.stepFunction(function, test=True, plot=True)

if True or __name__ == "__main__":
    hessian = Hessian()
    #hessian.collect()
    #hessian.train()
    hessian.test()