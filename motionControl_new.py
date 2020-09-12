import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import params as params
from formationCenter import formationCenter
from formationControl import formationControl
# from kalmanFilter import kalmanFilter
import numpy as np
import pandas as pd
import pickle as pkl
from numpy.linalg import norm
from keras.models import Sequential
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Concatenate
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
import os
import sys
from math import sqrt, floor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from utils import PlotVariables
from keras.utils.vis_utils import plot_model
from statistics import mean
import time

class Shape:
    def __init__(self, function):
        p = params.Params()
        self.params = p
        self.function = function
        self.name = self.function.name

        self.r_c, self.r_c_old, self.r, = p.r_c, p.r_c, p.r
        r_c = self.r_c
        self.z_c = function.f(r_c[0], r_c[1])
        self.dz_c = function.dz_f(r_c[0], r_c[1])

        self.y_2 = self.dz_c / norm(self.dz_c)
        self.x_2 = p.rotateRight @ self.y_2
        self.q, self.dq, self.u_r, self.vel_q = p.q, p.dq, p.u_r, p.vel_q
        self.p = np.zeros((3, 3))

        z_r = np.array([self.function.f(*pt) for pt in self.r])

        self.init_state = [self.r_c, self.z_c, self.dz_c, self.r_c_old, self.p, self.r,
                 self.q, self.dq, self.u_r, self.vel_q, self.x_2, self.y_2, z_r]
        self.state = self.init_state
        self.old_state = self.init_state
        self.pltVarShape = False

    def kalmanFilter(self, z_c, dz, r, z_r, r_c, r_c_old, p, hessian, model=False):
        px = params.Params()
        numSensors = px.numSensors
        # If there are two sensors, we combine past and current measurement to obtain 4 readings.
        if numSensors==2:
            numSensors=4
            r = np.concatenate((self.old_state[5], self.state[5]), axis=0)
            z_r = np.concatenate((self.old_state[-1], self.state[-1]), axis=0)
            # r_c_2 = np.array([r_c_old, r_c_old, r_c, r_c])
            # c = np.hstack((np.ones((numSensors, 1)), r - r_c_2))  # 4x3
            # d = 0.5 * np.vstack([np.kron(pt - r_c, pt - r_c) for pt, r_c in zip(r, r_c_2)])  # 4x4

        c = np.hstack((np.ones((numSensors, 1)), r - r_c))  # 4x3
        d = 0.5 * np.vstack([np.kron(pt - r_c, pt - r_c) for pt in r])  # 4x4
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
        hc = np.array([hessian[0][0], hessian[1][0], hessian[0][1], hessian[1][1]])  # 4x1
        # c = np.hstack((np.ones((numSensors, 1)), r - r_c))  # 4x3
        # d = 0.5 * np.vstack([np.kron(pt - r_c, pt - r_c) for pt in r])  # 4x4

        data = [s, hessian, r_c - r_c_old, r, z_r]
        if model:
            s_e = model.predict(data)
            if np.any(np.isinf(s_e)):
                s_e = s
            # s_e = model.predict([s, p])
            # if (p_e == np.zeros((3,3))).flatten().all():
            #     p_e = m
            p_e = m
            K = p_e @ c.T @ np.linalg.inv((c @ p_e @ c.T) + (d @ u @ d.T) + R)
            s = s_e + K @ (z_r - (c @ s_e) - (d @ hc))
            p = p_e
            # s = s_e
        else:
            s_e = a @ s + h
            p_e = a @ p @ a.T + m
            K = p_e @ c.T @ np.linalg.inv((c @ p_e @ c.T) + (d @ u @ d.T) + R)
            s = s_e + K @ (z_r - (c @ s_e) - (d @ hc))
            p = np.linalg.inv(np.linalg.inv(p_e) + c.T @ np.linalg.inv(d @ u @ d.T + R) @ c)  # 3x3
            # s = s+np.random.random_sample(3)

        self.pltVarShape.push('dz_x', s_e[1], s[1])
        self.pltVarShape.push('dz_y', s_e[2], s[2])
        z_c = s[0]
        dz = s[1:]
        # Use actual s and p value - not the one predicted by eq1 and eq2
        # return only s and p values for data here, lag values etc are done in training code.
        return z_c, dz, p, data

    def step(self, t, model=False):
        r_c, z_c, dz_c, r_c_old, p, r, q, dq, u_r, vel_q, x_2, y_2, z_r = self.state

        #Inputs
        z_r = np.array([self.function.f(*pt) for pt in r]) #Mimick measurements on robots
        hessian = self.function.hessian_f(r_c[0], r_c[1]) #True value
        z_c = self.function.f(*r_c) # Measure field value at center
        # dz_c = self.function.dz_f(*r_c) # Measure gradient value at the center

        if model:
            # z_c_k, dz_c_k, p_k, data = kalmanFilter(z_c, dz_c, r, z_r, r_c, r_c_old, p, hessian)
            z_c_p, dz_c_p, p_p, data = self.kalmanFilter(z_c, dz_c, r, z_r, r_c, r_c_old, p, hessian, model)

            z_c = z_c_p
            dz_c = dz_c_p
            p = p_p

        else:
            z_c, dz_c, p, data = self.kalmanFilter(z_c, dz_c, r, z_r, r_c, r_c_old, p, hessian)
        # z_c = f(r_c[0], r_c[1])
        # dz_c = dz_f(r_c[0], r_c[1])

        r_c_old = r_c
        # We dont need to calculate force. Velocity is sufficient.
        # If we dont do this, fc only tracks gradient, not at desired field value, but where it was left to start with.
        r_c, x_2, y_2 = formationCenter(r_c, z_c, dz_c, hessian, x_2, y_2,
                                           self.function.mu_f, self.function.z_desired)

        r, q, dq, u_r, vel_q = formationControl(r_c, r, dz_c, q, dq, u_r, vel_q, t)

        self.old_state = self.state
        self.state = [r_c, z_c, dz_c, r_c_old, p, r, q, dq, u_r, vel_q, x_2, y_2, z_r]
        # data = [r_c, z_c, dz_c, hessian, r, z_r, p]

        return data


    def trace(self, model=False):
        print("Tracing shape for {}".format(self.name))
        data = []
        r_c_plot = [self.r_c]
        r_plot = [self.r]
        p_det = []
        z_r = np.array([self.function.f(*pt) for pt in self.r])

        self.state = self.init_state
        self.old_state = self.init_state
        # print("Shape:trace:1: {}".format(time.time()))
        self.pltVarShape = PlotVariables("{}MODEL".format(self.name), ["dz_x", "dz_y"])
        for i in range(10000):
            d = self.step(i, model)
            data.append(d)
            r_c_plot.append(self.state[0])
            p_det.append(np.linalg.det(self.state[4]))
            if i % 150 == 0:
                r_plot.append(self.state[5])
        dataframe = pd.DataFrame(data, columns=['s', 'hessian', 'r_c', 'r', 'z_r'])
        # print("Shape:trace:2: {}".format(time.time()))
        #self.pltVarShape.plot()
        if model:
            cmpData = []
            self.state = self.init_state
            self.old_state = self.init_state
            self.pltVarShape = PlotVariables("{}KF".format(self.name), ["dz_x", "dz_y"])
            for i in range(10000):
                d = self.step(i)
                cmpData.append(d)
            cmpDataframe = pd.DataFrame(cmpData, columns=['s', 'hessian', 'r_c_delta', 'r_delta', 'z_r'])
            #self.pltVarShape.plot()
        else:
            cmpDataframe = False
        # print("Shape:trace:3: {}".format(time.time()))
        self.plot(r_c_plot, r_plot)
        return dataframe, cmpDataframe

    def simulate(self):
        model = MotionControlModel(loadModel=True)
        return self.trace(model)

    def plot(self, r_c_plot, r_plot):
        x = np.linspace(-10, 10, 200)
        y = np.linspace(-10, 10, 200)
        X, Y = np.meshgrid(x, y)
        Z = self.function.f(X, Y)

        plt.contour(x, y, Z, [self.function.z_desired])
        plt.plot(*zip(*r_c_plot), 'b')
        # r_plot = [] # Temporarily dont plot individual robots
        for r in r_plot:
            if r.shape[0] == 4:
                plt.plot([r[0, 0], r[1, 0]], [r[0, 1], r[1, 1]], 'yo-')
                plt.plot([r[2, 0], r[3, 0]], [r[2, 1], r[3, 1]], 'go-')
            if r.shape[0] == 2:
                plt.plot([r[0, 0], r[1, 0]], [r[0, 1], r[1, 1]], 'yo-')
        plt.title(self.name)
        # plt.show()
        plt.savefig("{}.pdf".format(self.name), bbox_inches='tight')
        plt.close()


class MotionControlModel:
    def __init__(self, loadModel=False):

        self.rootpath = "/tmp" if os.name=='posix' else ""
        self.model = False
        self.scalerX = False
        self.scalerY = False

        self.numFeaturesStateAdded = 21
        self.numFeaturesState = 3
        self.numFeaturesError = 9
        self.lagWindowSize = 50
        # self.history = np.zeros((self.lagWindowSize, self.numFeaturesStateAdded+self.numFeaturesError)).tolist()
        self.history = np.zeros((self.lagWindowSize, self.numFeaturesStateAdded)).tolist()

        self.lstmStateSize = 40
        self.lstmErrorSize = 20

        if loadModel:
            self.load()

    def createModel(self, batch_size, stateful):
        lagWindowSize = self.lagWindowSize
        numFeaturesState = self.numFeaturesState
        numFeaturesStateAdded = self.numFeaturesStateAdded
        input1 = Input(batch_shape=(batch_size, lagWindowSize, numFeaturesStateAdded))
        split = Lambda(lambda x: x[:, :, 0:3])(input1)
        lstm1 = LSTM(self.lstmStateSize, stateful=stateful)(split)
        output1 = Dense(numFeaturesState, name='dense_state')(lstm1)
        model = Model(inputs=[input1], outputs=[output1])
        #
        # split_z = Lambda(lambda x: x[:, :, 0:1])(input1)
        # split_dzx = Lambda(lambda x: K.concatenate((x[:,:, 1:2], x[:,:, 7:9]), axis=2))(input1)
        # split_dzy = Lambda(lambda x: K.concatenate((x[:, :, 2:3], x[:, :, 7:9]), axis=2))(input1)
        # split_dzy = Lambda(lambda x: x[:, :, 0:4])(input1)
        # split2 = Lambda(lambda x: x[:, :, -10::3])(input1)
        # z_vals = Concatenate()([split1, split2])
        # flatten1 = Flatten()(z_vals)
        # z = Dense(1)(Dense(10)(flatten1))
        # dz_x = Dense(1)(Dense(10)(flatten1))
        # dz_y = Dense(1)(Dense(10)(flatten1))
        # lstm1 = LSTM(3, stateful=False)(split_z)
        # cat1 = Dense(1)(lstm1)
        # lstm2 = LSTM(10, stateful=False)(split_dzx)
        # cat2 = Dense(1)(lstm2)
        # lstm3 = LSTM(10, stateful=False)(split_dzy)
        # cat3 = Dense(1)(lstm3)
        # output1 = Concatenate()([cat1, cat2, cat3])

        # z_l = LSTM(40, stateful=False)(z_vals)
        # z = Dense(1)(z_l)
        # dz_x_l = LSTM(40, stateful=False)(z_vals)
        # dz_x = Dense(1)(dz_x_l)
        # dz_y_l = LSTM(40, stateful=False)(z_vals)
        # dz_y = Dense(1)(dz_y_l)

        # s_e = LSTM(3, stateful=False)(split1)
        # denseInput = TimeDistributed(Dense(10))(split2)
        # lstm1 = SimpleRNN(3, stateful=False)(denseInput)
        # lstm1 = LSTM(3, stateful=False)(denseInput)
        # denseFlat = Flatten()(denseInput)
        # dense2 = Dense(10)(denseFlat)
        # lstm1 = LSTM(self.lstmStateSize, stateful=stateful)(denseInput)
        # merge1 = Concatenate()([s_e, lstm1])

        # output1 = Concatenate()([z, dz_x, dz_y])
        # output1 = Dense(numFeaturesState, name='dense_state')(merge1)
        # output1 = TimeDistributed(Dense(numFeaturesState, activation='linear'))(lstm1)

        # input2 = Input(batch_shape=(batch_size, lagWindowSize, numFeaturesError))
        # lstm2 = LSTM(self.lstmErrorSize, stateful=stateful)(input2)
        # output2 = Dense(numFeaturesError, name='dense_error')(lstm2)

        # model = Model(inputs=[input1, input2], outputs=[output1, output2])
        # model = Model(inputs=[input1], outputs=[output1])
        print(model.summary())
        return model

    def processData(self, df, learn=True):
        if isinstance(df, pd.DataFrame):
            raw_data = df.to_numpy()
        else:
            raw_data = df
        # ['s', 'hessian', 'r_c_delta', 'r_delta', 'z_r']
        px = params.Params()
        numSensors = px.numSensors
        unscaled_data = np.array([[x[0][0], x[0][1], x[0][2],
                               x[1][0][0], x[1][0][1], x[1][1][0], x[1][1][1],
                               x[2][0], x[2][1],
                               x[3][0][0], x[3][0][1], x[4][0],
                               x[3][1][0], x[3][1][1], x[4][1],
                               x[3][2][0], x[3][2][1], x[4][2],
                               x[3][3][0], x[3][3][1], x[4][3]
                            ] for x in raw_data])
        #Retain only every nth data element
        # unscaled_data = unscaled_data[::10,:]

        lagWindowSize = self.lagWindowSize
        numFeaturesState = self.numFeaturesState
        numFeaturesStateAdded = self.numFeaturesStateAdded
        datalength = unscaled_data.shape[0]

        if learn:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_data = scaler.fit_transform(unscaled_data)
            rootpath = "/tmp" if os.name == 'posix' else ""
            pkl.dump(scaler, open(os.path.join(rootpath, "scaler.p"), "wb"))
        else:
            self.history.append(unscaled_data[0])
            self.history = self.history[1:]
            scaled_data = self.scaler.transform(self.history)

        # LSTM needs input in [samples, timestep, features] format.
        if learn:
            X = np.array([scaled_data[i-lagWindowSize:i,:] for i in range(lagWindowSize, datalength)])
            Y = scaled_data[lagWindowSize:,:]
            # We dont need 4th parameter for prediction so -1
            Y = Y[:, 0:numFeaturesState]
            # Y = [Y[:, 0:numFeaturesState], Y[:, numFeaturesStateAdded:]]
        else:
            X = np.array([scaled_data])
            # X = [X[:, :, 0:numFeaturesStateAdded], X[:, :, numFeaturesStateAdded:]]
            Y = False
        return X, Y

    def trainShape(self, dataset, validationData, model=False):
        X, Y = self.processData(dataset)
        vX, vY = self.processData(validationData)

        batch_size = 60
        # bs = floor(X[0].shape[0] / batch_size) * batch_size
        bs = floor(X.shape[0] / batch_size) * batch_size
        # X = [X[0][:bs,:,:], X[1][:bs,:,:]]
        # Y = [Y[0][:bs, :], Y[1][:bs, :]]
        X = X[:bs, :, :]
        Y = Y[:bs, :]

        # vX = [vX[0][:bs, :, :], vX[1][:bs, :, :]]
        # vY = [vY[0][:bs, :], vY[1][:bs, :]]
        vX = vX[:bs, :, :]
        vY = vY[:bs, :]

        if model == False:
            model = self.createModel(batch_size=batch_size, stateful=True)
            model.compile(loss='mse', optimizer='adam')
            plot_model(model, show_shapes=True, to_file='model.png')

        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=40)
        mc = ModelCheckpoint(os.path.join(self.rootpath, "model.h5"),
            monitor='loss', mode='min', verbose=1, save_best_only=True)
        rs = LambdaCallback(on_epoch_end=lambda epoch, logs: model.reset_states())

        print("Shape of input data:{}".format(X.shape))
        # history = model.fit(X, Y, epochs=200, batch_size=batch_size, verbose=2, callbacks=[es, mc, rs], shuffle=False,
        #                     validation_data=(vX, vY))
        history = model.fit(X, Y, epochs=200, batch_size=batch_size, verbose=2, callbacks=[es, mc, rs], shuffle=False
                            )

        trainPredict = model.predict(X, batch_size = batch_size)
        # trainScore = [sqrt(x) for x in  mean_squared_error(np.hstack(Y), np.hstack(trainPredict), multioutput='raw_values')]
        trainScore = [sqrt(x) for x in
                      mean_squared_error(Y, trainPredict, multioutput='raw_values')]
        print("Train Score: {} RMSE".format(trainScore))
        return model

    def train(self, trainFunctions):
        shapesData = pkl.load(open('shapesData.p', 'rb'))
        model = False
        validationData = shapesData["elipse_1"]

        for name in trainFunctions:
            value = shapesData[name]
            print("Training State for {} shape".format(name))
            model = self.trainShape(value, validationData, model)

    def load(self):
        model_orig = load_model("model.h5")
        # re-define model
        model = self.createModel(batch_size=1, stateful=True)
        # Transfer learned weights
        model.set_weights(model_orig.get_weights())
        self.model = model
        self.scaler = pkl.load(open("scaler.p", "rb"))

    def predict(self, data):
        s, hessian, r_c_delta, r_delta, z_r = data
        # return s

        X, Y = self.processData([data], learn=False)
        scaled_predict = self.model.predict(X, batch_size = 1)
        if self.numFeaturesStateAdded > self.numFeaturesState:
            scaled_predict = np.hstack((scaled_predict, [np.ones(self.numFeaturesStateAdded-self.numFeaturesState)]))
        # if self.numFeaturesStateAdded > self.numFeaturesState:
        #     scaled_predict = np.hstack((scaled_predict[0], [[1]], scaled_predict[1]))
        # else:
        #     scaled_predict = np.hstack((scaled_predict[0], scaled_predict[1]))

        predict = self.scaler.inverse_transform(scaled_predict)[0]

        s_e = predict[0:self.numFeaturesState]
        # p_e = predict[self.numFeaturesStateAdded:]
        # p_e = np.reshape(p_e, (3, 3))

        return s_e

    def evaluate(self, functionNames):
        shapesData = pkl.load(open('shapesData.p', 'rb'))
        for name in functionNames:
            dataset = shapesData[name].to_numpy()
            trainPredict = []
            for x in dataset:
                s_e = self.predict(x)
                # s_e, p_e = x
                trainPredict.append([s_e])

            #Score
            dataset = dataset[1:] #Shift by one.
            trainPredict = trainPredict[:-1] # Cant score last prediction

            # dataset = [[*x[0], *x[1].flatten()] for x in dataset]
            # trainPredict = [[*x[0], *x[1][0], *x[1][1], *x[1][2]] for x in trainPredict]

            trainScore = [sqrt(x) for x in
                          mean_squared_error(dataset, trainPredict, multioutput='raw_values')]
            print("Evaluate Score for shape {}: {} RMSE".format(name, trainScore))

class Experiment:
    def __init__(self):
        pass
        # self.allShapes = [Shape(x) for x in self.params.functions if x.name in ['elipse']]

    def collect(self):
        shapesData = pd.Series(dtype='object')
        allNames = ['circle_1', 'circle_4_1', 'circle_6_1', 'elipse_1', 'irregular1_1',
             'irregular2_1', 'irregular1_9', 'circle_4_8',
             'elipse_6', 'irregular2_8', 'elipse_9', 'elipse_8', 'circle_3',
             'elipse_2', 'circle_4', 'elipse_5', 'circle_7', 'elipse_3',
             'irregular2_2', 'circle_6', 'irregular1_8', 'circle_10', 'circle_4_7',
             'circle_2', 'circle_6_8']
        allNames = ['elipse_1']
        fg = params.FunctionGenerator()
        allShapes = [Shape(fg.getFunction(name)) for name in allNames]
        for shape in allShapes:
            data, _ = shape.trace()
            shapesData = shapesData.append(pd.Series({shape.name: data}))
        rootpath = "/tmp" if os.name == 'posix' else ""
        pkl.dump(shapesData, open(os.path.join(rootpath, "shapesData.p"), "wb"))

    def train(self):
        shapesData = pkl.load(open("shapesData.p", "rb"))
        trainShapes = ['circle_4_1', 'circle_6_1', 'irregular1_1',
             'irregular2_1', 'irregular1_9', 'circle_4_8',
             'circle_3',
             'circle_4', 'circle_7',
             'irregular2_2', 'circle_6', 'irregular1_8', 'circle_10', 'circle_4_7',
             'circle_2', 'circle_6_8']
        trainShapes = ['elipse_1']
        model = MotionControlModel()
        model.train(trainShapes)

    def test(self):
        shapesData = pkl.load(open("shapesData.p", "rb"))
        testNames = ['circle_1', 'circle_4_1', 'circle_6_1', 'elipse_1', 'irregular1_1',
             'irregular2_1', 'irregular1_9', 'circle_4_8',
             'elipse_6', 'irregular2_8', 'elipse_9', 'elipse_8', 'circle_3',
             'elipse_2', 'circle_4', 'elipse_5', 'circle_7', 'elipse_3',
             'irregular2_2', 'circle_6', 'irregular1_8', 'circle_10', 'circle_4_7',
             'circle_2', 'circle_6_8']
        testNames = ['elipse_1']
        fg = params.FunctionGenerator()
        testShapes = [Shape(fg.getFunction(name)) for name in testNames]
        for shape in testShapes:
            df, refdf = shape.simulate()

            pltVar = PlotVariables(shape.name, ["z", "dz_x", "dz_y"])
            pltVar.set('z', refdf['s'].apply(lambda x: x[0]).to_numpy(),
                       df['s'].apply(lambda x: x[0]).to_numpy())
            pltVar.set('dz_x', refdf['s'].apply(lambda x: x[1]).to_numpy(),
                       df['s'].apply(lambda x: x[1]).to_numpy())
            pltVar.set('dz_y', refdf['s'].apply(lambda x: x[2]).to_numpy(),
                       df['s'].apply(lambda x: x[2]).to_numpy())
            pltVar.plot()

if True or __name__ == "__main__":
    np.random.seed(11)
    experiment = Experiment()
    if 'collect' in sys.argv:
        experiment.collect()
    if 'train'in sys.argv:
        experiment.train()
    if 'test'in sys.argv:
        experiment.test()
    if 'evaluate' in sys.argv:
        model = MotionControlModel(loadModel=True)
        shapesData = pkl.load(open("shapesData.p", "rb"))
        testNames = shapesData.keys()
        model.evaluate(testNames)
