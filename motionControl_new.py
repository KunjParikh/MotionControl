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

from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
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


def kalmanFilter(z_c, dz, r, z_r, r_c, r_c_old, p, hessian, model = False):

    px = params.Params()
    numSensors = px.numSensors

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
    c = np.hstack((np.ones((numSensors, 1)), r - r_c))  # 4x3
    d = 0.5 * np.vstack([np.kron(pt - r_c, pt - r_c) for pt in r])  # 4x4


    if model:
        s_e, p_e = model.predict([s,p])
        if (p_e == np.zeros((3,3))).flatten().all():
            p_e = m

    else:
        s_e = a @ s + h  # 3x1
        p_e = a @ p @ a.T + m  # 3x3 @ 3x1 @ 3x3 = 3x3

    s_old = s
    p_old = p
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
    # Use actual s and p value - not the one predicted by eq1 and eq2
    # return only s and p values for data here, lag values etc are done in training code.
    return z_c, dz, p, [s, p]

# !ls '/content/drive/My Drive/SJSU/Final Project/model_state.h5'
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

    def step(self, state, model=False):
        r_c, z_c, dz_c, r_c_old, p, r, q, dq, u_r, vel_q, x_2, y_2 = state

        #Inputs
        z_r = np.array([self.function.f(*pt) for pt in r]) #Mimick measurements on robots
        hessian = self.function.hessian_f(r_c[0], r_c[1]) #True value

        if model:
            z_c_k, dz_c_k, p_k, data = kalmanFilter(z_c, dz_c, r, z_r, r_c, r_c_old, p, hessian)
            z_c_p, dz_c_p, p_p, data = kalmanFilter(z_c, dz_c, r, z_r, r_c, r_c_old, p, hessian, model)

            z_c = z_c_p
            dz_c = dz_c_p
            p = p_p

        else:
            z_c, dz_c, p, data = kalmanFilter(z_c, dz_c, r, z_r, r_c, r_c_old, p, hessian)
        # z_c = f(r_c[0], r_c[1])
        # dz_c = dz_f(r_c[0], r_c[1])

        r_c_old = r_c
        r_c, x_2, y_2 = formationCenter(r_c, z_c, dz_c, hessian, x_2, y_2,
                                           self.function.mu_f, self.function.z_desired)

        r, q, dq, u_r, vel_q = formationControl(r_c, r, q, dq, u_r, vel_q)

        state = [r_c, z_c, dz_c, r_c_old, p, r, q, dq, u_r, vel_q, x_2, y_2]
        # data = [r_c, z_c, dz_c, hessian, r, z_r, p]

        return state, data


    def trace(self, model=False):
        print("Tracing shape for {}".format(self.name))
        data = []
        r_c_plot = [self.r_c]
        r_plot = [self.r]
        p_det = []

        state = [self.r_c, self.z_c, self.dz_c, self.r_c_old, self.p, self.r,
                 self.q, self.dq, self.u_r, self.vel_q, self.x_2, self.y_2]

        for i in range(10000):
            state, d = self.step(state, model)
            data.append(d)
            r_c_plot.append(state[0])
            p_det.append(np.linalg.det(state[4]))
            if i % 150 == 0:
                r_plot.append(state[5])
        dataframe = pd.DataFrame(data, columns=['s', 'p'])

        if model:
            cmpData = []
            state = [self.r_c, self.z_c, self.dz_c, self.r_c_old, self.p, self.r,
                     self.q, self.dq, self.u_r, self.vel_q, self.x_2, self.y_2]
            for i in range(10000):
                state, d = self.step(state)
                cmpData.append(d)
            cmpDataframe = pd.DataFrame(cmpData, columns=['s', 'p'])
        else:
            cmpDataframe = False

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
        for r in r_plot:
            plt.plot([r[0, 0], r[1, 0]], [r[0, 1], r[1, 1]], 'yo-')
            plt.plot([r[2, 0], r[3, 0]], [r[2, 1], r[3, 1]], 'go-')
        plt.title(self.name)
        plt.savefig("{}.pdf".format(self.name), bbox_inches='tight')
        plt.close()


class MotionControlModel:
    def __init__(self, loadModel=False):

        self.rootpath = "/tmp" if os.name=='posix' else ""
        self.model = False
        self.scalerX = False
        self.scalerY = False

        self.numFeaturesStateAdded = 3
        self.numFeaturesState = 3
        self.numFeaturesError = 9
        self.lagWindowSize = 50
        self.history = np.zeros((self.lagWindowSize, self.numFeaturesStateAdded+self.numFeaturesError)).tolist()

        self.lstmStateSize = 40
        self.lstmErrorSize = 20

        if loadModel:
            self.load()

    def processData(self, df, learn=True):
        if isinstance(df, pd.DataFrame):
            raw_data = df.to_numpy()
        else:
            raw_data = df
        unscaled_data = np.array([[x[0][0], x[0][1], x[0][2],
                            x[1][0][0], x[1][0][1], x[1][0][2], x[1][1][0], x[1][1][1], x[1][1][2], x[1][2][0],
                            x[1][2][1], x[1][2][2]
                            ] for x in raw_data])

        lagWindowSize = self.lagWindowSize
        numFeaturesState = self.numFeaturesState
        numFeaturesStateAdded = self.numFeaturesStateAdded

        # if learn:
        #     #Skip data while not on shape
        #     # calculate summary statistics
        #     data = unscaled_data[:,0]
        #     data_mean, data_std = np.mean(data), np.std(data)
        #     # identify outliers
        #     cut_off = data_std * 3
        #     lower, upper = data_mean - cut_off, data_mean + cut_off
        #     # identify outliers
        #     onShape = 0
        #     for value in data:
        #         if value < lower or value > upper:
        #             onShape = onShape + 1
        #         else:
        #             break
        #     # Instead of removing, add a parameter - holiday effect.
        #     # unscaled_data = unscaled_data[onShape:,:]
        #     datalength = unscaled_data.shape[0]
        #     onShape = np.array([*np.zeros((onShape)), *np.ones((datalength-onShape))])
        #     onShape = np.reshape(onShape, (datalength, 1))
        #     unscaled_data = np.hstack((unscaled_data[:,0:numFeaturesState], onShape, unscaled_data[:,numFeaturesState:]))
        # else:
        #     onShape = np.array([[1]])
        #     unscaled_data = np.hstack(
        #         (unscaled_data[:, 0:numFeaturesState], onShape, unscaled_data[:, numFeaturesState:]))

        datalength = unscaled_data.shape[0]

        if learn:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_data = scaler.fit_transform(unscaled_data)
            pkl.dump(scaler, open("scaler.p", "wb"))
        else:
            self.history.append(unscaled_data[0])
            self.history = self.history[1:]
            scaled_data = self.scaler.transform(self.history)

        # LSTM needs input in [samples, timestep, features] format.
        if learn:
            X = np.array([scaled_data[i-lagWindowSize:i,:] for i in range(lagWindowSize, datalength)])
            Y = scaled_data[lagWindowSize:,:]
            X = [X[:, :, 0:numFeaturesStateAdded], X[:, :, numFeaturesStateAdded:]]
            # We dont need 4th parameter for prediction so -1
            Y = [Y[:, 0:numFeaturesState], Y[:, numFeaturesStateAdded:]]
        else:
            X = np.array([scaled_data])
            X = [X[:, :, 0:numFeaturesStateAdded], X[:, :, numFeaturesStateAdded:]]
            Y = False

        return X, Y

    def trainShape(self, dataset, validationData, model=False):
        X, Y = self.processData(dataset)
        vX, vY = self.processData(validationData)

        numFeaturesState = self.numFeaturesState
        numFeaturesStateAdded = self.numFeaturesStateAdded
        numFeaturesError = self.numFeaturesError

        batch_size = 60
        bs = floor(X[0].shape[0] / batch_size) * batch_size
        X = [X[0][:bs,:,:], X[1][:bs,:,:]]
        Y = [Y[0][:bs, :], Y[1][:bs, :]]

        vX = [vX[0][:bs, :, :], vX[1][:bs, :, :]]
        vY = [vY[0][:bs, :], vY[1][:bs, :]]

        lagWindowSize = self.lagWindowSize
        if model == False:
            input1 = Input(batch_shape=(batch_size, lagWindowSize, numFeaturesStateAdded))
            lstm1 = LSTM(self.lstmStateSize, stateful=False)(input1)
            output1 = Dense(numFeaturesState, name='dense_state')(lstm1)

            input2 = Input(batch_shape=(batch_size, lagWindowSize, numFeaturesError))
            lstm2 = LSTM(self.lstmErrorSize, stateful=False)(input2)
            output2 = Dense(numFeaturesError, name='dense_error')(lstm2)

            model = Model(inputs=[input1, input2], outputs=[output1, output2])
            model.compile(loss='mse', optimizer='adam')
            plot_model(model, show_shapes=True, to_file='model.png')

        es = EarlyStopping(monitor='val_dense_state_loss', mode='min', verbose=1, patience=40)
        mc = ModelCheckpoint(os.path.join(self.rootpath, "model.h5"),
            monitor='val_dense_state_loss', mode='min', verbose=1, save_best_only=True)
        rs = LambdaCallback(on_epoch_end=lambda epoch, logs: model.reset_states())
        history = model.fit(X, Y, epochs=200, batch_size=batch_size, verbose=2, callbacks=[es, mc, rs], shuffle=False,
                            validation_data=(vX, vY))

        trainPredict = model.predict(X, batch_size = batch_size)
        trainScore = [sqrt(x) for x in  mean_squared_error(np.hstack(Y), np.hstack(trainPredict), multioutput='raw_values')]
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
        batch_size = 1
        lagWindowSize = self.lagWindowSize
        numFeaturesState = self.numFeaturesState
        numFeaturesStateAdded = self.numFeaturesStateAdded
        numFeaturesError = self.numFeaturesError
        # re-define model
        input1 = Input(batch_shape=(batch_size, lagWindowSize, numFeaturesStateAdded))
        lstm1 = LSTM(self.lstmStateSize, stateful=True)(input1)
        output1 = Dense(numFeaturesState)(lstm1)

        input2 = Input(batch_shape=(batch_size, lagWindowSize, numFeaturesError))
        lstm2 = LSTM(self.lstmErrorSize, stateful=True)(input2)
        output2 = Dense(numFeaturesError)(lstm2)

        model = Model(inputs=[input1, input2], outputs=[output1, output2])
        # Transfer learned weights
        model.set_weights(model_orig.get_weights())
        model.compile(loss='mse', optimizer='adam')

        self.model = model
        self.scaler = pkl.load(open("scaler.p", "rb"))


    def predict(self, data):
        s, p = data
        # return s, p

        X, Y = self.processData([data], learn=False)
        scaled_predict = self.model.predict(X, batch_size = 1)
        if self.numFeaturesStateAdded > self.numFeaturesState:
            scaled_predict = np.hstack((scaled_predict[0], [[1]], scaled_predict[1]))
        else:
            scaled_predict = np.hstack((scaled_predict[0], scaled_predict[1]))
        predict = self.scaler.inverse_transform(scaled_predict)[0]

        s_e = predict[0:self.numFeaturesState]
        p_e = predict[self.numFeaturesStateAdded:]
        p_e = np.reshape(p_e, (3, 3))

        return s_e, p_e

    def evaluate(self, functionNames):
        shapesData = pkl.load(open('shapesData.p', 'rb'))
        for name in functionNames:
            dataset = shapesData[name].to_numpy()
            trainPredict = []
            for x in dataset:
                s_e, p_e = self.predict(x)
                # s_e, p_e = x
                trainPredict.append([s_e, p_e])

            #Score
            dataset = dataset[1:] #Shift by one.
            trainPredict = trainPredict[:-1] # Cant score last prediction

            dataset = [[*x[0], *x[1].flatten()] for x in dataset]
            trainPredict = [[*x[0], *x[1][0], *x[1][1], *x[1][2]] for x in trainPredict]

            trainScore = [sqrt(x) for x in
                          mean_squared_error(dataset, trainPredict, multioutput='raw_values')]
            print("Evaluate Score for shape {}: {} RMSE".format(name, trainScore))

class Experiment:
    def __init__(self):
        pass
        # self.allShapes = [Shape(x) for x in self.params.functions if x.name in ['elipse']]

    def collect(self):
        p = params.Params()
        allShapes = [Shape(x) for x in p.functions]
        shapesData = pd.Series(dtype='object')
        for shape in allShapes:
            data, _ = shape.trace()
            shapesData = shapesData.append(pd.Series({shape.name: data}))
        pkl.dump(shapesData, open("shapesData.p", "wb"))

    def train(self):
        shapesData = pkl.load(open("shapesData.p", "rb"))
        # trainShapes = [x for x in shapesData.keys() if
        #                x in ['circle_4_1', 'circle_6_1', 'elipse_1', 'irregular1_1', 'irregular2_1'
        #                      ]]
        trainShapes = ['irregular2_1', 'irregular1_1', 'irregular1_8', 'circle_6_1', 'circle_4_1'
                             ]
        # trainShapes = [x for x in shapesData.keys() if x in ['elipse_1']]
        model = MotionControlModel()
        model.train(trainShapes)

    def test(self):
        shapesData = pkl.load(open("shapesData.p", "rb"))
        testNames = [x for x in shapesData.keys() if
                       x in ['circle_1', 'irregular2_1', 'elipse_1'
                             ]]
        # testNames = [x for x in shapesData.keys() if x in ['elipse_1']]
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
