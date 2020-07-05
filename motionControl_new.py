import matplotlib
# matplotlib.use('Agg')
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
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
import os
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from utils import PlotVariable

var1 = PlotVariable("z_c")
var2 = PlotVariable("dz_c_x")
var3 = PlotVariable("dz_c_y")
var4 = PlotVariable("var4")

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
    return z_c, dz, p, [s_old, p_old, s_e, p_e]

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
            # var1.push(z_c_k, z_c_p)
            # var2.push(dz_c_k[0], dz_c_p[0])
            # var3.push(dz_c_k[1], dz_c_p[1])
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
        r_c_p, x_2_p, y_2_p = formationCenter(r_c, z_c, dz_c, hessian, x_2, y_2,
                                           self.function.mu_f, self.function.z_desired)
        var1.push(r_c[0], r_c_p[0])
        var2.push(r_c[1], r_c_p[1])
        var3.push(x_2[0], x_2_p[0])
        var4.push(x_2[1], x_2_p[1])

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

        dataframe = pd.concat([pd.DataFrame([i], columns=['s', 'p', 's_e', 'p_e']) for i in data], ignore_index=True)
        self.plot(r_c_plot, r_plot)
        return dataframe

    def simulate(self):
        model = Model(loadModel=True)
        self.trace(model)

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

class Experiment:
    def __init__(self):
        pass
        # self.allShapes = [Shape(x) for x in self.params.functions if x.name in ['elipse']]

    def collect(self):
        params = params.Params()
        allShapes = [Shape(x) for x in params.functions]
        shapesData = pd.Series(dtype='object')
        for shape in allShapes:
            data = shape.trace()
            shapesData = shapesData.append(pd.Series({shape.name: data}))
        pkl.dump(shapesData, open("shapesData.p", "wb"))

    def train(self):
        shapesData = pkl.load(open("shapesData.p", "rb"))
        trainShapes = [x for x in shapesData.keys() if x not in ['elipse_1']]
        model = Model()
        model.train(trainShapes)

    def test(self):
        shapesData = pkl.load(open("shapesData.p", "rb"))
        # testNames = [x for x in shapesData.keys() if x in ['elipse_1']]
        testNames = [x for x in shapesData.keys()]
        fg = params.FunctionGenerator()
        testShapes = [Shape(fg.getFunction(name)) for name in testNames]
        for shape in testShapes:
            shape.simulate()

class Model:
    def __init__(self, loadModel=False):

        self.rootpath = ""
        self.model = False
        self.scalerX = False
        self.scalerY = False
        if loadModel:
            self.load()

    def processData(self, df, scaleY=True):
        features = df.to_numpy()
        # s - 3x1
        # p - 3x3
        # s_e - 3x1
        # p_e - 3x3
        trainX = np.array([[x[0][0], x[0][1], x[0][2],
                              x[1][0][0], x[1][0][1], x[1][0][2], x[1][1][0], x[1][1][1], x[1][1][2], x[1][2][0], x[1][2][1], x[1][2][2]
                              ] for x in features])

        trainY = np.array([[x[2][0], x[2][1], x[2][2],
                              x[3][0][0], x[3][0][1], x[3][0][2], x[3][1][0], x[3][1][1], x[3][1][2], x[3][2][0], x[3][2][1], x[3][2][2]
                              ] for x in features])

        # pw = 1 # predict 1 step ahead: predictWindow
        # trainX[pw:][:] = trainX[:-pw][:]
        # trainX[0:pw][:] = trainX[pw][:]
        # trainY[0:pw][:] = trainY[pw][:]

        scaler = MinMaxScaler(feature_range=(-1, 1))
        trainX = scaler.fit_transform(trainX)
        pkl.dump(scaler, open("scalerX.p", "wb"))
        scaler = MinMaxScaler(feature_range=(-1, 1))
        trainY_scaled = scaler.fit_transform(trainY)
        pkl.dump(scaler, open("scalerY.p", "wb"))

        trainY_return = False
        if scaleY:
            trainY_return = trainY_scaled
        else:
            trainY_return = trainY

        return trainX, trainY_return

    def trainShape(self, dataset, model=False):
        trainX_c, trainY_c = self.processData(dataset)

        if model == False:
            model_state = False
            model_error = False
        else:
            model_state, model_error = model

        #model_state
        trainX = trainX_c[:,0:3]
        trainY = trainY_c[:,0:3]
        numFeaturesX = trainX.shape[1]
        numFeaturesY = trainY.shape[1]
        np.random.seed(7)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

        batch_size = 25
        if model_state == False:
            model_state = Sequential()
            numNeurons = numFeaturesX + 1
            model_state.add(LSTM(numNeurons, batch_input_shape=(batch_size, 1, numFeaturesX), stateful=True))
            model_state.add(Dense(numFeaturesY))
            model_state.compile(loss='mean_squared_error', optimizer='adam')

        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
        mc = ModelCheckpoint(os.path.join(self.rootpath, "model_state.h5"),
            monitor='loss', mode='min', verbose=1, save_best_only=True)
        rs = LambdaCallback(on_epoch_end=lambda epoch, logs: model_state.reset_states())
        history = model_state.fit(trainX, trainY, epochs=200, batch_size=batch_size, verbose=2, callbacks=[es, mc, rs], shuffle=False)

        trainPredict = model_state.predict(trainX, batch_size = batch_size)
        trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
        print("Train Score: {} RMSE".format(trainScore))

        # model_error
        trainX = trainX_c[:,3:]
        trainY = trainY_c[:,3:]
        numFeaturesX = trainX.shape[1]
        numFeaturesY = trainY.shape[1]
        np.random.seed(7)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

        batch_size = 25
        if model_error == False:
            model_error = Sequential()
            numNeurons = numFeaturesX + 1
            model_error.add(LSTM(numNeurons, batch_input_shape=(batch_size, 1, numFeaturesX), stateful=True))
            model_error.add(Dense(numFeaturesY))
            model_error.compile(loss='mean_squared_error', optimizer='adam')

        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
        mc = ModelCheckpoint(os.path.join(self.rootpath, "model_error.h5"),
            monitor='loss', mode='min', verbose=1, save_best_only=True)
        rs = LambdaCallback(on_epoch_end=lambda epoch, logs: model_error.reset_states())
        history = model_error.fit(trainX, trainY, epochs=200, batch_size=batch_size, verbose=2, callbacks=[es, mc, rs], shuffle=False)

        trainPredict = model_error.predict(trainX, batch_size = batch_size)
        trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
        print("Train Score: {} RMSE".format(trainScore))

        model = [model_state, model_error]

        return model

    def train(self, trainFunctions):
        shapesData = pkl.load(open('shapesData.p', 'rb'))
        model = False

        for name in trainFunctions:
            value = shapesData[name]
            print("Training State for {} shape".format(name))
            model = self.trainShape(value, model)

    def load(self):
        model_orig = load_model("model_state.h5")
        batch_size = 1
        # re-define model
        model = Sequential()
        model.add(LSTM(model_orig.layers[0].units, batch_input_shape=(batch_size, 1, model_orig.input_shape[2]), stateful=True))
        model.add(Dense(model_orig.layers[1].units))
        model.set_weights(model_orig.get_weights())
        # compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        self.model_state = model

        model_orig = load_model("model_error.h5")
        batch_size = 1
        # re-define model
        model = Sequential()
        model.add(LSTM(model_orig.layers[0].units, batch_input_shape=(batch_size, 1, model_orig.input_shape[2]), stateful=True))
        model.add(Dense(model_orig.layers[1].units))
        model.set_weights(model_orig.get_weights())
        # compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        self.model_error = model

        self.scalerX = pkl.load(open("scalerX.p", "rb"))
        self.scalerY = pkl.load(open("scalerY.p", "rb"))

    def predict(self, data):
        s, p = data
        data = np.array([[*s, *np.reshape(p, (9))]])
        data = self.scalerX.transform(data)

        testX = data[:,0:3]
        s_e = self.model_state.predict(np.reshape(testX, (testX.shape[0], 1, testX.shape[1])),
                    batch_size = 1)[0]

        testX = data[:,3:]
        p_e = self.model_error.predict(np.reshape(testX, (testX.shape[0], 1, testX.shape[1])),
                                         batch_size=1)[0]

        data_e = [[*s_e, *p_e]]
        data_e = self.scalerY.inverse_transform(data_e)[0]
        s_e = data_e[0:3]
        p_e = data_e[3:]
        p_e = np.reshape(p_e, (3, 3))

        return s_e, p_e

    def evaluate(self, functionNames):
        shapesData = pkl.load(open('shapesData.p', 'rb'))
        for name in functionNames:
            dataset = shapesData[name]


            trainX, trainY = self.processData(dataset, scaleY=False)


            np.random.seed(7)

            trainPredict = []
            for x in trainX:
                s = x[0:3]
                p = np.reshape(x[3:], (3,3))
                s_e, p_e = self.predict([s,p])
                trainPredict.append([*s_e,*np.reshape(p_e, (9))])


            trainPredict = np.array(trainPredict)

            trainScore = math.sqrt(mean_squared_error(trainY, trainPredict_invScaled))
            print("Evaluate Score for shape {}: {} RMSE".format(name, trainScore))


if True or __name__ == "__main__":
    experiment = Experiment()
    # experiment.collect()
    # experiment.train()
    experiment.test()
    # var1.plot()
    # var2.plot()
    # var3.plot()
    # var4.plot()
    # model = Model(loadModel=True)
    # model.evaluate(["elipse"])



