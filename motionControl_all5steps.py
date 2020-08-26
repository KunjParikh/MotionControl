import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import params as params
from formationCenter import formationCenter
from formationControl import formationControl
from kalmanFilter import kalmanFilter
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
            z_c_k, dz_c_k, p_k = kalmanFilter(z_c, dz_c, r, z_r, r_c, r_c_old, p, hessian)
            z_c_p, dz_c_p, p_p = model.predict(z_c, dz_c, r, z_r, r_c, r_c_old, p, hessian)
            # var1.push(z_c_k, z_c_p)
            # var2.push(dz_c_k[0], dz_c_p[0])
            # var3.push(dz_c_k[1], dz_c_p[1])
            z_c = z_c_k
            dz_c = dz_c_k
            p = p_k
        else:
            z_c, dz_c, p = kalmanFilter(z_c, dz_c, r, z_r, r_c, r_c_old, p, hessian)
        # z_c = f(r_c[0], r_c[1])
        # dz_c = dz_f(r_c[0], r_c[1])

        r_c_old = r_c
        r_c, x_2, y_2 = formationCenter(r_c, z_c, dz_c, hessian, x_2, y_2,
                                           self.function.mu_f, self.function.z_desired)
        r_c_p, x_2_p, y_2_p = formationCenter(r_c, z_c_p, dz_c_p, hessian, x_2, y_2,
                                           self.function.mu_f, self.function.z_desired)
        var1.push(r_c[0], r_c_p[0])
        var2.push(r_c[1], r_c_p[1])
        var3.push(x_2[0], x_2_p[0])
        var4.push(x_2[1], x_2_p[1])

        r, q, dq, u_r, vel_q = formationControl(r_c, r, q, dq, u_r, vel_q)

        state = [r_c, z_c, dz_c, r_c_old, p, r, q, dq, u_r, vel_q, x_2, y_2]
        data = [r_c, z_c, dz_c, hessian, r, z_r, p]

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

        dataframe = pd.concat([pd.DataFrame([i], columns=['r_c', 'z_c', 'dz_c', 'hessian', 'r', 'z_r', 'p']) for i in data], ignore_index=True)
        self.plot(r_c_plot, r_plot)
        return dataframe

    def simulate(self):
        model = Model("model.h5")
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
        self.params = params.Params()
        self.allShapes = [Shape(x) for x in self.params.functions]
        self.trainShapes = [x for x in self.allShapes if x.name in ['elipse']]
        self.testShapes = [x for x in self.allShapes if x.name in ['elipse']]

    def collect(self):
        shapesData = pd.Series(dtype='object')
        for shape in self.allShapes:
            data = shape.trace()
            shapesData = shapesData.append(pd.Series({shape.name: data}))
        pkl.dump(shapesData, open("shapesData.p", "wb"))

    def train(self):
        model = Model()
        model.train([x.function for x in self.trainShapes])

    def test(self):
        for shape in self.testShapes:
            shape.simulate()

class Model:
    def __init__(self, file=False):

        self.rootpath = ""
        self.model = False
        self.scalerX = False
        self.scalerY = False
        if file:
            self.load(file)

    def processData(self, df, scaleY=True):
        features = df.to_numpy()
        # r_c - 2x1
        # z_c - 1 > pred
        # dz_c - 2x1 > pred
        # hessian - 2x2
        # r - 4x2
        # z_r - 4x1
        # p - 3x3 > pred
        trainX = np.array([[x[0][0], x[0][0],
                              x[1],
                              x[2][0], x[2][1],
                              x[3][0][0], x[3][0][1], x[3][1][0], x[3][1][1],
                              x[4][0][0], x[4][0][1], x[4][1][0], x[4][1][1], x[4][2][0], x[4][2][1], x[4][3][0], x[4][3][1],
                              x[5][0], x[5][1], x[5][2], x[5][3],
                              x[6][0][0], x[6][0][1], x[6][0][2], x[6][1][0], x[6][1][1], x[6][1][2], x[6][2][0], x[6][2][1], x[6][2][2]
                              ] for x in features])

        trainY = np.array([[x[1],
                            x[2][0], x[2][1],
                            x[6][0][0], x[6][0][1], x[6][0][2], x[6][1][0], x[6][1][1], x[6][1][2], x[6][2][0], x[6][2][1], x[6][2][2]
                            ] for x in features])

        pw = 1 # predict 1 step ahead: predictWindow
        trainX[pw:][:] = trainX[:-pw][:]
        trainX[0:pw][:] = trainX[pw][:]
        trainY[0:pw][:] = trainY[pw][:]

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
        trainX, trainY = self.processData(dataset)
        numFeaturesX = trainX.shape[1]
        numFeaturesY = trainY.shape[1]
        np.random.seed(7)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

        batch_size = 25
        if model == False:
            model = Sequential()
            numNeurons = numFeaturesX + 1
            model.add(LSTM(numNeurons, batch_input_shape=(batch_size, 1, numFeaturesX), stateful=True))
            model.add(Dense(numFeaturesY))
            model.compile(loss='mean_squared_error', optimizer='adam')

        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
        mc = ModelCheckpoint(os.path.join(self.rootpath, "model.h5"),
            monitor='loss', mode='min', verbose=1, save_best_only=True)
        rs = LambdaCallback(on_epoch_end=lambda epoch, logs: model.reset_states())
        history = model.fit(trainX, trainY, epochs=200, batch_size=batch_size, verbose=2, callbacks=[es, mc, rs], shuffle=False)

        trainPredict = model.predict(trainX, batch_size = batch_size)
        trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
        print("Train Score: {} RMSE".format(trainScore))
        return model

    def train(self, trainFunctions):
        shapesData = pkl.load(open('shapesData.p', 'rb'))
        model = False

        for function in trainFunctions:
            name = function.name
            value = shapesData[name]
            print("Training State for {} shape".format(name))
            model = self.trainShape(value, model)

    def load(self, file):
        model_orig = load_model(file)
        batch_size = 1
        # re-define model
        model = Sequential()
        model.add(LSTM(model_orig.layers[0].units, batch_input_shape=(batch_size, 1, model_orig.input_shape[2]), stateful=True))
        model.add(Dense(model_orig.layers[1].units))
        model.set_weights(model_orig.get_weights())
        # compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = model
        self.scalerX = pkl.load(open("scalerX.p", "rb"))
        self.scalerY = pkl.load(open("scalerY.p", "rb"))

    def predict(self, z_c, dz_c, r, z_r, r_c, r_c_old, p, hessian):
        state = [[r_c, z_c, dz_c, hessian, r, z_r, p]]
        testX = np.array([[x[0][0], x[0][0],
                              x[1],
                              x[2][0], x[2][1],
                              x[3][0][0], x[3][0][1], x[3][1][0], x[3][1][1],
                              x[4][0][0], x[4][0][1], x[4][1][0], x[4][1][1], x[4][2][0], x[4][2][1], x[4][3][0], x[4][3][1],
                              x[5][0], x[5][1], x[5][2], x[5][3],
                              x[6][0][0], x[6][0][1], x[6][0][2], x[6][1][0], x[6][1][1], x[6][1][2], x[6][2][0], x[6][2][1], x[6][2][2]
                              ] for x in state])
        testX = self.scalerX.transform(testX)

        testY = self.model.predict(np.reshape(testX, (testX.shape[0], 1, testX.shape[1])),
                                            batch_size=1)
        testY = self.scalerY.inverse_transform(testY)[0]

        z_c = testY[0]
        dz_c = testY[1:3]
        p = np.reshape(testY[3:], (3, 3))
        return z_c, dz_c, p

    def evaluate(self, functionNames):
        shapesData = pkl.load(open('shapesData.p', 'rb'))
        for name in functionNames:
            dataset = shapesData[name]

            # plt.plot([x[1] for x in dataset.to_numpy()])
            # plt.title('z_c')
            # plt.show()
            #
            # plt.plot([x[2][0] for x in dataset.to_numpy()])
            # plt.title('dz_c_x')
            # plt.show()
            #
            # plt.plot([x[2][1] for x in dataset.to_numpy()])
            # plt.title('dz_c_y')
            # plt.show()

            trainX, trainY = self.processData(dataset, scaleY=False)

            # plt.plot([x[2] for x in trainX])
            # plt.title('z_c')
            # plt.show()
            #
            # plt.plot([x[3] for x in trainX])
            # plt.title('dz_c_x')
            # plt.show()
            #
            # plt.plot([x[4] for x in trainX])
            # plt.title('dz_c_y')
            # plt.show()

            np.random.seed(7)

            trainPredict = []
            state = [0] * trainX.shape[1]
            state[2:5] = trainX[0][2:5]
            for x in trainX:
                state[0:2] = x[0:2]
                state[5:] = x[5:]
                y = np.array([state])
                y = np.reshape(y, (y.shape[0], 1, y.shape[1]))
                predY = self.model.predict(y, batch_size=1)[0]

                state[2:5] = predY[2:5]
                trainPredict.append(predY)


            trainPredict = np.array(trainPredict)
            # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            # batch_size = 1
            # trainPredict = self.model.predict(trainX, batch_size=batch_size)
            trainPredict_invScaled = self.scalerY.inverse_transform(trainPredict)


            plt.plot([x[0] for x in trainPredict_invScaled])
            plt.title('z_c')
            plt.show()

            plt.plot([x[1] for x in trainPredict_invScaled])
            plt.title('dz_c_x')
            plt.show()

            plt.plot([x[2] for x in trainPredict_invScaled])
            plt.title('dz_c_y')
            plt.show()

            trainScore = math.sqrt(mean_squared_error(trainY, trainPredict_invScaled))
            print("Evaluate Score for shape {}: {} RMSE".format(name, trainScore))

if True or __name__ == "__main__":
    experiment = Experiment()
    # experiment.collect()
    # experiment.train()
    experiment.test()
    var1.plot()
    var2.plot()
    var3.plot()
    var4.plot()
    # model = Model("model.h5")
    # model.evaluate(["elipse"])

