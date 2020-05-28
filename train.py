# LSTM for international airline passengers problem with regression framing
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
import os
import pandas as pd
import params as Params
#Comment this to use GPU. But looks like for me CPU (~10s) is faster than GPU (~80s).
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
rootpath = ""

def scale_data(dataset, trainFunctions):
    params = Params.Params()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    training_functions = [function.name for function in trainFunctions]
    #fit_data = pd.concat([dataset[name] for name in training_functions])
    #scaler = scaler.fit(fit_data)
    #for name in training_functions:
    #    dataset[name] = pd.DataFrame(scaler.transform(dataset[name]))
    scaler_return = False
    for name in training_functions:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # print(dataset[name].describe(include='all'))
        dataset[name] = pd.DataFrame(scaler.fit_transform(dataset[name]))
        # print(dataset[name].describe(include='all'))
        # Returning ellipse's scaler as it works better than rhombus in general.
        if name == "elipse":
            scaler_return = scaler
    return dataset, scaler_return

def train_lstm(dataset, name, model=False):
    # fix random seed for reproducibility
    np.random.seed(7)
    # load the dataset
    #dataset =  df_state.to_numpy()
    d_shape = dataset.shape
    nfeatures = round(d_shape[1] / 2)
    # normalize the dataset

    # split into train and test sets
    # train_size = int(len(dataset) * 0.67)
    # test_size = len(dataset) - train_size
    # train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    train = dataset
    # reshape into X=t and Y=t+1
    trainX, trainY = train[:,0:nfeatures], train[:,nfeatures:]
    # testX, testY = test[:, 0:nfeatures], test[:,nfeatures:]
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    batch_size = 25
    if model == False:
        model = Sequential()
        model.add(LSTM(nfeatures+1, batch_input_shape=(batch_size, 1, nfeatures), stateful=True))
        model.add(Dense(nfeatures))
        model.compile(loss='mean_squared_error', optimizer='adam')

    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(os.path.join(rootpath, "model_{}.h5".format(name)),
        monitor='loss', mode='min', verbose=1, save_best_only=True)
    rs = LambdaCallback(on_epoch_end=lambda epoch, logs: model.reset_states())
    history = model.fit(trainX, trainY, epochs=200, batch_size=batch_size, verbose=2, callbacks=[es, mc, rs], shuffle=False)

    # make predictions
    trainPredict = model.predict(trainX, batch_size = batch_size)

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    print("Train Score: {} RMSE".format(trainScore))
    # testScore = math.sqrt(mean_squared_error(testY, testPredict))
    # print("Test Score: {} RMSE".format(testScore))
    return model

def experiment():
    params = Params.Params()
    df_error = pickle.load(open('df_error.p', 'rb'))
    df_state = pickle.load(open('df_state.p', 'rb'))
    model = False

    trainFunctions = [x for x in params.functions if x.name not in ['circle_4']]
    #trainFunctions = [x for x in params.functions if x.name in ['elipse']]

    df_state, scaler_state = scale_data(df_state, trainFunctions)
    df_error, scaler_error = scale_data(df_error, trainFunctions)
    pickle.dump(scaler_state, open("scaler_state.p", "wb"))
    pickle.dump(scaler_error, open("scaler_error.p", "wb"))

    for function in trainFunctions:
        name = function.name
        value = df_state[name]
        print("Training State for {} shape".format(name))
        model = train_lstm(value.to_numpy(), 'state', model)

    model = False
    for function in trainFunctions:
        name = function.name
        value = df_error[name]
        print("Training Error for {} shape".format(name))
        model = train_lstm(value.to_numpy(), 'error', model)

#Uncomment to retrain.
if __name__ == "__main__":
    experiment()



