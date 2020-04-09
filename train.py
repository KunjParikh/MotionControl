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
import os
import pandas as pd
#Comment this to use GPU. But looks like for me CPU (~10s) is faster than GPU (~80s).
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
rootpath = "/tmp"

def train_lstm(dataset, name, model=False):
  # fix random seed for reproducibility
  np.random.seed(7)
  # load the dataset
  #dataset =  df_state.to_numpy()
  d_shape = dataset.shape
  nfeatures = round(d_shape[1] / 2)
  # normalize the dataset
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)
  # split into train and test sets
  train_size = int(len(dataset) * 0.67)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  # reshape into X=t and Y=t+1
  trainX, trainY = train[:,0:nfeatures], train[:,nfeatures:]
  testX, testY = test[:, 0:nfeatures], test[:,nfeatures:]
  # reshape input to be [samples, time steps, features]
  trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
  testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

  # create and fit the LSTM network
  if model == False:
    model = Sequential()
    model.add(LSTM(nfeatures+1, input_shape=(1, nfeatures)))
    model.add(Dense(nfeatures))
    model.compile(loss='mean_squared_error', optimizer='adam')

  es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
  mc = ModelCheckpoint(os.path.join(rootpath, "model_{}.h5".format(name)), 
         monitor='loss', mode='min', verbose=1, save_best_only=True)
  history = model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, callbacks=[es, mc])

  # make predictions
  trainPredict = model.predict(trainX)
  testPredict = model.predict(testX)
  # invert predictions
  trainPredict = scaler.inverse_transform(np.hstack((trainPredict, np.zeros(trainPredict.shape))))[:,0:nfeatures]
  trainY = scaler.inverse_transform(np.hstack((trainY, np.zeros(trainY.shape))))[:,0:nfeatures]
  testPredict = scaler.inverse_transform(np.hstack((testPredict, np.zeros(testPredict.shape))))[:,0:nfeatures]
  testY = scaler.inverse_transform(np.hstack((testY, np.zeros(testY.shape))))[:,0:nfeatures]

  # calculate root mean squared error
  trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
  print("Train Score: {} RMSE".format(trainScore))
  testScore = math.sqrt(mean_squared_error(testY, testPredict))
  print("Test Score: {} RMSE".format(testScore))
  return scaler, model

#Uncomment to retrain.
if __name__ == "__main__":
    df_error = pickle.load( open( 'df_error.p', 'rb'))
    df_state = pickle.load( open( 'df_state.p', 'rb'))
    model = False
    scaler = pd.Series(dtype='object')
    for name, value in df_state.items():
      print("Training State for {} shape".format(name))
      s, model = train_lstm(value.to_numpy(), 'state', model)
      scaler = scaler.append(pd.Series({name: s}))
    pickle.dump(scaler, open("scaler_state.p", "wb"))

    model = False
    scaler = pd.Series(dtype='object')
    for name, value in df_error.items():
      print("Training Error for {} shape".format(name))
      s, model = train_lstm(value.to_numpy(), 'error', model)
      scaler = scaler.append(pd.Series({name: s}))
    pickle.dump(scaler, open("scaler_error.p", "wb"))
