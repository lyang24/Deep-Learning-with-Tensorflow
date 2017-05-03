from __future__ import print_function
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import  train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

# to not display the warnings of tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

prices_dataset =  pd.read_csv(r'C:\Users\lYang4\Desktop\prices.csv', header=0)
#choose priceline
prices_dataset.close.nlargest(20)
prices_dataset.iloc[[833115]]

priceline = prices_dataset[prices_dataset['symbol'] == 'PCLN']

priceline_prices = priceline.close.values.astype('float32')

priceline_prices = priceline_prices.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))
priceline_prices = scaler.fit_transform(priceline_prices)

train_size = int(len(priceline_prices) * 0.7)
test_size = len(priceline_prices) - train_size

train, test = priceline_prices[0:train_size,:], priceline_prices[train_size:len(priceline_prices),:]

look_back = 7
epochs = 1000
batch_size = 32

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=epochs, batch_size=batch_size)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

trainPredictPlot = np.empty_like(priceline_prices)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict


testPredictPlot = np.empty_like(priceline_prices)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(priceline_prices)-1, :] = testPredict

plt.plot(scaler.inverse_transform(priceline_prices))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()