#imports for the file
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import math

import tensorflow as tf
from tensorflow import keras as K
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
#----------------------------------------------import of the data set in the file-----------------------------------------
data_1 = pd.read_csv("archive/us_covid19_daily.csv")
data_2 = pd.read_csv("archive/us_counties_covid19_daily.csv")
data_3 = pd.read_csv("archive/us_states_covid19_daily.csv")

#-------------------------------------------------------------------------data set visualization--------------------------------

# print("data_1")
# print(data_1.head())
# print(data_1.describe())
# print("data_2")
# print(data_2.head())
# print(data_2.describe())
# print("data_3")
# print(data_3.head())
# print(data_3.describe())

pos = data_1[data_1['positive'] > 0]
pos = pos['positive']
pos = pos.loc[::-1].reset_index(drop=True)
pos = pos.to_numpy()
pos = pos.astype('float32')
pos = pos.reshape(-1, 1)
# print(pos)
# plt.plot(pos)
# plt.show()
scaler = MinMaxScaler(feature_range=(0,1))
pos = scaler.fit_transform(pos)

train_size = int(len(pos) * 0.75)
test_size = len(pos) - train_size
train, test = pos[0:train_size], pos[train_size:]
# print(len(train), len(test))

def create_dataset(df, look_back = 1):
    X, Y = [], []
    for i in range(len(df) - look_back - 1):
        temp = df[i:(i + look_back)]
        X.append(temp)
        Y.append(df[i + look_back])
    return np.array(X), np.array(Y)

look_back = 1
x_train, y_train = create_dataset(train, look_back)
x_test, y_test = create_dataset(test, look_back)
# print(x_train, y_train)
# print(x_train.shape)

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

print(x_train.shape)

model = Sequential()
model.add(LSTM(32, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size = 1, validation_data=(x_test, y_test))

trainPredict = model.predict(x_train)
testPredict = model.predict(x_test)

temp = np.expand_dims(pos, axis=1)
print(temp.shape)
fullPredict = model.predict(temp)

trainPredict = scaler.inverse_transform(trainPredict)
y_train = scaler.inverse_transform(y_train)
testPredict = scaler.inverse_transform(testPredict)
y_test = scaler.inverse_transform(y_test)
fullPredict = scaler.inverse_transform(fullPredict)

# print(y_train.shape, y_train)
# print(trainPredict.shape, trainPredict)
print("Train RMSE:", math.sqrt(mean_squared_error(y_train[:, 0], trainPredict[:, 0])))
print("Test RMSE:", math.sqrt(mean_squared_error(y_test[:, 0], testPredict[:, 0])))

print(pos.shape)
# shift train predictions for plotting
trainPredictPlot = np.empty_like(pos)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(pos)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(pos)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(pos))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.plot(fullPredict)
plt.show()