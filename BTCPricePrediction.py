import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from tensorflow._api.v1 import train
import numpy as np
fd = pd.read_csv('BTC-USD_latest.csv')

print(fd['Close'])

# fd['Close'].plot()
data = fd.iloc[:, 5:6]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaler = scaler.fit_transform(data)
# training_set = data_scaler[:600]
# test_set = data_scaler[601:]

X_train = []
Y_train = []

for i in range(50, 600):
    X_train.append(data_scaler[i-50:i, 0])
    Y_train.append(data_scaler[i])

X_train = np.array(X_train)
Y_train = np.array(Y_train)
# X_train = training_set[0: len(training_set)-1]
# Y_train = training_set[1: len(training_set)]
#

X_test = []
Y_test = []
for i in range(600, len(data_scaler)):
    X_test.append(data_scaler[i-50:i, 0])
    Y_test.append(data_scaler[i])

X_test = np.array(X_test)
Y_test = np.array(Y_test)

# X_test = test_set[0: len(test_set)-1]
# Y_test = test_set[1: len(test_set)]


#
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()
model.add(LSTM(20, return_sequences=True,  input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dropout(.1))
model.add(LSTM(40))
# model.add(Dropout(.2))
# model.add(LSTM(60, return_sequences=True))
# model.add(Dropout(.2))
# model.add(LSTM(50))
model.add(Dense(units=1))


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy', 'categorical_accuracy'])
model.summary()
history = model.fit(X_train, Y_train, epochs=200, batch_size=24, verbose=2, shuffle="batch", validation_data=(X_test, Y_test))

LSTM_prediction = model.predict(X_test)
predicted_price = scaler.inverse_transform(LSTM_prediction)
actual_price = scaler.inverse_transform(Y_test)

print(predicted_price)







from matplotlib import pyplot as plt
plt.figure(figsize=(20, 8))
plt.plot(predicted_price, color='blue', label='Predicted price of the bitcoin')
plt.plot(actual_price, color='red', label='Real price')
plt.title('Predicted vs real price of bitcoin')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(['Predicted price of the bitcoin', 'Real price'], loc='upper right')
plt.show()
plt.savefig('plot.png')



# #Train Scatter
# plt.scatter(range(20), predicted_price, c='r')
# plt.scatter(range(20), actual_price, c='r')
# plt.show()

#loss Plot
plt.plot(history.history['loss'])
plt.title('Loss Plot')
plt.legend(['loss'], loc='upper right')
plt.show()


predicts_moving = []
moving_window = [X_test[-1,:].tolist()]
moving_window = np.array(moving_window)
moving_window = np.reshape(moving_window,(1, 1, 50))
for i in range(10):
    LSTM_predictions = model.predict(moving_window)
    predicts_moving.append(LSTM_predictions[0, 0])
    LSTM_predictions = LSTM_predictions.reshape(1, 1, 1)
    # moving_window = np.reshape(moving_window, (len(X_test), 1, X_test.shape[1])))
    moving_window = np.concatenate((moving_window[:, :, 1:], LSTM_predictions), axis=2)

predicts_moving = np.array(predicts_moving, ndmin=2)
predicts_moving = np.array(predicts_moving)
predicts_moving = predicts_moving.reshape(-1, 1)
predicts_moving = scaler.inverse_transform(predicts_moving)

predicted_price = np.concatenate((predicted_price[:, :], predicts_moving), axis=0)

plt.plot(predicted_price, color='blue', label='Predicted price of the bitcoin + 10 days Future prediction')
plt.plot(actual_price, color='red', label='Real price')
plt.title('Predicted vs real price of bitcoin')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(['Predicted price of the bitcoin', 'Real price'], loc='upper right')
plt.show()
plt.savefig('plot.png')





#Loss Vs Validation Loss
from matplotlib import pyplot
pyplot.plot(history.history['val_loss'])
pyplot.plot(history.history['loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

