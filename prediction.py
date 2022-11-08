import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')

# from itertools import product
# import statsmodels.api as sm
# from itertools import cycle
# import plotly.offline as py
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# import math
# import datetime as dt
# import os

plt.style.use('seaborn-darkgrid')
dolar_data_set = pd.read_csv("exchange-ratio.csv")
group = dolar_data_set[['Close']]

prediction_days = 1000

df_train = group[:len(group) - prediction_days].values.reshape(-1, 1)
df_test = group[len(group) - prediction_days:].values.reshape(-1, 1)

chosen_col = 'Close'

scaler_train = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler_train.fit_transform(df_train)

scaler_test = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler_test.fit_transform(df_test)


def dataset_generator_lstm(dataset, look_back=5):
    dataX, dataY = [], []

    for i in range(len(dataset) - look_back):
        window_size_x = dataset[i:(i + look_back), 0]
        dataX.append(window_size_x)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


trainX, trainY = dataset_generator_lstm(scaled_train)

testX, testY = dataset_generator_lstm(scaled_test)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

model = Sequential()
model.add(LSTM(units=128, activation='relu', return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=64, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

checkpoint_path = 'my_best_model.hdf5'

checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

earlystopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

callbacks = [checkpoint, earlystopping]

history = model.fit(trainX, trainY, batch_size=32, epochs=300, verbose=1, shuffle=False, validation_data=(testX, testY),
                    callbacks=callbacks)

model_from_saved_checkpoint = load_model(checkpoint_path)

plt.figure(figsize=(16, 7))
plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

predicted_dolar_price_train_data = model_from_saved_checkpoint.predict(trainX)

predicted_dolar_price_train_data = scaler_train.inverse_transform(predicted_dolar_price_train_data.reshape(-1, 1))

train_actual = scaler_train.inverse_transform(trainY.reshape(-1, 1))

plt.figure(figsize=(16, 7))

plt.plot(predicted_dolar_price_train_data, 'r', label='Education Estimated Price', linewidth="1")

plt.plot(train_actual, label='Real Price', linewidth="1")

plt.legend()
plt.show()

predicted_dolar_price_test_data = model_from_saved_checkpoint.predict(testX)

predicted_dolar_price_test_data = scaler_test.inverse_transform(predicted_dolar_price_test_data.reshape(-1, 1))

test_actual = scaler_test.inverse_transform(testY.reshape(-1, 1))

# -------------------------------------------------------


plt.figure(figsize=(16, 7))

plt.plot(predicted_dolar_price_test_data, 'r', label='Estimated Price of Test', linewidth="0.4")
plt.plot(test_actual, label='Real Price', linewidth="0.4")
plt.legend()
plt.show()

lookback_period = 30

testX_last_5_day = testX[testX.shape[0] - lookback_period:]

predicted_5_days_forecast_price_test_x = []

for i in range(30):
    predicted_forecast_price_test_x = model_from_saved_checkpoint.predict(testX_last_5_day[i:i + 1])
    predicted_forecast_price_test_x = scaler_test.inverse_transform(predicted_forecast_price_test_x.reshape(-1, 1))

    predicted_5_days_forecast_price_test_x.append(predicted_forecast_price_test_x)

predicted_5_days_forecast_price_test_x = np.array(predicted_5_days_forecast_price_test_x)

predicted_5_days_forecast_price_test_x = predicted_5_days_forecast_price_test_x.flatten()

predicted_dolar_price_test_data = predicted_dolar_price_test_data.flatten()

predicted_dolar_test_concatenate = np.concatenate(
    (predicted_dolar_price_test_data, predicted_5_days_forecast_price_test_x))

# ----------------------------------------------------------------------------

plt.figure(figsize=(16, 7))
plt.plot(predicted_dolar_test_concatenate, 'r', marker='.', label="Estimated Price", linewidth="0.8", alpha=1)
plt.plot(test_actual, label="Real Price", marker='.', linewidth="0.8")
plt.legend()
plt.show()
