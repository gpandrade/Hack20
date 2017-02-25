import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time, csv

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

#keep fixed for reproducibility
np.random.seed(123)


def get_data(sequence_length=15):
    # load data from .csv file
    metal_data = pd.read_csv('LSPC_data.csv')
	
    x = np.stack((np.array(metal_data['Copper Result']), np.array(metal_data['Iron Result']), np.array(metal_data['Chloride Result'])), axis=0).T
    y = np.array(metal_data['Lead Result'].values.reshape((1365, 1)))	

    new_x = np.zeros((91, 15, 3))
    new_y = np.zeros((91, 15, 1))
    
    for idx in range(0, x.shape[0], sequence_length):
        new_x[idx / 15, :, :] = x[idx : idx + sequence_length, :]
        new_y[idx / 15, :, :] = y[idx : idx + sequence_length, :]
        
    x, y = np.array(new_x), np.array(new_y)

    # split up data (approx. 80% / 20% split for now) for training, testing datsets
    return x[:73, :, :], y[:73, :, :], x[73:, :, :], y[73:, :, :]


def build_model_simp():
    model = Sequential()
    layers = [3, 50, 1]

    model.add(LSTM(layers[1], input_shape=(15, 3)))
    model.add(Dense(layers[2]))
    model.add(Activation('linear'))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print 'Compilation Time : ', time.time() - start
    
    return model


def build_model_simp_n_drop():
    model = Sequential()
    layers = [3, 50, 1]

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    
    model.add(Dropout(0.3))

    model.add(Dense(
        output_dim=layers[2]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print "Compilation Time : ", time.time() - start

    return model


def build_model_multi():
    model = Sequential()
    layers = [3, 50, 100, 1]

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print "Compilation Time : ", time.time() - start

    return model


def run_network(model='simp'):
    global_start_time = time.time()
    epochs = 10
    
    X_train, y_train, X_test, y_test = get_data()

    if model == 'simp':
        model = build_model_simp()
    elif model == 'simp_n_drop':
        model = build_model_simp_n_drop()
    elif model == 'multi':
   	    model = build_model_multi()
    else:
        raise NotImplementedError

    model.fit(X_train, y_train, batch_size=1, nb_epoch=epochs, validation_split=0.2)
    
    predicted = model.predict(X_test)
    predicted = np.reshape(predicted, (predicted.size,))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_test[:100])
    plt.plot(predicted[:100])
    plt.show()
    
    print 'Training duration (s) : ', time.time() - global_start_time
    
    return model, y_test, predicted


if __name__ == '__main__':
	run_network('simp')

