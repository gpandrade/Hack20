import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

#keep fixed for reproducibility
np.random.seed(123)



def build_model_simp():
    model = Sequential()
    layers = [3, 50, 1]

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))

    model.add(Dense(output_dim=layers[2]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print "Compilation Time : ", time.time() - start
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



def run_network(model=simp, data):
    global_start_time = time.time()
    epochs = 10
    path_to_dataset = 'tbd'    #################################################Dont Forget!!!!!!!!!!!!!!!!!!!!#########################################################

    X_train, y_train, X_test, y_test = data #Might not need

    print '\nCompiling...\n'

    if model is simp:
    	model = build_model_simp()
    elif model is simp_n_drop:
    	model = build_model_simp_n_drop()
   	else:
   		model = build_model_multi()

    model.fit(
        X_train, y_train,
        batch_size=1, nb_epoch=epochs, validation_split=0.10)
    predicted = model.predict(X_test)
    predicted = np.reshape(predicted, (predicted.size,))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_test[:100])
    plt.plot(predicted[:100])
    plt.show()
    
    print 'Training duration (s) : ', time.time() - global_start_time
    
return model, y_test, predicted