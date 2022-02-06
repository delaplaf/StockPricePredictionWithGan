import os
import sys
import inspect
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pickle import load

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import models

try:
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    sys.path.insert(0, parentdir)

    from preprocessing import *
    from test_pred import *
    from plot_data import *

    from gru import *
    from lstm import *
    from bidir_lstm import *

except:
    pass

from tensorflow.keras import callbacks

# 0 -> gru
# 1 -> ltsm
# 2 -> bidir_lstm
NB_MODEL = 2

if __name__ == '__main__':
    # Hyperparameter
    N_STEPS_IN = 7
    N_STEPS_OUT = 1

    EPOCHS = 200
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0001

   #----------------------    Load Data    -----------------------
    data = pd.read_csv(r'Data\DataFacebook.csv', parse_dates=['date'])
    y_scaler_function = all_preprocessing(data, N_STEPS_IN, N_STEPS_OUT)

    path = r'Data\dataPreprocessed'
    X_train = np.load(os.path.join(path, "X_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(path, "y_train.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(path, "X_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(path, "y_test.npy"), allow_pickle=True)
    yc_train = np.load(os.path.join(path, "yc_train.npy"), allow_pickle=True)
    yc_test = np.load(os.path.join(path, "yc_test.npy"), allow_pickle=True)

    input_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_dim = y_train.shape[1]

    #----------------------    Training    -----------------------

    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    
    es = callbacks.EarlyStopping(verbose=1, patience=10)
    save_path = r'Models\trying\gru_best.h5'
    checkpoint = callbacks.ModelCheckpoint(filepath=save_path, save_best_only=True)

    callback = [es, checkpoint]

    if NB_MODEL == 0:
        model = simple_gru(input_dim, output_dim, feature_size)
    elif NB_MODEL == 1:
        model = simple_lstm(input_dim, output_dim, feature_size)
    elif NB_MODEL == 2:
        model = simple_bidir_lstm(input_dim, output_dim, feature_size)

    model.compile(optimizer=optimizer, loss='mse')
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                        validation_data=(X_test, y_test), callbacks=callback,
                        verbose=2, shuffle=False)

    plot_loss(history)

    #----------------------    Test    -----------------------

    # Load test data & model
    path = r'Data\dataPreprocessed'
    test_predict_index = np.load(os.path.join(path,"index_test.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(path,"X_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(path,"y_test.npy"), allow_pickle=True)
    test_model = models.load_model(save_path )

    if N_STEPS_OUT > 1:
        get_test_global_metrics(X_test, y_test, test_model, y_scaler_function)
    plot_test_pred(X_test, y_test, test_model, y_scaler_function, test_predict_index)