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
    
    from customCallBacks import *
    from Discriminator import *
    from Generator import *
    from Metric_rmse_scaled import *
    from Wgan_gp import *
except:
    pass

from tensorflow.keras import callbacks


if __name__ == '__main__':
    # Hyperparameter
    N_STEPS_IN = 7
    N_STEPS_OUT = 1

    EPOCHS = 2
    BATCH_SIZE = 128
    D_STEPS = 1
    G_STEPS = 3
    GP_WEIGHT = 10
    D_LEARNING_RATE = 0.0001
    G_LEARNING_RATE = 0.0001

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

    input_generator_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_generator_dim = y_train.shape[1]

    data_train = tf.data.Dataset.from_tensor_slices((X_train, y_train, yc_train))
    data_train = data_train.batch(BATCH_SIZE)

    data_test = tf.data.Dataset.from_tensor_slices((X_test , y_test , yc_test ))
    data_test = data_test.batch(BATCH_SIZE)

    #----------------------    Training    -----------------------

    # Instantiate the optimizer for both networks
    discriminator_optimizer = tf.keras.optimizers.Adam(D_LEARNING_RATE)
    generator_optimizer = tf.keras.optimizers.Adam(G_LEARNING_RATE)

    generator = Generator(input_generator_dim, output_generator_dim, feature_size)
    generator.compile()
    discriminator = Discriminator()
    wgan_gp = WGAN_GP(generator, discriminator, Metric_rmse_scaled(y_scaler_function), D_STEPS, G_STEPS, GP_WEIGHT)

    # Compile the WGAN model.
    wgan_gp.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        d_loss_fn=discriminator_loss,
        g_loss_fn=generator_loss
    )

    save_path = r'Models\trying'
    EPOCH_MODEL_SAVE = 1
    
    es = callbacks.EarlyStopping(monitor='val_d_loss', mode='min', verbose=1, patience=20)
    callback = [es, SaveModel(save_path, EPOCH_MODEL_SAVE), SaveBestModel(save_path)]

    history = wgan_gp.fit(data_train, epochs=EPOCHS, callbacks=callback, validation_data=data_test)

    plot_d_loss(history)
    plot_g_loss(history)
    plot_rmse(history)


    #----------------------    Test    -----------------------

    # Load test data & model
    path = r'Data\dataPreprocessed'
    test_predict_index = np.load(os.path.join(path,"index_test.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(path,"X_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(path,"y_test.npy"), allow_pickle=True)
    G_model = models.load_model(r'Models\trying\wgan_gp_best.h5')

    if N_STEPS_OUT > 1:
        get_test_global_metrics(X_test, y_test, G_model, y_scaler_function)
    plot_test_pred(X_test, y_test, G_model, y_scaler_function, test_predict_index)