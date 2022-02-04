import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import models
import numpy as np
from pickle import load
from sklearn.metrics import mean_squared_error
import itertools



def get_pred_rescaled(X_test, y_test, G_model, y_scaler):
    y_predicted = G_model(X_test)
    rescaled_real_y = y_scaler.inverse_transform(y_test)
    rescaled_predicted_y = y_scaler.inverse_transform(y_predicted)
    return rescaled_real_y, rescaled_predicted_y


def get_real_pred_flat(X_test, y_test, G_model, y_scaler):
    # Set output steps
    output_dim = y_test.shape[1]

    rescaled_real_y, rescaled_predicted_y = get_pred_rescaled(X_test, y_test, G_model, y_scaler)

    rescaled_real_y = list(itertools.chain.from_iterable(rescaled_real_y))
    rescaled_predicted_y = list(itertools.chain.from_iterable(rescaled_predicted_y))
    return rescaled_real_y, rescaled_predicted_y


def get_test_global_rmse(X_test, y_test, G_model, y_scaler):
    real, predicted = get_real_pred_flat(X_test, y_test, G_model, y_scaler)
    print('-- RMSE -- ', np.sqrt(mean_squared_error(predicted, real)))


def plot_test_pred(X_test, y_test, G_model, y_scaler, test_predict_index):
    output_dim = y_test.shape[1]
    rescaled_real_y, rescaled_predicted_y = get_pred_rescaled(X_test, y_test, G_model, y_scaler)

    plt.figure(figsize=(16, 8))
    plt.xlabel("Date")
    plt.ylabel("Stock price")

    print(type(test_predict_index))
    if output_dim > 1:
        test_predict_index = test_predict_index[:-(output_dim-1)]

    real_y = [e[0] for e in rescaled_real_y]
    real_price = pd.DataFrame(real_y, columns=["real_price"], 
                                                index=test_predict_index)
    plt.plot(real_price, label='real')

    for nb_day in range(output_dim):
        pred_y = [e[nb_day] for e in rescaled_predicted_y]

        predict_result = pd.DataFrame(pred_y, columns=["predicted_price"],
                                                        index=test_predict_index)

        plt.plot(predict_result, label='%s day ahead' % (nb_day + 1))
        print('-- RMSE -- %s day ahead --', np.sqrt(mean_squared_error(predict_result, real_price)))
    
    plt.legend(loc="upper left" % (nb_day + 1), fontsize=16)
    plt.show()