import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pickle import load
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import itertools


def get_metrics(y_true, y_pred, whichOneString):
    """
    - A lower value of MAPE is the desired result from a prediction method.
    - POCID is the percentage of the correct trend of the model
    relative to the trend of expected value.
    - SLG less than zero indicates financial losses.
    """
    print(whichOneString)
    print('-- RMSE --', whichOneString, ':', np.sqrt(mean_squared_error(y_true, y_pred)))
    print('-- R2 --', whichOneString, ':', r2_score(y_true, y_pred))
    print('-- MAPE --', whichOneString, ':', mean_absolute_percentage_error(y_true, y_pred))
    print('-- POCID --', whichOneString, ':', pocid(y_true, y_pred))
    print('-- SLG --', whichOneString, ':', slg(y_true, y_pred), '\n')


def pocid(y_true, y_pred):
    """Prediction on change of direction"""
    return 100 * np.mean((np.diff(y_true) * np.diff(y_pred)) > 0)


def slg(y_true, y_pred):
    """
    The SLG was inspired by POCID. It defined as the mean of the losses and
    gains of the model.
    """
    all_dir = (np.diff(y_true) * np.diff(y_pred)) > 0
    lt = np.abs(np.diff(y_true))
    for ind, e in enumerate(lt):
        if all_dir[ind] == False:
            lt[ind] = - lt[ind]
    return np.mean(lt)

        
def get_pred_rescaled(X_test, y_test, G_model, y_scaler, pred_arima=None):
    if pred_arima is None:
        y_predicted = G_model(X_test)
    else:
        y_predicted = pred_arima
    rescaled_real_y = y_scaler.inverse_transform(y_test)
    rescaled_predicted_y = y_scaler.inverse_transform(y_predicted)
    return rescaled_real_y, rescaled_predicted_y


def get_real_pred_flat(X_test, y_test, G_model, y_scaler, pred_arima=None):
    # Set output steps
    output_dim = y_test.shape[1]

    rescaled_real_y, rescaled_predicted_y = get_pred_rescaled(X_test, y_test, G_model, y_scaler, pred_arima)

    rescaled_real_y = list(itertools.chain.from_iterable(rescaled_real_y))
    rescaled_predicted_y = list(itertools.chain.from_iterable(rescaled_predicted_y))
    return rescaled_real_y, rescaled_predicted_y


def get_test_global_metrics(X_test, y_test, G_model, y_scaler, pred_arima=None):
    real, predicted = get_real_pred_flat(X_test, y_test, G_model, y_scaler, pred_arima)
    get_metrics(real, predicted, 'Global')


def plot_test_pred(X_test, y_test, G_model, y_scaler, test_predict_index, pred_arima=None):
    output_dim = y_test.shape[1]
    rescaled_real_y, rescaled_predicted_y = get_pred_rescaled(X_test, y_test, G_model, y_scaler, pred_arima)

    plt.figure(figsize=(16, 8))
    plt.xlabel("Date")
    plt.ylabel("Stock price")

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

        day_ahead = '%s Day ahead' % (nb_day + 1)
        plt.plot(predict_result, label=day_ahead)
        get_metrics(real_y, pred_y, day_ahead)
   
    plt.legend(loc="upper left", fontsize=16)
    plt.show()