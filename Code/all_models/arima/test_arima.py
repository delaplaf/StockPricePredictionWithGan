import os
import sys
import inspect
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima

try:
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    sys.path.insert(0, parentdir)

    from preprocessing import *
    from test_pred import *
except:
    pass


N_STEPS_IN = 7
N_STEPS_OUT = 1

#----------------------    Load Data    -----------------------
data = pd.read_csv(r'Data\DataFacebook.csv', parse_dates=['date'])
close_price = data[['date', 'close']].copy()
y_scaler_function = all_preprocessing(close_price, N_STEPS_IN, N_STEPS_OUT, is_arima=True)

path = r'Data\dataPreprocessed'
# train == test for arima
X_test = np.load(os.path.join(path, "X_test.npy"), allow_pickle=True)
y_test = np.load(os.path.join(path, "y_test.npy"), allow_pickle=True)
test_predict_index = np.load(os.path.join(path,"index_test.npy"), allow_pickle=True)

pred_arima = []
nb_pred = len(X_test)
for i, sample in enumerate(X_test):
    arima_model = auto_arima(sample)
    pred = arima_model.predict(n_periods=N_STEPS_OUT)
    pred_arima.append(pred)
    print(i + 1, '/', nb_pred)

if N_STEPS_OUT > 1:
        get_test_global_metrics(X_test, y_test, None, y_scaler_function, pred_arima=pred_arima)
plot_test_pred(X_test, y_test, None, y_scaler_function, test_predict_index, pred_arima=pred_arima)

