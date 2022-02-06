import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_technical_indicators(data):
    """
    Calculate technical indicators
    """
    # Create 7 and 21 days Moving Average
    data['MA7'] = data.iloc[:,1].rolling(window=7).mean()
    data['MA21'] = data.iloc[:,1].rolling(window=21).mean()

    # Create MACD
    data['MACD'] = data.iloc[:,1].ewm(span=26).mean() - data.iloc[:,2].ewm(span=12,adjust=False).mean()

    # Create Bollinger Bands
    data['20SD'] = data.iloc[:, 1].rolling(20).std()
    data['upper_band'] = data['MA21'] + (data['20SD'] * 2)
    data['lower_band'] = data['MA21'] - (data['20SD'] * 2)

    # Create Exponential moving average
    data['EMA'] = data.iloc[:,1].ewm(com=0.5).mean()

    # Create LogMomentum
    data['logmomentum'] = np.log(data.iloc[:,1] - 1)

    return data


def get_fourier_transfer(data):
    """
    Getting the Fourier transform features
    """
    # Get the columns for doing fourier
    data_FT = data[['date', 'close']]

    close_fft = np.fft.fft(np.asarray(data_FT['close'].tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    fft_list = np.asarray(fft_df['fft'].tolist())
    fft_com_df = pd.DataFrame()
    for num_ in [3, 6, 9]:
        fft_list_m10 = np.copy(fft_list);
        fft_list_m10[num_:-num_] = 0
        fft_ = np.fft.ifft(fft_list_m10)
        fft_com = pd.DataFrame({'fft': fft_})
        fft_com['absolute of ' + str(num_) + ' comp'] = fft_com['fft'].apply(lambda x: np.abs(x))
        fft_com['angle of ' + str(num_) + ' comp'] = fft_com['fft'].apply(lambda x: np.angle(x))
        fft_com = fft_com.drop(columns='fft')
        fft_com_df = pd.concat([fft_com_df, fft_com], axis=1)

    return fft_com_df


def manage_nan(data):
    # Replace 0 by NA
    data.replace(0, np.nan, inplace=True)
    data.iloc[:, 1:] = pd.concat([data.iloc[:, 1:].ffill(), data.iloc[:, 1:].bfill()]).groupby(level=0).mean()


def manage_dates(data):
    datetime_series = pd.to_datetime(data['date'])
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    data = data.set_index(datetime_index)
    data = data.sort_values(by='date')
    data = data.drop(columns='date')
    return data


def get_X_y(X_data, y_data, n_steps_in, n_steps_out):
    X = list()
    y = list()
    yc = list()

    length = len(X_data)
    for i in range(0, length, 1):
        X_value = X_data[i: i + n_steps_in][:, :]
        y_value = y_data[i + n_steps_in: i + (n_steps_in + n_steps_out)][:, 0]
        yc_value = y_data[i: i + n_steps_in][:, :]
        if len(X_value) == n_steps_in and len(y_value) == n_steps_out:
            X.append(X_value)
            y.append(y_value)
            yc.append(yc_value)

    return np.array(X), np.array(y), np.array(yc)


def predict_index(data, X_train, n_steps_in, n_steps_out):
    """
    get the train test predict index
    """

    # get the predict data (remove the in_steps days)
    train_predict_index = data.iloc[n_steps_in : X_train.shape[0] + n_steps_in + n_steps_out - 1, :].index
    test_predict_index = data.iloc[X_train.shape[0] + n_steps_in:, :].index

    return train_predict_index, test_predict_index


def split_train_test(data):
    """
    Split train/test dataset
    """
    train_size = round(len(data) * 0.7)
    data_train = data[0:train_size]
    data_test = data[train_size:]
    return data_train, data_test


def reshape_dataset(path, data, X_scaled, y_scaled, n_steps_in = 3, n_steps_out = 1):
    """
    Set the data input steps and output steps, 
    we use 30 days data to predict 1 day price here, 
    reshape it to (None, input_step, number of features) used for LSTM input
    path : where to save the data
    """
    n_features = X_scaled.shape[1]
    # Get data and check shape
    X, y, yc = get_X_y(X_scaled, y_scaled, n_steps_in, n_steps_out)
    X_train, X_test, = split_train_test(X)
    y_train, y_test, = split_train_test(y)
    yc_train, yc_test, = split_train_test(yc)
    index_train, index_test, = predict_index(data, X_train, n_steps_in, n_steps_out)

    print('X shape: ', X.shape)
    print('y shape: ', y.shape)
    print('X_train shape: ', X_train.shape)
    print('y_train shape: ', y_train.shape)
    print('y_c_train shape: ', yc_train.shape)
    print('X_test shape: ', X_test.shape)
    print('y_test shape: ', y_test.shape)
    print('y_c_test shape: ', yc_test.shape)
    print('index_train shape:', index_train.shape)
    print('index_test shape:', index_test.shape)

    # Save all data to easily access
    np.save(os.path.join(path, "X_train.npy"), X_train)
    np.save(os.path.join(path,"y_train.npy"), y_train)
    np.save(os.path.join(path,"X_test.npy"), X_test)
    np.save(os.path.join(path,"y_test.npy"), y_test)
    np.save(os.path.join(path,"yc_train.npy"), yc_train)
    np.save(os.path.join(path,"yc_test.npy"), yc_test)
    np.save(os.path.join(path,'index_train.npy'), index_train)
    np.save(os.path.join(path,'index_test.npy'), index_test)
    print("Everything saved in ", path)


def all_preprocessing(n_steps_in, n_steps_out):
    data = pd.read_csv(r'Data\DataFacebook.csv', parse_dates=['date'])

    # Get technical features
    technical_data = get_technical_indicators(data)
    technical_data = technical_data.iloc[20:,:].reset_index(drop=True)

    # Get Fourier features
    fourier_data = get_fourier_transfer(technical_data)

    # Get all features
    data_final = pd.concat([technical_data, fourier_data], axis=1)

    manage_nan(data_final)
    data_final = manage_dates(data_final)

    # Get features and target
    X = pd.DataFrame(data_final.iloc[:, :])
    y = pd.DataFrame(data_final.iloc[:, 0])

    # Normalized the data
    X_scaler_function = MinMaxScaler(feature_range=(-1, 1))
    y_scaler_function = MinMaxScaler(feature_range=(-1, 1))

    X_scaled = X_scaler_function.fit_transform(X)
    y_scaled = y_scaler_function.fit_transform(y)

    pathToSave = r'Data\dataPreprocessed'
    reshape_dataset(pathToSave, data_final, X_scaled, y_scaled, n_steps_in, n_steps_out)
    return y_scaler_function