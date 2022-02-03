import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


def plotFacebookStock(data):
    """
    Create Facebook stock price plot
    https://www.earthdatascience.org/courses/use-data-open-source-python/use-time-series-data-in-python/date-time-types-in-pandas-python/customize-dates-matplotlib-plots-python/
    """
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(data['date'], data['close'], label='Facebook stock')
    ax.set(xlabel="Date",
        ylabel="USD",
        title="Facebook Stock Price")
    date_form = DateFormatter("%Y")
    ax.xaxis.set_major_formatter(date_form)
    plt.show()


def plot_technical_indicators(data, last_days):
       plt.figure(figsize=(16, 10), dpi=100)
       shape_0 = data.shape[0]
       xmacd_ = shape_0 - last_days

       data = data.iloc[-last_days:, :]
       x_ = range(3, data.shape[0])
       x_ = list(data.index)

       # Plot first subplot
       plt.subplot(2, 1, 1)
       plt.plot(data['MA7'], label='MA 7', color='g', linestyle='--')
       plt.plot(data['close'], label='Closing Price', color='b')
       plt.plot(data['MA21'], label='MA 21', color='r', linestyle='--')
       plt.plot(data['upper_band'], label='Upper Band', color='c')
       plt.plot(data['lower_band'], label='Lower Band', color='c')
       plt.fill_between(x_, data['lower_band'], data['upper_band'], alpha=0.35)
       plt.title('Technical indicators for Facebook - last {} days.'.format(last_days))
       plt.ylabel('USD')
       plt.legend()

       # Plot second subplot
       plt.subplot(2, 1, 2)
       plt.title('MACD')
       plt.plot(data['MACD'], label='MACD', linestyle='-.')
       plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
       plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
       plt.plot(data['logmomentum'], label='Momentum', color='b', linestyle='-')
       
       plt.legend()
       plt.show()


def plot_Fourier(data):
    plt.figure(figsize=(14, 7), dpi=100)
    for i in [3, 6, 9]:
        fft_num = data['absolute of {} comp'.format(i)]
        plt.plot(fft_num, label='{} components'.format(i))
    plt.plot(data['close'], label='Real')
    plt.xlabel('Days')
    plt.ylabel('USD')
    plt.title('Facebook (close) stock prices & Fourier absolute transforms')
    plt.legend()
    plt.show()


def plot_loss(history):
    """
    summarize history for loss
    """
    plt.plot(history.history['d_loss'])
    plt.plot(history.history['g_loss'])
    plt.plot(history.history['val_d_loss'])
    plt.plot(history.history['val_g_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['d_loss', 'g_loss', 'val_d_loss', 'val_g_loss'], loc='upper left')
    plt.show()


def plot_rmse(history):
    """
    summarize history for loss
    """
    plt.plot(history.history['rmse'])
    plt.plot(history.history['val_rmse'])
    plt.title('model rmse')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()