from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Sequential


def simple_bidir_lstm(input_dim, output_dim, feature_size):
    model = Sequential()

    model.add(Bidirectional(LSTM(units= 128), input_shape=(input_dim, feature_size)))

    model.add(Dense(64))

    model.add(Dense(units=output_dim))

    return model