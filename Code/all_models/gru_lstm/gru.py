from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential


def simple_gru(input_dim, output_dim, feature_size):
    model = Sequential()

    model.add(GRU(units=128, return_sequences = True, input_shape=(input_dim, feature_size)))
    model.add(Dropout(0.2))

    model.add(GRU(units=64, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(GRU(units=32))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=output_dim))

    return model