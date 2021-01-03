from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

def create_rnn_model(n_timesteps, n_features, n_outputs):
    model = Sequential()
    model.add(LSTM(256, input_shape=(n_timesteps, n_features), return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    # model.add(Dense(10, activation='softmax'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.summary()
    return model