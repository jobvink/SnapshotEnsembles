from tensorflow.keras.models import Sequential, Embedding, LSTM
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

def create_model():
    model = Sequential()
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    model.add(Embedding(input_dim=1000, output_dim=64))

    # Add a LSTM layer with 128 internal units.
    model.add(LSTM(128))

    # Add a Dense layer with 10 units.
    model.add(Dense(10))

    model.summary()
    return model