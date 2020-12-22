# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import joblib

from tensorflow import keras
from tensorflow.keras import metrics
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras import activations
from tensorflow.python.ops.gen_dataset_ops import OneShotIterator
from snapshot import SnapshotCallbackBuilder
from models import RNN
from scipy.optimize import minimize
from sklearn.metrics import log_loss

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

n_models = 10
n_steps = 5
folder_name = './RNN-snapshot-48'

elec = pd.read_csv('../data/electricity-normalized.csv')
X = elec.values[:,0:8].astype(np.float)
y = elec.values[:,8]
enc = LabelBinarizer()
y = enc.fit_transform(y.reshape(-1, 1))

model_names = os.listdir(folder_name)
snapshot_model = RNN.create_rnn_model(n_timesteps=n_steps, n_features=8, n_outputs=1)
snapshot_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])
dataset_test = timeseries_dataset_from_array(X, y, sequence_length=n_steps, batch_size=int(0.3 * len(X)), start_index=int(0.7 * len(X)))
X_test, y_test = list(dataset_test)[0]

X_test = X_test.numpy()
y_test = y_test.numpy()


preds = []
for fn in model_names:
    snapshot_model.load_weights(f'{folder_name}/{fn}')

    prediction_y = snapshot_model.predict(X_test, batch_size=128)
    preds.append(prediction_y)

    print("Obtained predictions from model with weights = %s" % (fn))


def calculate_weighted_accuracy(prediction_weights):
    weighted_predictions = np.zeros((X_test.shape[0], 1), dtype='float32')
    for weight, prediction in zip(prediction_weights, preds):
        weighted_predictions += weight * prediction
    y_predicted = np.argmax(weighted_predictions, axis=1)
    accuracy = accuracy_score(y_test, y_predicted) * 100
    error = 100 - accuracy
    print("Accuracy: ", accuracy)
    print("Error: ", error)

# Create the loss metric 
def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = np.zeros((X_test.shape[0], 1), dtype='float32')

    for weight, prediction in zip(weights, preds):
        final_prediction += weight * prediction

    return log_loss(y_test, final_prediction)


# Evenly weighted
print('\n\n--------------- Evenly weighted ---------------\n')
prediction_weights = [1. / len(model_names)] * len(model_names)
calculate_weighted_accuracy(prediction_weights)

print('\n\n------------- Calculating Weights -------------\n')
best_acc = 0.0
best_weights = None

# Parameters for optimization
constraints = ({'type': 'eq', 'fun':lambda w: 1 - sum(w)})
bounds = [(0, 1)] * len(preds)

# Check for NUM_TESTS times
for iteration in range(2):
    # Random initialization of weights
    prediction_weights = np.random.random(len(model_names))
    
    # Minimise the loss 
    result = minimize(log_loss_func, prediction_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    print('Best Ensemble Weights: {weights}'.format(weights=result['x']))
    
    weights = result['x']
    weighted_predictions = np.zeros((X_test.shape[0], 1), dtype='float32')
    
    # Calculate weighted predictions
    for weight, prediction in zip(weights, preds):
        weighted_predictions += weight * prediction

    y_prediction = np.argmax(weighted_predictions, axis=1)

    # Calculate weight prediction accuracy
    accuracy = accuracy_score(y_test, y_prediction) * 100
    error = 100 - accuracy
    print("Iteration %d: Accuracy: " % (iteration + 1), accuracy)
    print("Iteration %d: Error: " % (iteration + 1), error)
    
    # Save current best weights 
    if accuracy > best_acc:
        best_acc = accuracy
        best_weights = weights

print("Best Accuracy: ", best_acc)
print("Best Weights: ", best_weights)
calculate_weighted_accuracy(best_weights)
# %%
