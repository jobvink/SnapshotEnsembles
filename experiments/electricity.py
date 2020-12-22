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

sns.set_style("darkgrid")

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

elec = pd.read_csv('../data/electricity-normalized.csv')
X = elec.values[:,0:8].astype(np.float)
y = elec.values[:,8]
enc = LabelBinarizer()
y = enc.fit_transform(y.reshape(-1, 1))

n_epochs = 200

# %%

dataset = timeseries_dataset_from_array(X, y, sequence_length=5, batch_size=300, end_index=int(0.7 * len(X)))
dataset_test = timeseries_dataset_from_array(X, y, sequence_length=5, batch_size=300, start_index=int(0.7 * len(X)))

retrain = False
if not os.path.exists('../weights/RNN.h5') or retrain:
    model = RNN.create_rnn_model(n_timesteps=5, n_features=8, n_outputs=1)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(dataset, epochs=n_epochs)
    history = hist.history

    # persist
    model.save(f'../weights/RNN.h5')
    joblib.dump(hist.history, '../weights/RNN-history.gz')
else:
    print('Loading model from filesystem...')
    model = keras.models.load_model('../weights/RNN.h5')
    history = joblib.load('../weights/RNN-history.gz')

# %%

df = pd.DataFrame(history)
fig = sns.lineplot(data=df[['accuracy']])
fig.set_title('Electricity - Single Model')
fig.set_xlabel('Epoch')
fig.set_ylabel('Accuracy')

accuracy = model.evaluate(dataset_test)[1]
print(f'Accuracy: {accuracy}')

# %%
######################### SNAPSHOT #########################

n_models = 10
n_steps = 20
snapshot = SnapshotCallbackBuilder(nb_epochs=n_epochs, nb_snapshots=n_models, init_lr=0.1)

split_idx = int(len(X) * 0.7)
chunks_X = np.array_split(X[:split_idx], n_models)
chunks_y = np.array_split(y[:split_idx], n_models)

snapshot_model = RNN.create_rnn_model(n_timesteps=n_steps, n_features=8, n_outputs=1)
snapshot_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

accuracies = []
train_acc = []
val_acc = []

for train_X, train_y in zip(chunks_X, chunks_y):
    dataset = timeseries_dataset_from_array(train_X, train_y, sequence_length=n_steps, batch_size=100, end_index=int(0.9 * len(train_X)))
    dataset_test = timeseries_dataset_from_array(train_X, train_y, sequence_length=n_steps, batch_size=100, start_index=int(0.9 * len(train_X)))

    hist = snapshot_model.fit(
        dataset, 
        epochs=int(n_epochs // n_models),
        callbacks=snapshot.get_callbacks(
            model_prefix="RNN-snapshot"
        ),  # Build snapshot callbacks
        validation_data=dataset_test,
    )

    train_acc.extend(hist.history['acc'])
    val_acc.extend(hist.history['val_acc'])
    # accuracies.append(model.evaluate(dataset_test)[1])

    df = pd.DataFrame({ 'accuracy': train_acc, 'val_accuracy': val_acc })
    fig = sns.lineplot(data=df)
    fig.set_title('Electricity - Snapshot')
    fig.set_xlabel('Epoch')
    fig.set_ylabel('Accuracy')
    plt.show()

dataset_test = timeseries_dataset_from_array(X, y, sequence_length=5, batch_size=300, start_index=int(0.7 * len(X)))
ss_accuracy = snapshot_model.evaluate(dataset_test)[1]
print(f'Accuracy: {ss_accuracy}')

# %%

model_names = os.listdir('./RNN-snapshot-weights')
snapshot_model = RNN.create_rnn_model(n_timesteps=5, n_features=8, n_outputs=1)
snapshot_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
dataset_test = timeseries_dataset_from_array(X, y, sequence_length=5, batch_size=int(0.3 * len(X)), start_index=int(0.7 * len(X)))
X_test, y_test = list(dataset_test)[0]

X_test = X_test.numpy()
y_test = y_test.numpy()

preds = []
for fn in model_names:
    model.load_weights(f'./RNN-snapshot-weights/{fn}')

    prediction_y = model.predict(X_test, batch_size=128)
    preds.append(prediction_y)

    print("Obtained predictions from model with weights = %s" % (fn))

# %%

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

# %%

# Evenly weighted
prediction_weights = [1. / len(model_names)] * len(model_names)
calculate_weighted_accuracy(prediction_weights)

# %%

best_acc = 0.0
best_weights = None

# Parameters for optimization
constraints = ({'type': 'eq', 'fun':lambda w: 1 - sum(w)})
bounds = [(0, 1)] * len(preds)

# Check for NUM_TESTS times
for iteration in range(20):
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
# %%
