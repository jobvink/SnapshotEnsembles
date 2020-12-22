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

n_epochs = 150
n_models = 15
n_steps = 48
model_prefix = f"RNN-snapshot-{n_steps}-{n_epochs}"

sns.set_style("darkgrid")

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

elec = pd.read_csv('../data/electricity-normalized.csv')
X = elec.values[:,0:8].astype(np.float)
y = elec.values[:,8]
enc = LabelBinarizer()
y = enc.fit_transform(y.reshape(-1, 1))

snapshot = SnapshotCallbackBuilder(nb_epochs=n_epochs, nb_snapshots=n_models, init_lr=0.1)

split_idx = int(len(X) * 0.7)
chunks_X = np.array_split(X[:split_idx], n_models)
chunks_y = np.array_split(y[:split_idx], n_models)

snapshot_model = RNN.create_rnn_model(n_timesteps=n_steps, n_features=8, n_outputs=1)
snapshot_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])

accuracies = []
train_acc = []
val_acc = []

for train_X, train_y, i in zip(chunks_X, chunks_y, range(n_models)):
    dataset = timeseries_dataset_from_array(train_X, train_y, sequence_length=n_steps, batch_size=100, end_index=int(0.9 * len(train_X)))
    dataset_test = timeseries_dataset_from_array(train_X, train_y, sequence_length=n_steps, batch_size=100, start_index=int(0.9 * len(train_X)))

    hist = snapshot_model.fit(
        dataset, 
        epochs=int(n_epochs // n_models),
        callbacks=snapshot.get_callbacks(
            model_prefix=f'{model_prefix}-{i}'
        ),  # Build snapshot callbacks
        validation_data=dataset_test
    )

    train_acc.extend(hist.history['acc'])
    val_acc.extend(hist.history['val_acc'])
    # accuracies.append(model.evaluate(dataset_test)[1])

df = pd.DataFrame({ 'accuracy': train_acc, 'val_accuracy': val_acc })
# df = pd.DataFrame({ 'accuracy': train_acc })
fig = sns.lineplot(data=df)
fig.set_title('Electricity - Snapshot')
fig.set_xlabel('Epoch')
fig.set_ylabel('Accuracy')
plt.savefig(f'../results/{model_prefix}.pdf')

with open(f'../results/{model_prefix}.csv', mode='w') as f:
    df.to_csv(f)

dataset_test = timeseries_dataset_from_array(X, y, sequence_length=n_steps, batch_size=300, start_index=int(0.7 * len(X)))
ss_accuracy = snapshot_model.evaluate(dataset_test)[1]
print(f'Accuracy: {ss_accuracy}')

# %%
