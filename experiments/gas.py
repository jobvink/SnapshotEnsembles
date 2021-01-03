import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append("../../..")

import argparse
from distutils.util import strtobool

str2bool = lambda x: bool(strtobool(x))
parser = argparse.ArgumentParser(description='RNN on Gas Dataset')
parser.add_argument('--epochs', type=int, default=400, help='Total of number of epochs to train on')
parser.add_argument('--lr', '--learning-rate', type=float, default=0.1, help='Learning rate used with snapshot ensembling')
parser.add_argument('--models', type=int, default=10, help='Number of snapshots to take/batches to split data on')
parser.add_argument('--steps', type=int, default=24, help='Number of timesteps in a sliding window')
parser.add_argument('--snapshot', type=str2bool, default=False, nargs='?', help='Train snapshot model or default')
parser.add_argument('-n', type=int, default=1, help='Number of times to run')

args = parser.parse_args()
print(args)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras import activations
from tensorflow.python.ops.gen_dataset_ops import OneShotIterator
from snapshot import SnapshotCallbackBuilder
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
from models import RNN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM, ConvLSTM2D, TimeDistributed, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

from sklearn.metrics import accuracy_score
sns.set_style("darkgrid")

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

cats = ['Ethanol', 'Ethylene', 'Ammonia', 'Acetaldehyde', 'Acetone', 'Toluene']
enc = OneHotEncoder(categories=[cats], handle_unknown="ignore")

def get_m_snapshots(folder_name, m):
    model_names = os.listdir(f'weights/{folder_name}')
    model_names = [k for k in model_names if not 'Best' in k]
    return model_names[-min(len(model_names), m):]

def calculate_weighted_accuracy(predictions, y_test):
    predictions = np.array(predictions)
    prediction_weights = [1. / predictions.shape[0]] * predictions.shape[0]
    weighted_predictions = np.zeros((predictions.shape[1], len(cats)), dtype='float32')
    for weight, prediction in zip(prediction_weights, predictions):
        weighted_predictions += weight * prediction
    yPred = enc.inverse_transform(weighted_predictions)
    yTrue = enc.inverse_transform(y_test)
    return accuracy_score(yTrue, yPred)

model_prefix = f"RNN-gas-{'snapshot-' if args.snapshot else ''}{args.models}M-{args.steps}T-{args.epochs}E-{args.lr}lr"

gas = pd.read_csv('./data/gas-normalized.csv')
X = gas.values[:,0:-1].astype(np.float)
y = gas.values[:,-1]

y = enc.fit_transform(y.reshape(-1, 1)).toarray()

split_idx = int(len(X) * 0.7)
chunks_X = np.array(np.array_split(X[:split_idx], args.models))
chunks_y = np.array(np.array_split(y[:split_idx], args.models))

for i in range(1, args.n + 1):
    model_folder = f'{model_prefix}/{i}'

    if not os.path.exists(f'weights/{model_folder}'):
        os.makedirs(f'weights/{model_folder}')

    print(f'\n\nTraining {model_prefix}, iteration {i}...\n\n')
    snapshot = SnapshotCallbackBuilder(nb_epochs=args.epochs, nb_snapshots=args.models, init_lr=args.lr)
    model = RNN.create_rnn_model(n_timesteps=args.steps, n_features=len(X[0]), n_outputs=len(cats))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model_predictor = RNN.create_rnn_model(n_timesteps=args.steps, n_features=len(X[0]), n_outputs=len(cats))
    model_predictor.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    train_acc = []
    true_val_acc = []
    val_acc = []

    for train_X, train_y, j in zip(chunks_X, chunks_y, range(args.models)):
        dataset = timeseries_dataset_from_array(train_X, train_y, sequence_length=args.steps, batch_size=1500, end_index=int(0.9 * len(train_X)))
        dataset_test = timeseries_dataset_from_array(train_X, train_y, sequence_length=args.steps, batch_size=len(train_X), start_index=int(0.9 * len(train_X)))

        if args.snapshot:
            hist = model.fit(
                dataset, 
                epochs=int(args.epochs // args.models),
                callbacks=snapshot.get_callbacks(
                    model_prefix=f'{model_folder}/{model_prefix}-{j}'
                ),  # Build snapshot callbacks
                validation_data=dataset_test,
            )

            predictions = []
            for fn in get_m_snapshots(model_folder, 5):
                model_predictor.load_weights(f'weights/{model_folder}/{fn}')
                prediction = model_predictor.predict(dataset_test, batch_size=100)
                predictions.append(prediction)

            train_acc.extend(hist.history['acc'])
            true_val_acc.extend(hist.history['val_acc'])
            validation_acc = calculate_weighted_accuracy(predictions, list(dataset_test)[0][1])
            val_acc.extend([validation_acc])

        else: # no snapshot
            hist = model.fit(
                dataset, 
                epochs=int(args.epochs // args.models),
                validation_data=dataset_test,
            )

            train_acc.extend(hist.history['acc'])
            true_val_acc.extend(hist.history['val_acc'])
            yPred = enc.inverse_transform(model.predict(dataset_test))
            yTrue = enc.inverse_transform(list(dataset_test)[0][1])
            validation_accuracy = accuracy_score(yTrue, yPred)
            val_acc.extend([validation_accuracy])

    if not args.snapshot:
        print(f'Saving model in weights/{model_folder}/{model_prefix}.h5')
        model.save_weights(f'weights/{model_folder}/{model_prefix}.h5')

    plt.clf()
    df = pd.DataFrame({'val_accuracy': val_acc })
    fig = sns.lineplot(data=df)
    fig.set_title(f"Gas - {'Snapshot' if args.snapshot else 'Single Model'}")
    fig.set_xlabel('Batch')
    fig.set_ylabel('Accuracy')
    plt.savefig(f'weights/{model_folder}/{model_prefix}.pdf')
    
    plt.clf()
    df2 = pd.DataFrame({ 'Accuracy': train_acc, 'Validation Accuracy': true_val_acc })
    fig = sns.lineplot(data=df2)
    fig.set_title(f"Gas - {'Snapshot' if args.snapshot else 'Single Model'} - Learning Curve")
    fig.set_xlabel('Epochs')
    fig.set_ylabel('Accuracy')
    plt.savefig(f'weights/{model_folder}/{model_prefix}-lc.pdf')

    with open(f'weights/{model_prefix}/{i}.csv', mode='w') as f:
        df.to_csv(f)
        
    with open(f'weights/{model_prefix}/{i}-lc.csv', mode='w') as f:
        df2.to_csv(f)