from numpy.core.numeric import argwhere
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf

import tensorflow.keras.utils as kutils
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from snapshot import SnapshotCallbackBuilder
from models import dense_net as DN
from models import wide_residual_net as WN

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

parser = argparse.ArgumentParser(description='CIFAR 100 Ensemble Prediction')
parser.add_argument('--snapshot', help='Whether to snapshot the model')
args = parser.parse_args()

epochs = 4

# Data pre-processing
img_rows, img_cols = 32, 32
(trainX, trainY), (testX, testY) = cifar100.load_data()

# Scale data
trainX = trainX.astype('float32')
trainX /= 255.0
testX = testX.astype('float32')
testX /= 255.0

all_x = np.concatenate([trainX, testX], axis=0)
all_y = np.concatenate([trainY, testY], axis=0)

n_batches = 10

active_classes = [i for i in range(50)]


# This generator allows us to effectively scale the amount of data useable by the network.
generator = ImageDataGenerator(
    rotation_range=15, width_shift_range=5./32, height_shift_range=5./32, horizontal_flip=False)
generator.fit(trainX, seed=0, augment=True)

# Regular DenseNet model
dense_net_model = DN.create_dense_net(nb_classes=100, img_dim=(img_rows, img_cols, 3), depth=40, nb_dense_block=1,
                                growth_rate=12, nb_filter=16, dropout_rate=0.2)
dense_net_model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["acc"])

model_prefix = ''
history = []
for batch in range(n_batches):
    # (1): Pick 10% of classes which will randomly leave the distribution
    leaving = np.random.choice(active_classes, size=len(active_classes) / 10, replace=False)
    # (2): Pick 10% of classes which will randomly be added to the distribution
    adding = np.random.choice([i not in active_classes for i in range(100)], size=len(100 - active_classes) / 10, replace=False)
    # Perform 1, 2
    active_classes = [c for c in active_classes if c not in leaving]
    active_classes = np.concatenate([active_classes, adding], axis=0)

    batch_idx = (all_y in active_classes).nonzero()
    batch_x = all_x[batch_idx]
    batch_y = all_y[batch_idx]

    # Split results 70/30 for pre batch results
    batch_x_train, batch_x_test, batch_y_train, batch_y_test = train_test_split(batch_x, batch_y, test_size=0.30, random_state=42)

    # Convert to one-hot encoding for the result
    
    batch_y_train = kutils.to_categorical(batch_y_train)
    batch_y_test = kutils.to_categorical(batch_y_test)

    # Perform training on the model
    if args.snapshot:
        print('Assembling snapshot ensemble')
        ''' Snapshot major parameters '''
        M = 2 # number of snapshots
        nb_epoch = T = epochs # number of epochs
        alpha_zero = 0.1 # initial learning rate
        snapshot = SnapshotCallbackBuilder(T, M, alpha_zero)

        model_prefix = 'DenseNet-CIFAR100-%d-%d-snapshot-%d' % (40, 12, batch)
        dense_net_model_history = dense_net_model.fit(
                                        generator.flow(batch_x_train, batch_y_train, batch_size=64), 
                                        callbacks=snapshot.get_callbacks(model_prefix=model_prefix), 
                                        epochs=epochs, 
                                        validation_data=(batch_x_test, batch_y_test)
                                    )
    else:
        print('Training regular network')
        model_prefix = 'DenseNet-CIFAR100-%d-%d' % (40, 12)
        dense_net_model_history = dense_net_model.fit(generator.flow(batch_x_train, batch_y_train, batch_size=64), epochs=epochs, validation_data=(batch_x_test, batch_y_test))
    
    history.append(dense_net_model_history)

# Store the history to a file in the FS
with open('results/' + model_prefix + ' training.csv', mode='w') as f:
    df = pd.DataFrame(columns=['','loss','acc','val_loss','val_acc'])
    for h in history:
        df2 = pd.DataFrame(h.history)
        df.append(df2)
    df.to_csv(f)

yPreds = dense_net_model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)
