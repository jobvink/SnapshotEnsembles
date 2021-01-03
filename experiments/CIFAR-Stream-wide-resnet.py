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
from models import wide_residual_net as WN

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

parser = argparse.ArgumentParser(description='CIFAR 100 Ensemble Prediction')
parser.add_argument('--snapshot', help='Whether to snapshot the model')
args = parser.parse_args()

epochs = 50

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

active_classes = [i for i in range(0, 50)]


# This generator allows us to effectively scale the amount of data useable by the network.
generator = ImageDataGenerator(
    rotation_range=15, width_shift_range=5./32, height_shift_range=5./32, horizontal_flip=False)
generator.fit(trainX, seed=0, augment=True)

# Regular ResNet model
resnet_model = WN.create_wide_residual_network((img_rows, img_cols, 3), nb_classes=100, N=2, k=4, dropout=0.00)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
resnet_model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])

# Convert to one-hot encoding for the result
    
all_y_cat = kutils.to_categorical(all_y)

model_prefix = ''
history = []
active_classes_history = [active_classes]

for batch in range(n_batches):
    # (1): Pick 10% of classes which will randomly leave the distribution
    leaving = np.random.choice(active_classes, size=int(len(active_classes) / 10), replace=False)
    # (2): Pick 10% of classes which will randomly be added to the distribution
    adding = np.random.choice([i for i in range(0, 100) if i not in active_classes], size=int(len(active_classes) / 10), replace=False)
    # Perform 1, 2
    active_classes = [c for c in active_classes if c not in leaving]
    active_classes.extend(adding)

    batch_idx = np.array([i in active_classes for i in all_y]).nonzero()
    batch_x = all_x[batch_idx]
    batch_y = all_y_cat[batch_idx]

    active_classes_history.append(active_classes)

    # Split results 70/30 for pre batch results
    batch_x_train, batch_x_test, batch_y_train, batch_y_test = train_test_split(batch_x, batch_y, test_size=0.30, random_state=42)

    print(f'Batch: {batch + 1}/{n_batches}')
    # Perform training on the model
    if args.snapshot:
        print('Assembling snapshot ensemble')
        ''' Snapshot major parameters '''
        M = 1 # number of snapshots
        T = epochs # number of epochs
        alpha_zero = 0.1 # initial learning rate
        snapshot = SnapshotCallbackBuilder(T, M, alpha_zero)

        model_prefix = 'Wide-ResNet-CIFAR100-%d-%d-snapshot-%d' % (40, 12, batch)
        resnet_model_history = resnet_model.fit(
                                        generator.flow(batch_x_train, batch_y_train, batch_size=64), 
                                        callbacks=snapshot.get_callbacks(model_prefix=model_prefix), 
                                        epochs=epochs, 
                                        validation_data=(batch_x_test, batch_y_test)
                                    )
    else:
        print('Training regular network')
        model_prefix = 'Wide-ResNet-CIFAR100-%d-%d' % (40, 12)
        resnet_model_history = resnet_model.fit(generator.flow(batch_x_train, batch_y_train, batch_size=64), epochs=epochs, validation_data=(batch_x_test, batch_y_test))
    
    history.append(resnet_model_history)

# Store the history to a file in the FS
with open('results/' + model_prefix + ' training.csv', mode='w') as f:
    df = pd.DataFrame(columns=['','loss','acc','val_loss','val_acc'])
    for h in history:
        df2 = pd.DataFrame(h.history)
        df = df.append(df2)
    df.to_csv(f)

with open(f'results/{model_prefix}-active-labels') as f:
    df = pd.DataFrame(active_classes_history, columns=np.arange(0, 50))

yPreds = resnet_model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)
