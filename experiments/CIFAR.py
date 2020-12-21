import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import argparse

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

# This generator allows us to effectively scale the amount of data useable by the network.
generator = ImageDataGenerator(
    rotation_range=15, width_shift_range=5./32, height_shift_range=5./32, horizontal_flip=False)
generator.fit(trainX, seed=0, augment=True)


# Convert to one-hot encoding for the result
trainY = kutils.to_categorical(trainY)
testY_cat = kutils.to_categorical(testY)

# Regular DenseNet model
dense_net_model = DN.create_dense_net(nb_classes=100, img_dim=(img_rows, img_cols, 3), depth=40, nb_dense_block=1,
                                growth_rate=12, nb_filter=16, dropout_rate=0.2)
dense_net_model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["acc"])


# Perform training on the model
if args.snapshot:
    print('Assembling snapshot ensemble')
    ''' Snapshot major parameters '''
    M = 2 # number of snapshots
    nb_epoch = T = epochs # number of epochs
    alpha_zero = 0.1 # initial learning rate
    snapshot = SnapshotCallbackBuilder(T, M, alpha_zero)

    model_prefix = 'DenseNet-CIFAR100-%d-%d-snapshots' % (40, 12)
    dense_net_model_history = dense_net_model.fit(
                                    generator.flow(trainX, trainY, batch_size=64), 
                                    callbacks=snapshot.get_callbacks(model_prefix=model_prefix), 
                                    epochs=epochs, 
                                    validation_data=(testX, testY_cat)
                                )
else:
    print('Training regular network')
    model_prefix = 'DenseNet-CIFAR100-%d-%d' % (40, 12)
    dense_net_model_history = dense_net_model.fit(generator.flow(trainX, trainY, batch_size=64), epochs=epochs, validation_data=(testX, testY_cat))

# Store the history to a file in the FS
with open('results/' + model_prefix + ' training.csv', mode='w') as f:
    df = pd.DataFrame(dense_net_model_history.history)
    df.to_csv(f)

yPreds = dense_net_model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)
