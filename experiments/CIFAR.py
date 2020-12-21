# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar100
from tensorflow.keras import backend as K

from models import dense_net

img_rows, img_cols = 32, 32
(trainX, trainY), (testX, testY) = cifar100.load_data()
print(trainX.shape)
plt.imshow(trainX[0])

if K.image_data_format() == "channels_first":
    init = (3, img_rows, img_cols)
else:
    init = (img_rows, img_cols, 3)


########################################################
# %%
# Regular DenseNet model

dense_net_model = dense_net.create_dense_net(nb_classes=2, img_dim=init, depth=40, nb_dense_block=1,
                                growth_rate=12, nb_filter=16, dropout_rate=0.2)

########################################################
# %%

dense_net_snapshot_model = dense_net.create_dense_net(
    nb_classes=2, 
    img_dim=init, 
    depth=40, 
    nb_dense_block=1,
    growth_rate=args.dn_growth_rate, 
    nb_filter=16, 
    dropout_rate=0.2
)

hist = dense_net_snapshot_model.fit(generator.flow(trainX, trainY, batch_size=batch_size), epochs=nb_epoch,
                 callbacks=snapshot.get_callbacks(
                     model_prefix=model_prefix),  # Build snapshot callbacks
                 validation_data=(testX, testY_cat))
