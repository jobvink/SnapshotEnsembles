# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_model_prefix = lambda train_snapshot, m, t, e, lr: f"RNN-gas-{'snapshot-' if train_snapshot else ''}{m}M-{t}T-{e}E-{lr}lr"
args = 10, 20, 400, 0.1

single_data = pd.DataFrame()
for i in range(1, args[1] + 1):
    instance = pd.read_csv(f'{get_model_prefix(False, *args)}/{i}-lc.csv')
    single_data = pd.concat([single_data, instance], axis=1)
    
# snapshot_data = pd.DataFrame()
# for i in range(1, 21):
#     instance = pd.read_csv(f'{get_model_prefix(True, *args)}{i}.csv')
#     snapshot_data = pd.concat([snapshot_data, instance], axis=1)

single_data = single_data.drop(['Unnamed: 0'], axis=1)
# snapshot_data = snapshot_data.drop(['Unnamed: 0'], axis=1)\
#     .rename(columns={'accuracy': 'accuracy (Snapshot)', 'val_accuracy': 'validation accuracy (Snapshot)'})

data = pd.concat([single_data], axis=1)
sns.lineplot(data=data) # , err_style="bars", ci=68 for bars
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
# %%
