# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_model_prefix = lambda train_snapshot, m, t, e, lr: f"RNN-gas-{'snapshot-' if train_snapshot else ''}{m}M-{t}T-{e}E-{lr}lr"
args = [
    [True, 10, 10, 400, 0.1],
    [True, 10, 10, 400, 0.01],
    [True, 10, 10, 400, 0.001],
    [False, 10, 10, 400, 0.001],
]


data = pd.DataFrame()

for a in args:
    instance_data = pd.DataFrame()
    for i in range(1, a[1] + 1):
        instance = pd.read_csv(f'{get_model_prefix(*a)}/{i}-lc.csv')
        instance_data = pd.concat([instance_data, instance], axis=1)\
            .rename(columns={'Accuracy': f'accuracy ({a})', 'Validation Accuracy': f"Validation Accuracy ({f'Snapshot, lr={a[4]}' if a[0] else 'Single'})"})\
            .drop([f'accuracy ({a})'], axis=1)
    data = pd.concat([data, instance_data], axis=1)
    
# snapshot_data = pd.DataFrame()
# for i in range(1, 21):
#     instance = pd.read_csv(f'{get_model_prefix(True, *args)}{i}.csv')
#     snapshot_data = pd.concat([snapshot_data, instance], axis=1)

data = data.drop(['Unnamed: 0'], axis=1)
# snapshot_data = snapshot_data.drop(['Unnamed: 0'], axis=1)\
#     .rename(columns={'accuracy': 'accuracy (Snapshot)', 'val_accuracy': 'validation accuracy (Snapshot)'})

# data = pd.concat([single_data], axis=1)
ax, fig = sns.lineplot(data=data, err_style=None, dashes=False) # , err_style="bars", ci=68 for bars
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig.savefig("learning-rate.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
# %%
