import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import DataLoaders.load_avg_betas as loader
from scipy import stats
import sys


x = [[[1,2,3,4], [5,6,7,8]], [[4,3,2,1], [8,7,6,5]]]

y = []
for i in range(len(x)):
    y.append( x[i][0] )

print(y)

raise

df = pd.read_csv('./TrainData/subj01_conditions.csv')

print(len(df['nsd_key']))
print(sum(df['is_shared']))

df = pd.read_csv('./Log/attention_baseline_low_lr_bn/loss_history.csv')

print(df.describe())

sys.exit(0)


vgg_path = '/fast/seagie/data/subj_2/vgg16'
betas_path = '/fast/seagie/data/subj_2/betas_averaged'

df = pd.read_csv('./TrainData/subj02_conditions.csv')
unq_keys, shr_keys = loader.get_nsd_keys("", subject="subj02")


nsd_key = np.concatenate((unq_keys, shr_keys))

betas_batch = np.zeros((1,327684), dtype=np.float32) 

vgg_batch = np.zeros((10000, 4096), dtype=np.float32)

for i, key in enumerate(nsd_key):
    with open(f"{vgg_path}/SUB2_KID{key}.npy", "rb") as f:
        vgg_batch[i,:] = np.load(f)

with open(f"{betas_path}/subj02_KID{nsd_key[0]}.npy", "rb") as f:
    betas_batch[0, :] = np.load(f)

print("betas 0 norm:", np.linalg.norm(betas_batch[0]))

## VGG analysis 
vgg_avg = np.mean(vgg_batch, axis=0)

print("max", "|", "min", "|", "avg", "|", "std")
print(np.max(vgg_avg), "|", np.min(vgg_avg), "|", np.mean(vgg_avg), "|", np.std(vgg_avg))

print("vgg avg norm:", np.linalg.norm(vgg_avg))

fig = plt.figure()
plt.plot(vgg_avg)
plt.savefig('./vgg_avg.png')
plt.close(fig)


### z score data
vgg_zscore = stats.zscore(vgg_batch, axis=1)
print("vgg_zscore:", vgg_zscore.shape)

print("max", "|", "min", "|", "avg", "|", "std")
print(np.max(vgg_zscore), "|", np.min(vgg_zscore), "|", np.mean(vgg_zscore), "|", np.std(vgg_zscore))

vgg_zscore_avg = np.mean(vgg_zscore, axis=0)


fig = plt.figure()
plt.plot(vgg_zscore_avg)
plt.savefig('./vgg_zscore_avg.png')
plt.close(fig)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(betas_batch[0], color='b', alpha = 0.6)
ax2 = ax.twiny()
ax2.plot(vgg_zscore[0], color='r', alpha = 0.6)
plt.savefig('./nsd_6_betas_vgg.png')
plt.close(fig)



