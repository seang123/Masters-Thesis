"""

New feature extractor like ../ShowAndTell/feature_extractor.py, but which returns two tf.datasets (split into shr/unq).

"""

import numpy as np
import tensorflow as tf
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import utils
from nsd_access import NSDAccess
from nsdloader import NSDLoader as dans_loader
import pandas as pd


nsd_loader = NSDAccess('/home/seagie/NSD')
nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)

caps = nsd_loader.read_image_coco_info([0]) 

print(caps)

y = nsd_loader.read_behavior('subj02', 1, trial_index=[])

print(y.columns)
print(y.index)
#print(y.head())
#print(y.tail())

df = []

for i in range(1, 41):
    y = nsd_loader.read_behavior('subj02', i, trial_index=[])
    df.append(y)

df = pd.concat(df)
print(df.tail())

# 
#print(nsd_loader.stim_descriptions.columns)










