import numpy as np
import os, sys
import pandas as pd
from nsd_access import NSDAccess
from collections import defaultdict
import json

"""
subject 02 has 34 trials that have 6 captions the rest have 5
"""


nsda = NSDAccess("/home/seagie/NSD")

loc = "/huge/seagie/data/subj_2/captions"

df = pd.read_csv("../TrainData/subj02_conditions.csv")

print(df.columns)
print(df.head(1))

shrd = df['nsd_key'].loc[df['is_shared']==1]
unq = df['nsd_key'].loc[df['is_shared']==0]

print("shrd", len(shrd))
print("unq", len(unq))

captions = [i for i in nsda.read_image_coco_info(image_index=df['nsd_key'].values)] # [[{}]]
for k, v in enumerate(df['nsd_key'].values):
    cap = [j['caption'] for j in captions[k]]
    cap = cap[:5]
    with open(f'{loc}/SUB2_KID{v}.txt', "w") as f:
        for c in cap:
            f.write(f"{c}\n")


