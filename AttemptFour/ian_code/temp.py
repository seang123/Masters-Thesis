import numpy as np
import os, sys
import pandas as pd
from nsd_access import NSDAccess
from collections import defaultdict
import json

nsda = NSDAccess("/home/seagie/NSD2")

df = pd.read_csv("../TrainData/subj02_conditions.csv")

print(df.columns)
print(df.head(1))

shrd = df['nsd_key'].loc[df['is_shared']==1]
unq = df['nsd_key'].loc[df['is_shared']==0]

print("shrd", len(shrd))
print("unq", len(unq))

all_captions = defaultdict(list)
for k in df['nsd_key'].values:
    captions = [i['caption'] for i in nsda.read_image_coco_info(image_index=[int(k)])]
    all_captions[int(k)] = captions

with open("../TrainData/subj02_captions.json", "w") as f:
    json.dump(all_captions, f)

