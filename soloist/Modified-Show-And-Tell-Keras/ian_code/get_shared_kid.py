import numpy as np
import os, sys
import nsd_get_data as ngd

nsd_dir = "/home/seagie/NSD2/"
nsd_dir = os.path.join(nsd_dir)

stim1000 = ngd.get_1000(nsd_dir)

print(len(stim1000))

with open("/huge/seagie/data_meaned/subj02_betas_fsaverage_averaged_shared1000.txt", "w") as f:
    for i in stim1000:
        f.write(f"{i}\n")

