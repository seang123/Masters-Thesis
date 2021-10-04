import numpy as np
import os, sys



with open("/huge/seagie/data_meaned/subj02_betas_fsaverage_averaged.npy", "rb") as f:

    beta = np.load(f)


with open("/huge/seagie/data_meaned/subj02_betas_fsaverage_averaged_conditions.txt", "rb") as f:

    content = f.read()
    c = 0
    for line in content.splitlines():
        c += 1

print(beta.shape)
print(c)
