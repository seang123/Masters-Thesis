import numpy as np
import os, sys



with open("/huge/seagie/data/subj_2/betas/betas_SUB2_S9_R9_T9_KID49401.npy", "rb") as f:

    beta = np.load(f)

with open("/huge/seagie/data/subj_2/captions/SUB2_KID49401.txt", "r") as f:

    content = f.read()
    for line in content.splitlines():
        print(line)


print(beta.shape)
