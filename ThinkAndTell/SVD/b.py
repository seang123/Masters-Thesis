
import numpy as np

with open("./data/pca_subj02_betas_unq_vc.npy", "rb") as f:
    r = np.load(f)

print(r.shape)

