

from pathlib import Path
import numpy as np
import os

p = Path('./feature_weights.npy')
#with p.open('ab') as f:
#    np.save(f, np.zeros((1, 6, 20)))
#    np.save(f, np.ones((1, 6, 20)))

with p.open('rb') as f:
    fsz = os.fstat(f.fileno()).st_size
    out = np.load(f)
    while f.tell() < fsz:
        out = np.vstack((out, np.load(f)))
    print(out.shape)
        


