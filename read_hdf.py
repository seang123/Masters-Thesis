import h5py
import numpy as np
import matplotlib.pyplot as plt


filename = "/home/seagie/NSD/stimuli/nsd_stimuli.hdf5"

with h5py.File(filename, "r") as f:
    # List all groups
    print(list(f.keys()))
    a_group_key = list(f.keys())[0]
    dset = f[a_group_key] # (73k, 425, 425, 3)
    print(dset.shape)
    print(dset.dtype)
    img1 = np.array(dset[0])

    plt.imshow(img1)
    plt.show()


