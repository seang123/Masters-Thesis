import h5py
import numpy as np
import matplotlib.pyplot as plt


filename = "/home/seagie/NSD/stimuli/nsd_stimuli.hdf5"
filename2 = "C:\\Users\\giess\\OneDrive\\Documents\\University\\Master\\Masters Thesis\\Data\\nsd_stimuli.hdf5"

with h5py.File(filename2, "r") as f:
    # List all groups
    print(list(f.keys()))
    a_group_key = list(f.keys())[0]
    dset = f[a_group_key] # (73k, 425, 425, 3)

    print(f"{f.name + a_group_key} | {dset.shape} | {dset.dtype}")

    # Plot first 10 images
    imgs = np.array(dset[0:10])
    fig, ax = plt.subplots(nrows=2, ncols=5)

    c = 0
    for row in ax:
        for col in row:
            col.imshow(imgs[c, :, :, 1])
            c += 1

    plt.show()


