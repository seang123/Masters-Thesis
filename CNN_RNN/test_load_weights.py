import numpy as np
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import utils
import time
import h5py


#weights = np.load("feature_weights.npy", mmap_mode = 'r')

#print(weights.shape)

#print(weights.dtype)

#print(weights[0].shape)

#print("----- memmap -----")

#weights = np.memmap("feature_weights.npy", dtype = np.float32, mode='r')

#print(weights.shape)
#print(weights.dtype)


orig_data = np.random.uniform(0, 1, (1, 64, 10))

new_data = np.random.uniform(0, 1, (1, 64, 10))

#with h5py.File("save_test.hdf5", "w") as f:
#    f.create_dataset('features', data=orig_data, compression="gzip", chunks=True, maxshape=(None,None,None,), dtype=np.float32)

#try to create and append data to hdf5 file
#with h5py.File("save_test.hdf5", 'a') as f:
#    f['features'].resize((f['features'].shape[0] + new_data.shape[0]), axis = 0)
#    f['features'][-new_data.shape[0]:] = new_data


#print("File created and appended.")

print("Reading file")

f = h5py.File('img_features.hdf5', 'r')

print(f['features'].shape)


d = f['features']

x = d[0:5]

print(np.sum(x, (0,1,2)))

