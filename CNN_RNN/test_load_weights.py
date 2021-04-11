import numpy as np
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import utils
import time
import h5py
import tqdm

#weights = np.load("feature_weights.npy", mmap_mode = 'r')

#print(weights.shape)

#print(weights.dtype)

#print(weights[0].shape)

#print("----- memmap -----")

print("Reading memap file")

shape = 73000, 64, 2048
weights = np.memmap(filename="img_features_binary", dtype = 'float32', mode='r', order ='C', shape = shape )

print("memory mapped:", weights.shape)

#print(weights.shape)
#print(weights.dtype)


#orig_data = np.random.uniform(0, 1, (1, 64, 10))

#new_data = np.random.uniform(0, 1, (1, 64, 10))

#with h5py.File("save_test.hdf5", "w") as f:
#    f.create_dataset('features', data=orig_data, compression="gzip", chunks=True, maxshape=(None,None,None,), dtype=np.float32)

#try to create and append data to hdf5 file
#with h5py.File("save_test.hdf5", 'a') as f:
#    f['features'].resize((f['features'].shape[0] + new_data.shape[0]), axis = 0)
#    f['features'][-new_data.shape[0]:] = new_data


#print("File created and appended.")

print("Reading hdf5 file")

f = h5py.File('img_features.hdf5', 'r')

print("hdf5:", f['features'].shape)

g = f['features']

## Randomly load some data
N = 100
idx = np.random.randint(0, 73000, N)

print(f"Time to load {N} random image features")

ls = np.zeros((64, 2048))
start = time.time()
for i in range(0, len(idx)): # takes 131 seconds for 10k samples
    ls += g[idx[i],:,:]

print(f"Hdf: {(time.time() - start):.5f}")


ls = np.zeros((64, 2048))
start = time.time()
for i in range(0, len(idx)): # takes 0.048 seconds
    ls += weights[idx[i],:,:]

print(f"mem: {(time.time() - start):.5f}")


## ----- create mem-mapped file -----
#fp = np.memmap("img_features_binary", dtype='float32', mode='w+', shape=(73000, 64, 2048))

#x = f['features']
#for i in tqdm.tqdm(range(0, x.shape[0])):
#    fp[i,:,:] = x[i, :, :]

#print("done.")
## ---------------------------------

## Create a small datasubset
#data_subset = f['features'][0:5000,:,:]
#print(data_subset.shape)
#with h5py.File("img_features_small.hdf5", "w") as g:
#    g.create_dataset('features', data = data_subset, dtype=np.float32)
