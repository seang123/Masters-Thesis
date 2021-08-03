import keras
from tensorflow.python.client import device_lib
import numpy as np


print(keras.__version__)
print(device_lib.list_local_devices())

x = np.arange(0, 73000, 1, dtype=np.int32)
np.random.shuffle(x)

with open("shuffled_73k_cap_keys.txt", 'w+') as f:
    np.savetxt(f, x, fmt = '%i')
print("Done.")
