
import tensorflow as tf
import numpy as np
from load_dataset import load_dataset

"""

CREATE A .npy FILE HOLDING THE BETAS VECTORS FOR A SUBJECT (IN THIS CASE SUBJ 02)


"""


## === LOAD MASK ===

if os.path.exists("masks/visual_mask_rh.npy"):
    with open('masks/visual_mask_lh.npy', 'rb') as f, open('masks/visual_mask_rh.npy', 'rb') as g:
        visual_mask_lh = np.load(f)
        visual_mask_rh = np.load(g)
        print(" > visual region masks loaded from file") 


visual_mask = np.vstack([visual_mask_lh, visual_mask_rh]).squeeze()
DIM = int(np.sum(visual_mask))


def apply_mask(x, mask):
    # Apply the visual area mask to the verticies
    return x[mask == 1]


## === LOAD CAPTIONS ===

# Not necessary at the moment




## === LOAD DATASET ===

# returns: [betas, dim, subj(1,2,..), sess(1-40), idx, id73k]
dataset_unq = load_dataset("subj02", "unique", nparallel=54)#tf.data.experimental.AUTOTUNE)
dataset_shr = load_dataset("subj02", "shared", nparallel=54)#tf.data.experimental.AUTOTUNE)

## Apply the mask to unique data
dataset_unq = dataset_unq.map(lambda a,b: (apply_mask(a, visual_mask),b))
dataset_unq = dataset_unq.map(lambda a,b: (tf.ensure_shape(a, shape=(DIM,)),b))
# Apply mask to shared data
dataset_shr = dataset_shr.map(lambda a,b: (apply_mask(a, visual_mask),b))
dataset_shr = dataset_shr.map(lambda a,b: (tf.ensure_shape(a, shape=(DIM,)),b))

