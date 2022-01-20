import cortex

import nibabel as nb
import numpy as np

np.random.seed(1234)

# gather data Volume
volume = cortex.Volume.random(subject='S1', xfmname='fullhead')

# create viewer
cortex.webgl.show(data=volume)

