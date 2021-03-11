# Show and Tell implementation

import numpy as np
import matplotlib
matplotlib.use("Agg") # for headerless plot saving
from matplotlib import pyplot as plt
#import tensorflow as tf
import utils as u
#import nsdloader
from nsd_access import NSDAccess
import time

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# init the nsd_loader
nsd_loader = NSDAccess("/home/seagie/NSD")


# images (by index) to plot
img_id = [0,10,20]

tic = time.time()
# load image data
img1 = nsd_loader.read_images(img_id)
# img1 = img1.reshape(425, 425, 3)

print(f"images loaded: {time.time() - tic:.3f}")

tic = time.time()

# get annotation for image
annotations_img1 = nsd_loader.read_image_coco_info(img_id, info_type='captions', show_annot=False, show_img=False)

print(f"annotations loaded: {time.time() - tic:.3f}")

annotations_img1 = u.list_annotations(annotations_img1)
print(annotations_img1)

# plot image
for i in range(0, len(img_id)):
    plt.imshow(img1[i].reshape(425, 425, 3))
    plt.savefig("/home/seagie/NSD/Code/Figures/test_fig"+str(i)+".png", bbox_inches='tight')



print("done.")
