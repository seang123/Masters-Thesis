

import numpy as np
from nsd_access import NSDAccess
import pandas as pd
import sys
import my_utils as uu

import keras
from keras.utils import to_categorical
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

nsd_loader = NSDAccess("/home/seagie/NSD")

import data_loader as dl


def verify_cap_img():
    """
    VERIFY THAT THE ANNOTATIONS MATCH THE IMAGES FROM THE NSD_LOADER
    They do in fact match!!!
    """
    annt_dict = uu.load_json("../../modified_annotations_dictionary.json")

    key = 70000
    print(annt_dict[str(key)])

    img = nsd_loader.read_images([key])
    img = img[0]

    fig = plt.figure()
    plt.imshow(img)
    plt.title(f"{key}")
    plt.savefig(f"./img_{key}.png")
    plt.close(fig)


data_train, train_vector, data_val, val_vector, tokenizer, train_keys, val_keys = load_data_img(_max_length=20)


generator = data_generator(data_train, train_vector, _unit_size = units, _vocab_size = vocab_size, _batch_size = 10, training=False)

#for i in generator:
#    [features, cap, a0, c0], target = i



