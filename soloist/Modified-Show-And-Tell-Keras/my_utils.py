#Script to create a .json file containing the annotations for all 73k images
# This way it doesn't need to be recreated every traning loop.

# Will store the annotations in a dictionary where they key is the image id (idx in the hdf5 file) and the value is a list of all captions for that image.

from nsd_access import NSDAccess
import numpy as np
import json
from pathlib import Path
import os

nsd_loader = NSDAccess("/home/seagie/NSD")


img_ids = np.arange(0, 73000)

# read the images
#images = nsd_loader.read_images(img_ids)

def img_idx_to_id(img_idx):
    """
    Given a (list of) image indicies ( their position in the .hdf5 file ) return their id's
    """
    img_ids = []

    if isinstance(img_idx, list):
        for i in img_idx:
            img_idx.append( )
    else:
        img_ids.append()

    return img_ids


def load_annotations_dict(img_ids: list, key_id = False):
    """
    given image id's (indices in hdf5 file) load the relevant captions into a dictionary key'd by the image id.    

    If key_id = True then use the image ID as the key, otherwise use its index from the .hdf5 file.
    """

    nsd_loader = NSDAccess("/home/seagie/NSD")
    
    annotations = nsd_loader.read_image_coco_info(img_ids, info_type='captions', show_annot=False, show_img=False)

    annot_dict = {}

    for i in range(0, len(annotations)):
        annot_list = []
        img_id = None
        for j in range(0, len(annotations[i])):
            annot_list.append(annotations[i][j]['caption'])
            # get the current images ID
            if j == 0:
                if key_id:
                    img_id = annotations[i][j]['id']
                else:
                    img_id = img_ids[i]

        # store annotations to dictionary    
        annot_dict[img_id] = annot_list

    return annot_dict

def dump_json(data: dict, file_name: str ):
    with open(file_name, 'w') as fp:
        json.dump(data, fp)

def load_json(file_name: str):
    with open(file_name, 'r') as fp:
        data = json.load(fp)
    return data


def append_save_weights_npz(data, file_name: str):
    """
    appends the given data to a .npz file. 

    Args:
        data - np array
        file_name - file name/path without .npz 
    """
    p = Path(file_name + '.npy')
    with p.open('ab') as f:
        np.save(f, data)


def read_npz(file_name: str):
    p = Path(file_name + '.npy')
    with p.open('rb') as f:
        fsz = os.fstat(f.fileno()).st_size
        out = np.load(f)
        while f.tell() < fsz:
            out = np.vstack((out, np.load(f)))
    return out
