import tensorflow as tf
import numpy as np
import os, sys
import nsdloader as nsdl
from nsdloader import tfrecord_util as tfu
from config import config

loader = nsdl.nsdloader.NSDLoader(config['NSD_ROOT'])

fname_unique = "nsd_betas_subj02_unique"
fname_shared = "nsd_betas_subj02_shared"


#shared_stims, _ = loader.create_image_split(test_fraction=0., subset="shared", participant=config["SUBJ"]) # 1000 x 2
#unique_stims, _ = loader.create_image_split(test_fraction=0., subset="unique", participant="subj02") # 9000 x 2

#trials = loader.trials_for_stim(config["SUBJ"], shared_stims)
#print(trials)


