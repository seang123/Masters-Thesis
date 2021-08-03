import nsdloader as nsdl
from nsdloader import tfrecord_util as tfu
from config import config
import os, sys
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # do not hog GPUs for creating dataset


# create nsdloader instance
loader = nsdl.nsdloader.NSDLoader(config["NSD_ROOT"])

# file names
fname_unique = "nsd_betas_" + config["SUBJ"] + "_unique"
fname_shared = "nsd_betas_" + config["SUBJ"] + "_shared"


### shared stimuli ###
print("WRITING SHARED STIMULI *****")

shared_stims, _ = loader.create_image_split(test_fraction=0., subset="shared", participant=config["SUBJ"])  # gets cocoids and id73k
trials = loader.trials_for_stim([config["SUBJ"]], shared_stims)

print(trials)
sys.exit(0)

batchloader = loader.load_batch_data_with_info(trials, batchsize=config["TRIALS_PER_RECORD"], load_imgs=False, load_captions=False)



for i,b in enumerate(batchloader):
    num = str(i).zfill(3)
    file = config["WORK_DIR"] + fname_shared + "_" + num + ".tfrecords"
    print(file)
    betas, info = b[0][0], b[1]
    tfu.write_batch_to_tfrecord(betas, info, file)


### unique stimuli ###
print("WRITING UNIQUE STIMULI *****")

unique_stims, _ = loader.create_image_split(test_fraction=0., subset="unique", participant=config["SUBJ"])
trials = loader.trials_for_stim([config["SUBJ"]], unique_stims)

batchloader = loader.load_batch_data_with_info(trials, batchsize=config["TRIALS_PER_RECORD"], load_imgs=False, load_captions=False)
for i,b in enumerate(batchloader):
    num = str(i).zfill(3)
    file = config["WORK_DIR"] + fname_unique + "_" + num + ".tfrecords"
    print(file)
    betas, info = b[0][0], b[1]
    tfu.write_batch_to_tfrecord(betas, info, file)

