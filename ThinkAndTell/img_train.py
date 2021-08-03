# Primary training script for CNN-RNN network.


"""

Trains model on NSD 73k image dataset.
Doesn't use any fMRI data

"""

# imports
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from nsd_access import NSDAccess
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
sys.path.append('/home/seagie/sandbox/Tensorgram/')
import tensorbot as tb
# sys.path.append('C:\\Users\\giess\\OneDrive\\Documents\\University\\Master\\Masters Thesis\\Masters-Thesis')
import utils
import time
import json
import pandas as pd
import logging
import tensorflow as tf
import string
import time
import tqdm
from itertools import zip_longest
import h5py
import random
import collections
from model import Encoder, Decoder, CaptionGenerator
import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from contextlib import redirect_stdout 
import subprocess as sp
import re
import logging
import traceback
import nv_monitor as nv
from parameters import parameters as param
import argparse


gpu_to_use = nv.monitor(9000) 


parser = argparse.ArgumentParser(description="Img training script")
parser.add_argument("--name", type=str, default = 'default')

p_args = parser.parse_args()


data_path = param['data_path'] + p_args.name + "/"

if not os.path.isdir(data_path):
    os.makedirs(data_path)
    print("> created data folder:", data_path)

# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')

print("Loading annotations data...")
# Load the annotations from json
annt_dict = utils.load_json("../modified_annotations_dictionary.json")

train_captions = []

nr_training_samples = 73000
# Put all captions into a single list
for i in range(0, nr_training_samples):
    train_captions.extend(annt_dict[str(i)])

def max_length():
    return param['max_length']

print("Preprocessing annotations...")
# limit our vocab to the top N words
top_k = 5000

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = top_k, oov_token='<unk>', filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')

tokenizer.fit_on_texts(train_captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'


# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)

# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
max_length = max_length()
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=max_length, padding='post')

print("Annotations processed")



#
# Split data into train and testing sets
#
img_name_vector = []
for k, v in annt_dict.items():
    keys = [k for i in range(0, len(v))]
    img_name_vector.extend(keys) 
    # this should  give a 365193 len vector not a 1,7 million length one
    # img_name_vector.extend(k * len(v)) only takes the first char of the key string... 



# Connection between image and caption
img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(img_name_vector, cap_vector):
    img_to_cap_vector[img].append(cap)


## Create training and validation sets using an 80-20 split randomly

# shuffle image keys
img_keys = list(img_to_cap_vector.keys())
random.shuffle(img_keys)

# take 80-20 train-test split
train_percentage = 0.90
slice_index = int(len(img_keys)*train_percentage)
img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]


## if restarting a run from checkpoint - we want to use the same train/validation sets
if os.path.exists(f'{data_path}images_val_idx.txt') and os.path.exists(f'{data_path}images_train_idx.txt'):
    img_name_train_keys = list(np.loadtxt(f'{data_path}images_train_idx.txt', dtype = np.int32, delimiter='\n'))
    img_name_val_keys = list(np.loadtxt(f'{data_path}images_train_idx.txt', dtype = np.int32, delimiter='\n'))


# training captions
img_name_train = []
cap_train = []
for imgt in img_name_train_keys: # loop through test keys (img indicies)
    capt_len = len(img_to_cap_vector[imgt])   # nr of captions for this image
    img_name_train.extend([imgt] * capt_len)  # training img keys * nr captions for that image
    cap_train.extend(img_to_cap_vector[imgt]) # captions for the key

# testing captions | same as above
img_name_val = []
cap_val = []
for imgv in img_name_val_keys:
    capv_len = len(img_to_cap_vector[imgv])
    img_name_val.extend([imgv] * capv_len)
    cap_val.extend(img_to_cap_vector[imgv])


## save the validation set for later analysis
with open(f'{data_path}images_val_keys.txt', 'w') as f, open(f'{data_path}images_train_keys.txt', 'w') as g:
   for i in img_name_val_keys:
       f.write("%s\n" % i) 
   for i in img_name_train_keys:
       g.write("%s\n" % i) 
       
print(f"Train/Test sets generated. {train_percentage:.0%}|{1-train_percentage:.0%} - {len(img_name_train)}|{len(img_name_val)} split")


#
## Training Parameters
#
BATCH_SIZE = param['BATCH_SIZE']
BUFFER_SIZE = param['BUFFER_SIZE'] 
embedding_dim = param['embedding_dim']
units = param['units']
vocab_size = param['top_k'] + 1
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64
use_memmap_array = True # note: this will be very slow if non-primary axis is queried (ie. x[:,:,i] will be very slow (in C order), but x[i,:,:] is very fast)


## Prior to training, load the image features from the hdf5 file
#if use_memmap_array:
#    hdf_img_features = np.memmap(filename="../CNN_RNN/img_features_binary", dtype = 'float32', mode='r', order ='C', shape = (73000, 64, 2048) )
#else:
#    features_file = h5py.File("img_features.hdf5", "r")
#    hdf_img_features = features_file['features'] # (73k, 64, 2048)

hdf_img_features = np.load('../ShowAndTell/img_features_vgg16').astype(np.float32)

def map_func(img_idx, cap):
    """
    Arguments:
        img_idx - image index 
        cap - caption
        hdf_dataset - open hdf5 dataset reference from which image data will be read
    """
    #if use_memmap_array:
    #    img_tensor = hdf_img_features[int(img_idx.decode("utf-8"))]
    #else:
    #    img_tensor = hdf_img_features[img_idx]
    img_tensor = hdf_img_features[int(img_idx)]
    return img_tensor, cap

print("initialising tf.dataset...")

# create tf.dataset
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))


# Use map to load the files in parallel
# TODO: try adding batch before .map
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset_train = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE) 

## Create evaluation dataset
dataset_test = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
dataset_test = dataset_test.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset_test = dataset_test.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


print(f"tf.dataset created")


lr_schedule = tf.keras.experimental.CosineDecay(
    initial_learning_rate=1.0, decay_steps=2500 * param['EPOCHS'], alpha=0.0001, name=None
)


## Optimizer
#optimizer = tf.keras.optimizers.Adam(0.0001)
optimizer = tf.keras.optimizers.SGD(lr_schedule, momentum = 0.9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
metric_object = tf.keras.metrics.SparseCategoricalCrossentropy()

## Init the Encoder-Decoder
encoder = Encoder(embedding_dim, param['L2'], param['init_method'], param['dropout_fc'])
decoder = Decoder(embedding_dim, units, vocab_size, param['L2_lstm'], param['init_method'], param['dropout_lstm'])

# Current time string
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


## Checkpoints handler
checkpoint_path = f"{data_path}checkpoints/"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)

if start_epoch != 0:
    print(f"Checkpoint loaded. Starting from epoch {start_epoch}")
else:
    print(f"No checkpoint loaded.")

model = CaptionGenerator(encoder, decoder, tokenizer, max_length)
model.compile(optimizer, loss_object, metric_object, run_eagerly=True)


parameter_string = f"Parameters\nBatch Size: {BATCH_SIZE} | embedding dim: {embedding_dim} | units: {units} | vocab size: {vocab_size} | nr batches: {num_steps} | train set: {len(img_name_train)} | test set: {len(img_name_val)}"
print("###################")
print(parameter_string)
print("###################")

train_start_time = datetime.datetime.now().strftime('%H:%M:%S - %d/%m/%Y')

# end token
end_token = tokenizer.word_index['<end>']

save_data = True
save_checkpoints = True
run_bleu_tests = False # if True compute BLEU score on test set after epochs

EPOCHS = param['EPOCHS']

training_loss       = []
training_loss_total = []
testing_loss        = []
testing_loss_total  = []
test_images_idx     = []

def main():
    print("\n## Starting Training ##")

    placeholder = tf.constant([0])

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()

        total_epoch_loss = 0
        total_epoch_loss_test = 0
        
        pre_batch_time = 0
        dataset_train_size = 0
        for (batch, (betas, cap)) in dataset_train.enumerate():
            dataset_train_size += betas.shape[0]

            # TODO: test that cap fits image


            losses = model.train_step((betas, placeholder, cap))
            scce, l2, total_loss, _, _ = losses.values()
            total_epoch_loss += scce

            training_loss.append(scce)
            training_loss_total.append(total_loss)

            if batch % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch:4} | Scce {(scce):.4f} | L2 {(l2):.4f} | Loss {(total_loss):.4f} | {(time.time()-epoch_start-pre_batch_time):.2f} sec")
                pre_batch_time = time.time() - epoch_start

        print(f"Train {epoch} | Loss {(total_epoch_loss/num_steps):.4f} | Total Time: {(time.time() - epoch_start):.2f} sec")

        num_steps_test = 0
        pre_train_time = time.time()
        for (batch, (betas, cap)) in dataset_test.enumerate():
            num_steps_test += 1
            losses = model.test_step((betas, placeholder, cap))
            scce, l2, total_loss = losses.values()
            total_epoch_loss_test += scce

            testing_loss.append(scce)
            testing_loss_total.append(total_loss)

            # on the first epoch save the test image keys for later analysis
            if epoch == 0:
                test_images_idx.append(cap.numpy())


        print(f"Test  {epoch} | Loss {(total_epoch_loss_test/num_steps_test):.4f} | {(time.time()-pre_train_time):.2f} sec")

        print(f"--- Complete {epoch} ---")

        if save_checkpoints: 
            ckpt_manager.save()

    return 






def save_loss():
    loss_train = np.array(training_loss)
    loss_train_total = np.array(training_loss_total)

    loss_test = np.array(testing_loss)
    loss_test_total = np.array(testing_loss_total)
    with open(f'{data_path}loss_data_{EPOCHS}.npz', 'wb') as f:
        np.savez(f, train_loss=loss_train, train_loss_total=loss_train_total, test_loss=loss_test, test_loss_total=loss_test_total)


def save_model_sum():
    with open(f'{data_path}modelsummary_{EPOCHS}.txt', 'w') as f:
        with redirect_stdout(f):
            encoder.summary()
            decoder.summary()
        f.write("\n")
        f.write(parameter_string)
        f.write("\n")
        f.write(f"Total training epochs: {EPOCHS}")
        f.write(f"\nTraining started at: {train_start_time}")
        f.write(f"\nTraining completed at: {datetime.datetime.now().strftime('%H:%M:%S - %d/%m/%Y')}")
        #tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

    with open(f'{data_path}config.txt', 'w') as f:
        f.write(json.dumps(param))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        raise e
    except KeyboardInterrupt as e:
        print("--Keyboard Interrupt--")

    if save_data:
        try:
            save_model_sum()
            print("Training/Model summary saved")
        except Exception as e:
            print("Failed to store training/model summary")
            raise e

        try:
            save_loss()
            print("Loss data saved")
        except Exception as e:
            print("Failed to save loss data")

    print("Done.")
