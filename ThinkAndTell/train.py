# Main training file for fMRI data - LSTM network

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from nsd_access import NSDAccess
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
sys.path.append('/home/seagie/sandbox/Tensorgram/')
import tensorbot as tb
import utils
from model import Encoder, Decoder, CaptionGenerator
import datetime
import traceback
from contextlib import redirect_stdout 
from load_dataset import load_dataset
from param import config as c
import pandas as pd
import nibabel as nb
from nv_monitor import monitor
from parameters import parameters as param
print("imports complete")

gpu_to_use = monitor()
#gpu_to_use = 1

# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')


#### HYPER PARAMETERS ####
BATCH_SIZE = param['BATCH_SIZE']
BUFFER_SIZE = param['BUFFER_SIZE']
test_set_size = 1000
top_k = param['top_k']
vocab_size = top_k + 1

embedding_dim = param['embedding_dim']
units = param['units']


### GLASSER REGIONS ###
print("> loading data")


if os.path.exists("data/visual_mask_rh.npy"):
    with open('data/visual_mask_lh.npy', 'rb') as f, open('data/visual_mask_rh.npy', 'rb') as g:
        visual_mask_lh = np.load(f)
        visual_mask_rh = np.load(g)
        print(" > visual region masks loaded from file") 
else:
    glasser_lh = nb.load(c['GLASSER_LH']).get_data()
    glasser_rh = nb.load(c['GLASSER_RH']).get_data()

    print(glasser_lh.shape)
    print(glasser_rh.shape)

    visual_parcels = pd.read_csv(c['VISUAL_MASK'], index_col=0)
    visual_parcel_list = list(visual_parcels.values.flatten())

    visual_mask_lh = np.zeros(glasser_lh.shape)
    visual_mask_rh = np.zeros(glasser_rh.shape)

    assert len(glasser_lh) == len(glasser_rh)
    print(" > creating visual masks")
    for i in range(0, len(glasser_lh)):
        val_rh = glasser_rh[i, 0, 0]#.squeeze()
        val_lh = glasser_lh[i, 0, 0]#.squeeze()
        visual_mask_rh[i] = 0 if val_rh not in visual_parcel_list else 1
        visual_mask_lh[i] = 0 if val_lh not in visual_parcel_list else 1

    with open('data/visual_mask_lh.npy', 'wb') as f, open('data/visual_mask_rh.npy', 'wb') as g:
        np.save(f, visual_mask_lh, allow_pickle=False)
        np.save(g, visual_mask_rh, allow_pickle=False)
    
    print("> visual area mask created")

visual_mask = np.vstack([visual_mask_lh, visual_mask_rh]).squeeze()
DIM = int(np.sum(visual_mask))


def apply_mask(x, mask):
    # Apply the visual area mask to the verticies
    return x[mask == 1]

def max_length():
    return 75

##### Load image indicies ######

print("> preparing captions")
## get img_indicies for subj02
img_keys = []
with open("img_indicies.txt") as f:
    lines = f.readlines()
    for line in lines:
        img_keys.append(int(line.rstrip('\n')))

annt_dict = utils.load_json("../modified_annotations_dictionary.json")
captions = [] # captions for each image
nr_captions = [] # nr of captions for each image
for i in img_keys:
    caps = annt_dict[str(i)]
    captions.extend(caps)
    nr_captions.append(len(caps))


## Create Tokenizer ##
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = top_k, oov_token='<unk>', filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')

tokenizer.fit_on_texts(captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

train_seqs = tokenizer.texts_to_sequences(captions)

cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=max_length(), padding='post') # (150102, 75)

print("> Tokenizer created!")

##### Create Dataset ######

print("> Creating dataset")
dataset_cap = tf.data.Dataset.from_tensor_slices(cap_vector)

# returns: [betas, dim, subj(1,2,..), sess(1-40), idx, id73k]
dataset_unq = load_dataset("subj02", "unique", nparallel=tf.data.experimental.AUTOTUNE)
dataset_shr = load_dataset("subj02", "shared", nparallel=tf.data.experimental.AUTOTUNE)

## Apply the mask to unique data
dataset_unq = dataset_unq.map(lambda a,b,c: (apply_mask(a, visual_mask), b,c))
dataset_unq = dataset_unq.map(lambda a,b,c: (tf.ensure_shape(a, shape=(DIM,)),b,c))
# Apply mask to shared data
dataset_shr = dataset_shr.map(lambda a,b,c: (apply_mask(a, visual_mask), b,c))
dataset_shr = dataset_shr.map(lambda a,b,c: (tf.ensure_shape(a, shape=(DIM,)),b,c))


## Connect the unique and shared datasets into one ##
dataset_cmp = dataset_unq.concatenate(dataset_shr)

def extend_func(a, b, c):
    l = len(annt_dict[str(c)])
    caps = annt_dict[str(c)]
    seqs = tokenizer.texts_to_sequences(caps)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=max_length(), padding='post') # (150102, 75)

    # Properly reshape things to fit with the proper captions
    a1 = np.tile(a, l).reshape((l, a.shape[0]))
    b1 = np.tile(b, l).reshape((l, 1))
    c1 = np.tile(c, l).reshape((l, 1))
    return (a1, b1, c1, cap_vector)

dataset_cmp = dataset_cmp.map(lambda a, b, c: tf.numpy_function(extend_func, [a, b, c], [tf.float32, tf.int64, tf.int64, tf.int32]))
dataset_cmp = dataset_cmp.flat_map(lambda a,b,c,d: tf.data.Dataset.from_tensor_slices((a,b,c,d)))

## Randomly sample from either dataset (gives non-deterministic output)
#dataset_cmp = tf.data.experimental.sample_from_datasets([dataset_unq, dataset_shr]) 
#dataset_cmp = dataset_cmp.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

#dataset_cmp = dataset_cmp.shuffle(BUFFER_SIZE)

## Split into train/test sets
dataset_test = dataset_cmp.take(test_set_size).shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
dataset_train = dataset_cmp.skip(test_set_size).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
 

print("> dataset created")

#### INSTANTIATE MODEL #####

## Optimizer
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
metric_object = tf.keras.metrics.SparseCategoricalCrossentropy()

## Model 
encoder = Encoder(embedding_dim)
decoder = Decoder(embedding_dim, units, vocab_size)
model = CaptionGenerator(encoder, decoder, tokenizer, max_length)
model.compile(optimizer, loss_object, metric_object, run_eagerly=True)

#### CHECKPOINTS ####

data_path = f"./data/"
## Checkpoints handler
checkpoint_path = f"./checkpoints/train/"
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


### MORE HYPTER-PARAMETERS ###
num_steps = len(captions)//BATCH_SIZE
EPOCHS = param['EPOCHS']
save_checkpoints = True
save_data = True
train_start_time = datetime.datetime.now().strftime('%H:%M:%S - %d/%m/%Y')

parameter_string = f"Parameters:\nBatch Size: {BATCH_SIZE} | embedding dim: {embedding_dim} | units: {units} | vocab size: {vocab_size} | nr batches: {num_steps} | train set: {len(captions)}"
print(parameter_string)

training_loss = []
training_batch_loss = []
testing_loss =[]
testing_batch_loss = []

test_images_idx = []

def main():

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()

        total_epoch_loss = 0
        total_epoch_loss_test = 0
        
        pre_batch_time = 0
        for (batch, (betas, idx, img, cap)) in dataset_train.enumerate():
            
            losses = model.train_step((betas, cap))
            l1, l2 = losses.values()
            total_epoch_loss += l2

            training_loss.append(l2)
            training_batch_loss.append(l1)

            if batch % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch:4} | Loss {(l2):.4f} | {(time.time()-epoch_start-pre_batch_time):.2f} sec")
                pre_batch_time = time.time() - epoch_start

        for (batch, (betas, idx, img, cap)) in dataset_test.enumerate():
            losses = model.test_step((betas, cap))
            l1, l2 = losses.values()
            total_epoch_loss_test += l2

            testing_loss.append(l2)
            testing_batch_loss.append(l1)

            print("batch:", batch)

            if batch % 100 == 0 and batch != 0:
                print(f"Test  {epoch} | Batch {batch:4} | Loss {(l2):.4f} | {(time.time()-epoch_start-pre_batch_time):.2f} sec")

            # on the first epoch save the test image keys for later analysis
            if epoch == 0:
                test_images_idx.append(img.numpy())


        print(f"Epoch {epoch} complete. | Loss {(total_epoch_loss/num_steps):.4f} | Total Time: {(time.time() - epoch_start):.2f} sec")

        if save_checkpoints: 
            ckpt_manager.save()

    return 

def save_loss():
    t_loss = np.array(training_loss)
    t_b_loss = np.array(training_batch_loss)
    with open(f'{data_path}loss_data.npz', 'wb') as f:
        np.savez(f, xtrain=t_loss, ytrain=t_b_loss)

    with open(f'{data_path}test_img_keys.txt', 'w') as f:
        for k in test_images_idx:
            f.write(str(k) + "\n")

def save_model_sum():
    with open(f'{data_path}modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            encoder.summary()
            decoder.summary()
        f.write("\n")
        f.write(parameter_string)
        f.write("\n")
        f.write(f"Total training epochs: {EPOCHS}")
        f.write(f"\nTraining started at: {train_start_time}")
        f.write(f"\nTraining completed at: {datetime.datetime.now().strftime('%H:%M:%S - %d/%m/%Y')}")
        tf.keras.utils.plot_model(model, "model.png", show_shapes=True)



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

        try:
            save_loss()
            print("Loss data saved")
        except Exception as e:
            raise e
            print("Failed to save loss data")

    print("Done.")
