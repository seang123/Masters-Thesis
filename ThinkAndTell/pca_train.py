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
#import psutils
from nv_monitor import monitor
from parameters import parameters as param
print("imports complete")
#export TF_CPP_MIN_LOG_LEVEL="3"

gpu_to_use = monitor(10000)

# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')


#### HYPER PARAMETERS ####
BATCH_SIZE = param['BATCH_SIZE']
BUFFER_SIZE = param['BUFFER_SIZE']
data_path = f"./data/"
test_set_size = 1000
top_k = param['top_k']
vocab_size = top_k + 1

embedding_dim = param['embedding_dim']
units = param['units']


### GLASSER REGIONS ###
print("> loading data")


if os.path.exists("masks/visual_mask_rh.npy"):
    with open('masks/visual_mask_lh.npy', 'rb') as f, open('masks/visual_mask_rh.npy', 'rb') as g:
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
    return param['max_length']

##### Load image indicies ######

print("> preparing captions")
## get img_indicies for subj02
img_keys = []
with open("./keys/img_indicies.txt") as f:
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

cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=max_length(), padding='post') # (150102, 15)

print("> Tokenizer created!")

##### Create Dataset ######

print("> Creating dataset")

def extend_func(a, c):
    l = len(annt_dict[str(c)])
    caps = annt_dict[str(c)]
    seqs = tokenizer.texts_to_sequences(caps)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=max_length(), padding='post')

    # Properly reshape things to fit with the proper captions
    a1 = np.tile(a, l).reshape((l, a.shape[0]))
    c1 = np.tile(c, l).reshape((l, 1))
    return (a1, c1, cap_vector)


#dataset_test = dataset_test.shuffle(1000, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#dataset_train = dataset_train.shuffle(1000, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

with open("./SVD/data/pca_subj02_betas_shr_vc.npy", "rb") as f, open("./SVD/data/pca_subj02_betas_unq_vc.npy", "rb") as g:
    betas_shr = np.load(f).astype(np.float32)
    betas_unq = np.load(g).astype(np.float32)

unq_img_keys = []
shr_img_keys = []
with open("./keys/unq_img_keys.txt", "r") as f, open("./keys/shr_img_keys.txt", "r") as g:
    f_lines = f.readlines()
    for line in f_lines:
        unq_img_keys.append(int(line))

    g_lines = g.readlines()
    for line in g_lines:
        shr_img_keys.append(int(line))



dataset_unq = tf.data.Dataset.from_tensor_slices((betas_unq, unq_img_keys))
dataset_shr = tf.data.Dataset.from_tensor_slices((betas_shr, shr_img_keys))

dataset_unq = dataset_unq.map(lambda a,b: tf.numpy_function(extend_func, [a,b], [tf.float32, tf.int32, tf.int32]))
dataset_unq = dataset_unq.flat_map(lambda a,b,c: tf.data.Dataset.from_tensor_slices((a,b,c)))

dataset_shr = dataset_shr.map(lambda a,b: tf.numpy_function(extend_func, [a,b], [tf.float32, tf.int32, tf.int32]))
dataset_shr = dataset_shr.flat_map(lambda a,b,c: tf.data.Dataset.from_tensor_slices((a,b,c)))

dataset_train = dataset_unq.shuffle(1000, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
dataset_test = dataset_shr.shuffle(1000, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)




print(" > dataset created")

#### INSTANTIATE MODEL #####

# TODO: add lrScheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.01,
    decay_steps=2111 * param['EPOCHS'],
    decay_rate=0.1,
    staircase=False)

## Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
#optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
metric_object = tf.keras.metrics.SparseCategoricalCrossentropy()

## Model 
encoder = Encoder(embedding_dim)
decoder = Decoder(embedding_dim, units, vocab_size)
model = CaptionGenerator(encoder, decoder, tokenizer, max_length)
model.compile(optimizer, loss_object, metric_object, run_eagerly=True)

#### CHECKPOINTS ####

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
train_set_size = 2111 #dataset_train.reduce(np.int32(0), lambda x,_: x + 1).numpy()
test_set_size = 235 #dataset_test.reduce(np.int32(0), lambda x,_: x + 1).numpy()
num_steps = train_set_size
num_steps_test = test_set_size
EPOCHS = param['EPOCHS']
save_checkpoints = True
save_data = True
train_start_time = datetime.datetime.now().strftime('%H:%M:%S-%d/%m/%Y')


### LOGS for tensorboard callback
log_dir = 'logs/train'


parameter_string = f"Parameters:\nBatch Size: {BATCH_SIZE} | embedding dim: {embedding_dim} | units: {units} | vocab size: {vocab_size} | nr batches: {num_steps} | train set: {num_steps * BATCH_SIZE}"
print(parameter_string)

training_loss = []
training_batch_loss = []
testing_loss =[]
testing_batch_loss = []

test_images_idx = []
memory_usage = [] # in MiB

def main():

    print("\n## Starting Training ##\n")

    for epoch in range(start_epoch, EPOCHS):
        #memory_usage.append(psutil.Process().memory_info().rss / (1024**2))

        epoch_start = time.time()

        total_epoch_loss = 0
        total_epoch_loss_test = 0
        
        pre_batch_time = 0
        dataset_train_size = 0
        for (batch, (betas, img, cap)) in dataset_train.enumerate():
            dataset_train_size += betas.shape[0]

            losses = model.train_step((betas, cap))
            l1, l2 = losses.values()
            total_epoch_loss += l2

            training_loss.append(l2)
            training_batch_loss.append(l1)

            if batch % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch:4} | Loss {(l2):.4f} | {(time.time()-epoch_start-pre_batch_time):.2f} sec")
                pre_batch_time = time.time() - epoch_start

        if epoch == 0:
            print("dataset_train_size:", dataset_train_size)

        print(f"Train {epoch} | Loss {(total_epoch_loss/num_steps):.4f} | Total Time: {(time.time() - epoch_start):.2f} sec")

        pre_train_time = time.time()
        for (batch, (betas, img, cap)) in dataset_test.enumerate():
            losses = model.test_step((betas, cap))
            l1, l2 = losses.values()
            total_epoch_loss_test += l2

            testing_loss.append(l2)
            testing_batch_loss.append(l1)



            # on the first epoch save the test image keys for later analysis
            if epoch == 0:
                test_images_idx.append(img.numpy())


        print(f"Test  {epoch} | Loss {(total_epoch_loss_test/num_steps_test):.4f} | {(time.time()-pre_train_time):.2f} sec")

        print(f"--- Complete {epoch} ---")

        if save_checkpoints: 
            ckpt_manager.save()

    return 

def save_loss():
    t_loss = np.array(training_loss)
    t_b_loss = np.array(training_batch_loss)

    t_loss_test = np.array(testing_loss)
    t_b_loss_test = np.array(testing_batch_loss)
    with open(f'{data_path}loss_data.npz', 'wb') as f:
        np.savez(f, xtrain=t_loss, ytrain=t_b_loss, xtest=t_loss_test, ytest=t_b_loss_test)

    #with open(f'{data_path}test_img_keys.txt', 'w') as f:
    #    test_keys = [i for sublist in test_images_idx for i in sublist]
    #    for k in test_keys:
    #        f.write(str(k) + "\n")

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
        #tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch = '100, 110')
        #model.fit(dataset_train, epochs = 1, steps_per_epoch=num_steps, callbacks=[tb_callback])
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
            print("Failed to save loss data")

    print("Done.")
