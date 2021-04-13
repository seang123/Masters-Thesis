# Primary training script for CNN-RNN network.

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
from model import CNN_Encoder, RNN_Decoder
import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from contextlib import redirect_stdout 
from cython import cython_functions
import subprocess as sp
import re
from check_mem import check_mem
import logging
import traceback

# logging.basicConfig(filename='error.log', force=False, level=logging.DEBUG)


## start telegram bot
bot = tb.Tensorbot()
gpu_var   = bot.register_variable("GPU", "", autoupdate=True)
epoch_var = bot.register_variable("", "", autoupdate=True)
err_var   = bot.register_variable("ERROR:", "", autoupdate=True)

bot.autoupdate("train.py started")

## Check nvidia memory usage, if gpu is free run main 


#gpu_to_use = check_mem()
gpu_to_use = 2

#physcial_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.set_visible_devices(physcial_devices[:1], 'GPU')

# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

#gpus = tf.config.list_physical_devices('GPU')
#if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#    try:
#        tf.config.experimental.set_virtual_device_configuration(
#           gpus[0],
#           [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#    except RuntimeError as e:
#        # Virtual devices must be set before GPUs have been initialized
#        print(e)

print("Loading annotations data...")
# Load the annotations from json
annt_dict = utils.load_json("../modified_annotations_dictionary.json")

train_captions = []

nr_training_samples = 73000
# Put all captions into a single list
for i in range(0, nr_training_samples):
    train_captions.extend(annt_dict[str(i)])

def max_length(ls):
# return the length of the longest caption - its 260
    return max(len(t) for t in ls)
    #return 260

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
max_length = max_length(train_captions)
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
train_percentage = 0.95
slice_index = int(len(img_keys)*train_percentage)
img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]


## if restarting a run from checkpoint - we want to use the same train/validation sets
if os.path.exists('images_val_idx.txt') and os.path.exists('images_train_idx.txt'):
    img_name_train_keys = list(np.loadtxt('images_train_idx.txt', dtype = np.int32, delimiter='\n'))
    img_name_val_keys = list(np.loadtxt('images_train_idx.txt', dtype = np.int32, delimiter='\n'))


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
with open('images_val_keys.txt', 'w') as f, open('images_train_keys.txt', 'w') as g:
   for i in img_name_val_keys:
       f.write("%s\n" % i) 
   for i in img_name_train_keys:
       g.write("%s\n" % i) 
       
print(f"Train/Test sets generated. {train_percentage:.0%}|{1-train_percentage:.0%} - {len(img_name_train)}|{len(img_name_val)} split")


#
## Training Parameters
#
BATCH_SIZE = 64 # was 64
BUFFER_SIZE = 1000 # shuffle buffer size 
embedding_dim = 256 # was 256
units = 512 # was 512
vocab_size = top_k + 1
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64
use_memmap_array = True # note: this will be very slow if non-primary axis is queried (ie. x[:,:,i] will be very slow (in C order), but x[i,:,:] is very fast)


## Prior to training, load the image features from the hdf5 file
if use_memmap_array:
    hdf_img_features = np.memmap(filename="img_features_binary", dtype = 'float32', mode='r', order ='C', shape = (73000, 64, 2048) )
else:
    features_file = h5py.File("img_features.hdf5", "r")
    hdf_img_features = features_file['features'] # (73k, 64, 2048)


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
          num_parallel_calls=tf.data.experimental.AUTOTUNE)#.cache() <- this will cache ~191Gb of data

# Shuffle and batch
# autotune is returning -1 
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # tf.data.experimental.AUTOTUNE

## Create evaluation dataset
dataset_val = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
dataset_val = dataset_val.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset_val = dataset_val.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) 


print(f"tf.dataset created")


## Init the Encoder-Decoder
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

## Optimizer
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

## Loss function
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# Current time string
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


## Checkpoints handler
checkpoint_path = f"./checkpoints/train_attention_large"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
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


## Store training information
loss_plot = []
loss_plot_test = []

# if restarting from checkpoint, load previous loss data
#if os.path.exists('loss_data.npy'):
#    with open('loss_data.npy', 'rb') as f:
#        npyfile = np.load(f)
#        loss_plot = npyfile['x']
#        loss_plot_test = npyfile['y']

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

@tf.function
def train_step(img_tensor, target):
    loss = 0

    start = time.time()

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        # Pass image features through fully-connected
        # in: 64,64,2048
        # out 64,64,256
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)


    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss

@tf.function
def evaluate(image_tensor, captions):    
    """
    Evalutation function for a single image, and target sentence 
    """
    hidden = decoder.reset_state(batch_size = captions.shape[0])

    features = encoder(image_tensor)
 
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * captions.shape[0], 1)

    #print("-------Evaluate--------")
    #print("features:", features.shape) # (32, 64, 256)
    #print("dec_input:", dec_input.shape) # (32, 1)
    #print("hidden:", hidden.shape) # (32, 256)
    
    result = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    for i in range(max_length):
        predictions, hidden, _ = decoder(dec_input, features, hidden, training=False) # prediction: (32, 5001)

        # predicted_id = tf.random.categorical(predictions, 1, dtype = tf.int32)[0][0]#.numpy()
        predicted_id = tf.random.categorical(predictions, 1, dtype = tf.int32)
        result = result.write(i, predicted_id)

        #dec_input = tf.expand_dims([predicted_id], 1)
        dec_input = predicted_id

    return result.stack() # returns (260,32,1)


def bleu_score(predictions, target):
    """
    predictions: index numpy array: (260, 32) # assuming batch_size 32
    reference  : target sentence  : (32, 260) 
    For each sentence, if <end> token is present cut it off there. Otherwise just create a list of words of the 
    prediction which can be compared to the targets
    """
    score = 0

    for i in range(0, predictions.shape[1]):
        sentence_idx = predictions[:,i]

        end_idx = np.where(sentence_idx == end_token)[0]
        if len(end_idx) != 0: # if <end> token is generated then remove(pad to 0) everything after it
            sentence_idx[end_idx[0]+1:] = 0

        sentence_words = [tokenizer.index_word[j] for j in sentence_idx if j not in [0]]

        reference_words = [tokenizer.index_word[j] for j in target[i, :] if j not in [0]]
    
        score += sentence_bleu(reference_words, sentence_words, smoothing_function = SmoothingFunction().method2)

    # return total, and averaged bleu score for the batch
    return (score/predictions.shape[1]), score




# Loggers for Tensorboard
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

test_log_dir = 'logs/' + current_time + '/test'
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


parameter_string = f"Parameters\nBatch Size: {BATCH_SIZE} | embedding dim: {embedding_dim} | units: {units} | vocab size: {vocab_size} | nr batches: {num_steps} | train set: {len(img_name_train)} | test set: {len(img_name_val)}"
print("###################")
print(parameter_string)
print("###################")

# end token
end_token = tokenizer.word_index['<end>']

save_checkpoints = True
run_bleu_tests = False # if True compute BLEU score on test set after epochs

EPOCHS = 30

def main(EPOCHS = 3, gpu = 0):
    print(f"> Training for {EPOCHS}(cmpltd {start_epoch}) epochs!")
    epoch_var.update(f"Starting Training")

    #if start_epoch == 0:
    #    loss_plot = []
    #    loss_plot_test = []

    #with tf.device(logical_gpus[0].name): # /gpu:0 
    with tf.device(f'/gpu:{gpu}'):
        for epoch in range(start_epoch, EPOCHS):
            start = time.time()
            total_loss = 0

            pre_batch_time = 0
            for (batch, (img_tensor, target)) in enumerate(dataset):
                batch_loss, t_loss = train_step(img_tensor, target)
                total_loss += t_loss
                
                #train_loss(t_loss) # store for tensorboard display
                loss_plot.append(t_loss) # store loss after every batch
        
                # Record metric
                #with train_summary_writer.as_default():
                #    tf.summary.scalar('loss', train_loss.result(), step=epoch)
        
                if batch % 100 == 0:
                    print(f"Epoch {epoch} | Batch {batch:4} | Loss {(batch_loss.numpy()/int(target.shape[1])):.4f} | {(time.time()-start-pre_batch_time):.2f} sec")
                    pre_batch_time = time.time() - start

            print(f"Training done for epoch {epoch}")
            epoch_var.update(f"TRAIN: epoch {epoch} done\ntotal_loss: {total_loss:.4f}\ntotal time: {(time.time()-start):.2f} sec")

            print(f"Epoch {epoch} | Loss {(total_loss/num_steps):.6f} | Total Time {(time.time() - start):.2f} sec\n")

            # storing the epoch end loss value to plot later
            #loss_plot.append(total_loss / num_steps)
        
            #if epoch % 5 == 0:
            if save_checkpoints:
                ckpt_manager.save()
       
            pre_batch_time = 0
            if run_bleu_tests:
                # Evaluate network
                bleu_scores = []
                for (batch, (img_tensor, reference)) in enumerate(dataset_val):
                    prediction = evaluate(img_tensor, reference)
                    prediction = prediction.numpy().squeeze() # out: (260, 32) max_length*batch_size
            
                    batch_avg_score, _ = bleu_score(prediction, reference.numpy())
                    # batch_avg_score, score = cython_functions.bleu_score_c(prediction, reference.numpy(), tokenizer)

                    bleu_scores.append(batch_avg_score)

                    if batch % 100 == 0 and batch != 0:
                        print(f"Batch {batch} | last batch avg score {batch_avg_score:.4f} | {(time.time() - start - pre_batch_time):.2f} sec")
                        pre_batch_time = time.time() - start

                loss_plot_test.append(bleu_scores)
                #test_loss(bleu_scores)

                epoch_var.update(f"TEST: epoch {epoch} done\navg batch loss: {np.mean(np.array(bleu_scores)):.4f}\navg time: {(time.time()-start-pre_batch_time):.2f} sec")


    epoch_var.update(f"## Training Complete. ##")
    print("## Training Complete. ##")

    return loss_plot, loss_plot_test

print("saving training data...")

def save_loss(loss_plot, loss_plot_test):
    loss_plot = np.array(loss_plot)
    loss_plot_test = np.array(loss_plot_test)
    with open('loss_data.npy', 'wb') as f:
        np.savez(f, x=loss_plot, y=loss_plot_test)

def save_model_sum():
    ## store model summary
    with open('modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            encoder.summary()
            decoder.summary()
        f.write("\n")
        f.write(parameter_string)
        f.write("\n")
        f.write(f"total training epochs: {EPOCHS}")
#TODO: store training parameters as well (batch size, embedding dim, topk, ....)




if __name__ == "__main__":
    try:
        loss_plot, loss_plot_test = main(EPOCHS, gpu = gpu_to_use)
    except Exception as e:
        print(f"Caught error in main training loop. {datetime.datetime.now().strftime('%H:%M:%S - %d/%m/%Y')}")
        print(f"Logging error to error.log")
        err_var.update("caught during saving of loss and model summary")
        #logging.error('Error at %s', 'division', exc_info=e)
        #logging.error(traceback.format_exc())
        with open("error.log", "w") as f:
            f.write(traceback.format_exc()) 
    except KeyboardInterrupt as e:
        print(f"Caught Keyboard Interrupt")
        print(f"Saving partial data")


    try:
        save_loss(loss_plot, loss_plot_test)
        save_model_sum()
    except Exception as e:
        print("erro caught during saving of loss and model summary")
        err_var.update("caught during saving of loss and model summary")
        bot.kill()

    print("succesfully saved data")
    print("terminating bot process")
    bot.kill()
    print("## Done. ##")
