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

# Allow memory growth on GPU devices | Otherwise InceptionV3 won't run due to insufficient memory 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

print("Loading annotations data...")
# Load the annotations from json
annt_dict = utils.load_json("../modified_annotations_dictionary.json")

train_captions = []

# Put all captions into a single list
for i in range(0, 73000):
    train_captions.extend(annt_dict[str(i)])

def max_length(ls):
# return the length of the longest caption - its 260
#    return max(len(t) for t in ls)
    return 260

print("Preprocessing annotations...")
# limit our vocab to the top N words
top_k = 10000

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = top_k, oov_token='<unk>', filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')

tokenizer.fit_on_texts(train_captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'


# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)

# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=max_length(train_captions), padding='post')

max_length = max_length(train_captions)

print("Annotations processed")

#
# Split data into train and testing sets
#
img_name_vector = []
for k, v in annt_dict.items():
    img_name_vector.extend(k * len(v))

# Connection between image and caption
img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(img_name_vector, cap_vector):
    img_to_cap_vector[img].append(cap)

## Create training and validation sets using an 80-20 split randomly

# shuffle image keys
img_keys = list(img_to_cap_vector.keys())
random.shuffle(img_keys)

# take 80-20 train-test split
train_percentage = 0.8
slice_index = int(len(img_keys)*train_percentage)
img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

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

print(f"Train/Test sets generated. {train_percentage:.0%}|{1-train_percentage:.0%} - {len(img_name_train)}|{len(img_name_val)} split")

#
## Training Parameters
#
BATCH_SIZE = 5 # was 64
BUFFER_SIZE = 1000 # shuffle buffer size 
embedding_dim = 256
units = 256 # was 512
vocab_size = top_k + 1
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64

def map_func(img_idx, cap):
    """
    Arguments:
        img_idx - image index 
        cap - caption
        hdf_dataset - open hdf5 dataset reference from which image data will be read
    """
    img_tensor = hdf_img_features[img_idx]
    return img_tensor, cap

print("initialising tf.dataset...")

# create tf.dataset
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
# autotune is returning -1 
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # tf.data.experimental.AUTOTUNE

## Create evaluation dataset
dataset_val = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
dataset_val = dataset_val.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset_val = dataset_val.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) 


print(f"tf.dataset created")

#print("building models...")

## Init the Encoder-Decoder
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

# need to specifiy input shape to print summary here
#encoder.build(input_shape=(BATCH_SIZE, attention_features_shape, features_shape))
# decoder.build(input_shape=(vocab_size, embedding_dim)) # Cannot build model because it takes another input in its call 
#encoder.summary()

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
checkpoint_path = f"./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)

## Store training information
loss_plot = []
loss_plot_test = []
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
            predictions, hidden = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)


    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss

@tf.function # TODO: convert this function to a tf.function using tf.tensorarray
def evaluate(image_tensor, captions):    
    """
    Evalutation function for a single image, and target sentence

    TODO 
    """
    hidden = decoder.reset_state(batch_size = captions.shape[0])

    features = encoder(image_tensor)
 
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * captions.shape[0], 1)
    
    result = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    for i in range(max_length):
        predictions, hidden = decoder(dec_input, features, hidden)

        print("pred:", type(predictions))
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.write(i, predicted_id)

        #if tokenizer.index_word[predicted_id] == '<end>':
        #    break

        dec_input = tf.expand_dims([predicted_id], 1)

    return dec_input


# Loggers for Tensorboard
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

test_log_dir = 'logs/' + current_time + '/test'
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


## Prior to training, load the image features from the hdf5 file
features_file = h5py.File("img_features.hdf5", "r")
hdf_img_features = features_file['features'] # (73k, 64, 2048)

print("###################")
print(f"Parameters\nBatch Size: {BATCH_SIZE} | embedding dim: {embedding_dim} | units: {units} | vocab size: {vocab_size} | nr batches: {num_steps} ")
print("###################")

save_checkpoints = False
EPOCHS = 3
print(f"> Training for {EPOCHS} epochs!")

with tf.device('/gpu:0'):
    for epoch in range(start_epoch, EPOCHS):
        #with tf.profiler.experimental.Trace('train', step_num=epoch, _r=1):
        start = time.time()
        total_loss = 0
    
        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss
                
            train_loss(t_loss) # store for tensorboard display
            loss_plot.append(t_loss) # store loss after every batch
        
            # Record metric
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
        
            if batch % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch} | Loss {(batch_loss.numpy() / int(target.shape[1])):.4f}")
            
            break

        # storing the epoch end loss value to plot later
        #loss_plot.append(total_loss / num_steps)
        
        #if epoch % 5 == 0:
        if save_checkpoints:
            ckpt_manager.save()

        # Evaluate network
        bleu_scores = []
        #for i in tqdm.tqdm(range(0, len(img_name_val))):
        #    # loop through val set

        #    image = [int(img_name_val[i])] # img idx
        #    #real_caption = ' '.join([tokenizer.index_word[j] for j in cap_val[i] if j not in [0]])
        #    prediction = evaluate(image)
        #    print(prediction)
        #    print(type(prediction))
        #    sys.exit(0)
        #    reference = [tokenizer.index_word[j] for j in cap_val[i] if j not in [0]]
            
            # Compute BlEU score
        #    score = sentence_bleu(reference, prediction, smoothing_function = SmoothingFunction().method2)

        #    bleu_scores.append(score)

        for (batch, (img_tensor, reference)) in enumerate(dataset_val):
            prediction = evaluate(img_tensor, reference)
            print("----eval out----") 
            print(type(prediction))
            print(prediction.shape)
            sys.exit(0)

        loss_plot_test.append(bleu_scores)
        test_loss(bleu_scores) # for tensorboard -> avg. of all bleu for this epoch


        print ('Epoch {} Loss {:.6f}'.format(epoch,
                                 total_loss/num_steps))

        print(f"Time taken for epoch {epoch} - {time.time()-start} sec\n")

loss_plot = np.array(loss_plot)
loss_plot_test = np.array(loss_plot_test)

print("saving training data...")
with open('loss_data.npy', 'wb') as f:
    np.savez(f, x=loss_plot, y=loss_plot_test)

#fig = plt.figure()
#plt.plot(loss_plot)
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.title('Loss Plot')
#plt.savefig(f'./figures/{current_time}_loss_plot.png')
#plt.close(fig)

#fig = plt.figure()
#plt.plot(loss_plot_test)
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.title('Test loss Plot')
#plt.savefig(f'./figures/{current_time}_test_loss_plot.png')
#plt.close(fig)

#encoder.summary()
#decoder.summary()

print("## Training Complete. ##")
