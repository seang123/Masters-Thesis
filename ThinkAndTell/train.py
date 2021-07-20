# Main training file for fMRI data - LSTM network

"""

Trains the model on fMRI data, specifically the betas post GLM processing.

- Currently, masks out only the visual cortex for training ~62k vertices

"""


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from nsd_access import NSDAccess
import sys, os
import json
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
#sys.path.append('/home/seagie/sandbox/Tensorgram/')
#import tensorbot as tb
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
import argparse
from kerastuner.tuners import RandomSearch
from kerastuner import Objective
print("imports complete")
#export TF_CPP_MIN_LOG_LEVEL="3"

gpu_to_use = monitor(10000, gpu_choice=None)

# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')


current_time = datetime.datetime.now().strftime('%H:%M-%d/%m/%Y')
parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--name", type=str, default=current_time)
parser.add_argument("--save_data", default=True)

p_args = parser.parse_args()

#### HYPER PARAMETERS ####
BATCH_SIZE = param['BATCH_SIZE']
BUFFER_SIZE = param['BUFFER_SIZE']
data_folder = p_args.name + '/'
data_path = param['data_path'] + data_folder
test_set_size = 1000
top_k = param['top_k']
vocab_size = top_k + 1

embedding_dim = param['embedding_dim']
units = param['units']

if not os.path.isdir(data_path):
    os.makedirs(data_path)
    print("> created data folder:", data_path)


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
for i in range(0, len(img_keys)):
    key = img_keys[i]
    caps = annt_dict[str(key)]
    captions.extend(caps)
    nr_captions.append(len(caps))

nr_captions_train = sum(nr_captions[:27000])
nr_captions_test = sum(nr_captions[27000:])


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
#dataset_cap = tf.data.Dataset.from_tensor_slices(cap_vector)

use_pca = False
print(f"> using PCA data: {use_pca}")


if use_pca == False:
    # returns: [betas, dim, subj(1,2,..), sess(1-40), idx, id73k]
    dataset_unq = load_dataset("subj02", "unique", nparallel=54)#tf.data.experimental.AUTOTUNE)
    dataset_shr = load_dataset("subj02", "shared", nparallel=54)#tf.data.experimental.AUTOTUNE)

    ## Apply the mask to unique data
    dataset_unq = dataset_unq.map(lambda a,b: (apply_mask(a, visual_mask),b))
    dataset_unq = dataset_unq.map(lambda a,b: (tf.ensure_shape(a, shape=(DIM,)),b))
    # Apply mask to shared data
    dataset_shr = dataset_shr.map(lambda a,b: (apply_mask(a, visual_mask),b))
    dataset_shr = dataset_shr.map(lambda a,b: (tf.ensure_shape(a, shape=(DIM,)),b))
else:
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




## Connect the unique and shared datasets into one ##
#dataset_cmp = dataset_unq.concatenate(dataset_shr)

def extend_func(a, b):
    """
    Parameter
    ---------
        a - betas
        b - img key
    Return
    ------
        (betas * n_captions, img_key * n_captions, n_captions)
    """
    caps = annt_dict[str(b)]
    l = len(caps)
    seqs = tokenizer.texts_to_sequences(caps)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=max_length(), padding='post') # (150102, 75)

    # Properly reshape things to fit with the proper captions
    a1 = np.tile(a, l).reshape((l, a.shape[0]))
    b1 = np.tile(b, l).reshape((l, 1))
    return (a1, b1, cap_vector)


# TODO: split test/validation set here. Doing it afterwards is harder since its possible that brain-data has already been seen, jsut for a different
# caption target
dataset_test = dataset_shr.map(lambda a,b: tf.numpy_function(extend_func, [a,b], [tf.float32, tf.int32, tf.int32])) # pca uses int32 for img
dataset_test = dataset_test.flat_map(lambda a,b,c: tf.data.Dataset.from_tensor_slices((a,b,c)))

## Save validation dataset
if not os.path.exists(f"./data/test_dataset"):
    tf.data.experimental.save(dataset_test, f"./data/test_dataset")
    print(" > Test set saved to disk")

dataset_train = dataset_unq.map(lambda a,b: tf.numpy_function(extend_func, [a,b], [tf.float32, tf.int32, tf.int32]))
dataset_train = dataset_train.flat_map(lambda a,b,c: tf.data.Dataset.from_tensor_slices((a,b,c)))

dataset_test = dataset_test.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
dataset_train = dataset_train.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

print(" > dataset created")

#### INSTANTIATE MODEL #####

# TODO: add lrScheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.001,
    decay_steps=1056,
    decay_rate=0.1,
    staircase=False
)

lr_schedule = tf.keras.experimental.CosineDecay(
    initial_learning_rate=1.0, decay_steps=1056 * param['EPOCHS'], alpha=0.0001, name=None
)

## Optimizer
#optimizer = tf.keras.optimizers.Adam(learning_rate = param['LR'])
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name='RMSprop')
#optimizer = tf.keras.optimizers.Nadam(learning_rate = 0.001)
optimizer = tf.keras.optimizers.SGD(learning_rate = lr_schedule, momentum=0.9, nesterov=False)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
metric_object = tf.keras.metrics.SparseCategoricalCrossentropy()

## Model 
encoder = Encoder(embedding_dim, param['L2'], param['init_method'], param['dropout_fc'])
decoder = Decoder(embedding_dim, units, vocab_size, param['L2_lstm'], param['init_method'], param['dropout_lstm'])

train_callbacks = [
    #tf.keras.callbacks.EarlyStopping(
    #    monitor="val_scce", patience=3, min_delta=0.001, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_scce", factor=0.1, patience=2, verbose=1, min_lr = 0.001)
]

#encoder.build(input_shape=(None, 62756, ))

def build_model(hp):
    #L2_value = hp.Choice('L2', values=[0.0001, 0.001, 0.01, 0.1, 1.0])
    #L2_fc = hp.Choice('L2 FC', values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    #L2_lstm = hp.Choice('L2 LSTM', values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    init_method = param['init_method']
    dropout_fc = hp.Choice('FC dropout', values = [0.1, 0.2, 0.3, 0.5, 0.7])
    dropout_lstm = hp.Choice('LSTM dropout', values = [0.1, 0.2, 0.3, 0.5, 0.7])
    lr = 0.0001
    L2_fc = 1.0
    L2_lstm = 0.0001

    encoder = Encoder(embedding_dim, L2_fc, init_method, dropout_fc)
    decoder = Decoder(embedding_dim, units, vocab_size, L2_lstm, init_method, dropout_lstm)
    model = CaptionGenerator(encoder, decoder, tokenizer, max_length)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss = loss_object, run_eagerly=True)
    return model

def build_model2(init_method):
    """
    Build the model with different initilization methods
    """

    optimizer = tf.keras.optimizers.SGD(learning_rate = lr_schedule, momentum=0.9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    metric_object = tf.keras.metrics.SparseCategoricalCrossentropy()

    ## Model 
    encoder = Encoder(embedding_dim, param['L2'], init_method, param['dropout_fc'])
    decoder = Decoder(embedding_dim, units, vocab_size, param['L2_lstm'], init_method, param['dropout_lstm'])
    model = CaptionGenerator(encoder, decoder, tokenizer, max_length)
    model.compile(optimizer, loss_object, metric_object, run_eagerly=True)
    return model
    

#### CHECKPOINTS ####

start_epoch = 0
## Checkpoints handler
if 1 == 1:
    checkpoint_path = f"{data_path}checkpoints/"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder)
                               #optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

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


### MORE HYPTER-PARAMETERS ###
train_set_size = nr_captions_train #2111 #dataset_train.reduce(np.int32(0), lambda x,_: x + 1).numpy()
test_set_size = nr_captions_test #235 #dataset_test.reduce(np.int32(0), lambda x,_: x + 1).numpy()
num_steps = train_set_size // BATCH_SIZE
num_steps_test = test_set_size // BATCH_SIZE 
EPOCHS = param['EPOCHS']
save_checkpoints = param['SAVE_CHECKPOINT']
save_data = param['SAVE_DATA']
train_start_time = datetime.datetime.now().strftime('%H:%M:%S-%d/%m/%Y')


### LOGS for tensorboard callback
log_dir = 'logs/train'


parameter_string = f"Parameters:\nEpochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | embedding dim: {embedding_dim} | units: {units} | vocab size: {vocab_size} | nr batches: {num_steps} | train set: {nr_captions_train}"
print(parameter_string)

training_loss = [] # just scce loss
training_loss_total = [] # total loss (L2 reg) 
testing_loss =[]
testing_loss_total = []

test_images_idx = []
memory_usage = [] # in MiB

def get_init_weights(init_method_name, init_method):
    """
    For a given initilization method, get the weights at the start and after 1 epoch

    """

    print(f"building model with initlization: {init_method_name}")
    model = build_model2(init_method)

    for (batch, (betas, img, cap)) in dataset_train.enumerate():

        losses = model.train_step((betas, img, cap))
        scce, l2, total_loss, ew, dw = losses.values()

        if batch == 0:
            fc = ew[0].numpy()
            we = dw[0].numpy() # word embedding
            lstm_kernel = dw[1].numpy()
            lstm_r_kernel = dw[2].numpy()
            fc2 = dw[4].numpy()
            fc3 = dw[6].numpy()
            with open(f'{data_path}weights_{init_method_name}_0.npz', 'wb') as f:
                np.savez(f, fc = fc, we = we, lstm_kernel = lstm_kernel, lstm_r_kernel = lstm_r_kernel, fc2 = fc2, fc3 = fc3)

        if batch % 100 == 0:
            print(f"Batch {batch:4} | Scce {(scce):.4f} | L2 {(l2):.4f}")
    
        if batch == 500:
            fc = ew[0].numpy()
            we = dw[0].numpy()
            lstm_kernel = dw[1].numpy()
            lstm_r_kernel = dw[2].numpy()
            fc2 = dw[4].numpy()
            fc3 = dw[6].numpy()
            with open(f'{data_path}weights_{init_method_name}_1.npz', 'wb') as f:
                np.savez(f, fc = fc, we = we, lstm_kernel = lstm_kernel, lstm_r_kernel = lstm_r_kernel, fc2 = fc2, fc3 = fc3)
            break

    print(f"Done. Weights saved for method: {init_method_name}")


def main():

    print("\n## Starting Training ##\n")
    train_start_time = time.time()

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()

        total_epoch_loss = 0
        total_epoch_loss_test = 0
        
        num_steps = 0
        pre_batch_time = 0
        for (batch, (betas, img, cap)) in dataset_train.enumerate():
            num_steps += 1

            #nsd_loader = NSDAccess("/home/seagie/NSD")
            #nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)

            #img_idx = img[0] 
            #img_true = nsd_loader.read_images(img_idx)
            #fig = plt.figure()
            #plt.imshow(img_true)
            #plt.savefig(f"{data_path}test_img.png")
            #plt.close(fig)
            #temp = cap[0].numpy()
            #for i in temp:
            #    print(tokenizer.index_word[i])
            #print("exit !!!!")
            #sys.exit(0)

            losses = model.train_step((betas, img, cap))
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
        for (batch, (betas, img, cap)) in dataset_test.enumerate():
            num_steps_test += 1

            losses = model.test_step((betas, img, cap))
            scce, l2, total_loss = losses.values()
            total_epoch_loss_test += scce

            testing_loss.append(scce)
            testing_loss_total.append(total_loss)

            # on the first epoch save the test image keys for later analysis
            if epoch == 0:
                test_images_idx.append(img.numpy())


        print(f"Test  {epoch} | Loss {(total_epoch_loss_test/num_steps_test):.4f} | {(time.time()-pre_train_time):.2f} sec")

        if save_checkpoints: 
            try:
                ckpt_manager.save()
            except Exception as e:
                print("Failed to store checkpoint")

        print(f"--- Complete {epoch+1} ---")

    print(f"Total elapsed time: {(time.time() - train_start_time):.2f}")

    return 

def save_loss():
    loss_train = np.array(training_loss)
    loss_train_total = np.array(training_loss_total)

    loss_test = np.array(testing_loss)
    loss_test_total = np.array(testing_loss_total)
    with open(f'{data_path}loss_data_{start_epoch}.npz', 'wb') as f:
        np.savez(f, train_loss=loss_train, train_loss_total=loss_train_total, test_loss=loss_test, test_loss_total=loss_test_total)

    with open(f'{data_path}test_img_keys.txt', 'w') as f:
        test_keys = [i for sublist in test_images_idx for i in sublist]
        for k in test_keys:
            f.write(str(k) + "\n")

def save_model_sum():
    with open(f'{data_path}modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            encoder.summary()
            decoder.summary()
    print("Model summary saved to file")

def save_summary(save_model_sum = True):
    with open(f'{data_path}modelsummary.txt', 'w+') as f:
        if save_model_sum == True:
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
        f.write("\n")

    with open(f'{data_path}config.txt', 'w') as f:
        f.write(json.dumps(param)) # store model config dict


def main_tuner():
    print("Running gridsearch")
    tuner = RandomSearch(
            build_model,
            objective=Objective("val_scce", direction="min"),
            max_trials=50,
            executions_per_trial=1,
            directory='tuner_data',
            project_name='L2-Init-Search')

    print("---------")
    tuner.search(dataset_train,
            epochs = 10,
            validation_data=dataset_test)

def main_init_weights():
    ru = tf.keras.initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None)
    gn = tf.keras.initializers.GlorotNormal(seed=None)
    gu = tf.keras.initializers.GlorotUniform(seed=None)
    z  = tf.keras.initializers.Zeros()
    lu = tf.keras.initializers.LecunUniform(seed=None)
    ln = tf.keras.initializers.LecunNormal(seed=None)
    init_methods = {"ru":ru, "gn":gn, "gu":gu, "z":z, "lu":lu, "ln":ln}
    init_methods = {'ru':ru, 'gn':gn, 'gu':gu}
    
    for k, v in init_methods.items():
        get_init_weights(k, v)


if __name__ == '__main__':
    try:
        main()
        #main_init_weights()
        #main_tuner()
        #history_callback = model.fit(dataset_train, validation_data = dataset_test, epochs = param['EPOCHS'], callbacks=train_callbacks)#, steps_per_epoch=num_steps)
        #training_loss = history_callback.history["scce"] 
        #training_loss_total = history_callback.history['loss']
        #testing_loss = history_callback.history['val_scce']
        #testing_loss_total = history_callback.history['val_loss']
    except Exception as e:
        raise e
    except KeyboardInterrupt as e:
        print("--Keyboard Interrupt--")

    if save_data:
        try:
            save_loss()
            print("Loss data saved")
        except Exception as e:
            print("Failed! to save loss data")

        try:
            save_summary(True)
            print("Training/Model summary saved")
        except Exception as e:
            print("Failed! to store training/model summary")
            raise e

    print("Done.")
