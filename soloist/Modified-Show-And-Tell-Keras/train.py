'''
File to train the NIC model, based on the paper:

https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf
'''

import keras
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History 
from keras.callbacks import CSVLogger
import callbacks as custom_callbacks
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils import to_categorical
import tensorflow as tf

from NIC import model
from preprocessing.text import create_tokenizer
from utils import batch_generator

from TensorBoardCaption import TensorBoardCaption

import numpy as np
import sys, os
#sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import my_utils as uu
from nsd_access import NSDAccess
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

#from data_loader import load_data_pca, data_generator, load_data_img
import data_loader
import load_betas
from parameters import params_dir

# set np seed
np.random.seed(42)


#keras.backend.set_session(keras.backend.tf.Session(config=keras.backend.tf.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)))




def train(_epochs, _input_size, _unit_size, _batch_size = 128, _max_length = 20, _reg = 1e-4, _lr = 0.001, _decay=0., _initial_epoch=0):
    """ Main training function

    """
    data_dir = params_dir['data_dir']
    
    #data_train, train_vector, data_val, val_vector, tokenizer, train_keys, val_keys = data_loader.load_data_img(_max_length = _max_length, train_test_split = 0.9)
    #data_train, train_vector, data_val, val_vector, tokenizer, train_keys, val_keys, ext_train_keys, ext_val_keys = data_loader.load_data_betas(_max_length = _max_length)

    #data_train, train_vector, data_val, val_vector, tokenizer, train_keys, val_keys = load_betas.load_data_betas_partial(load_train=True, load_val=True, shuffle_data=False)
    
    data_train, train_vector, data_val, val_vector, tokenizer, train_keys, val_keys = load_betas.load_data_betas(top_k = 5000, _max_length = _max_length)

    train_keys_set = list(set(train_keys)) # unq keys
    train_keys_set_split = int(len(train_keys_set) * 0.9) # split unq keys
    train_keys_set_1 = train_keys_set[:train_keys_set_split] # 8100
    train_keys_set_2 = train_keys_set[train_keys_set_split:] # 900

    print("train_keys_set_1", len(train_keys_set_1))
    print("train_keys_set_2", len(train_keys_set_2))


    #train_x = np.array([np.where(train_keys == i)[0] for i in train_keys_set_1]) # (8100,)
    #train_y = np.array([np.where(train_keys == i)[0] for i in train_keys_set_2]) # (900,)

    train_x = []
    train_y = []
    for k, v in enumerate(train_keys):
        if v in train_keys_set_1:
            train_x.append(k)
        elif v in train_keys_set_2:
            train_y.append(k)
        else:
            raise Exception("oops")


    train_x = np.array(train_x)
    train_y = np.array(train_y)
    print("train_x", train_x.shape)
    print("train_y", train_y.shape)

    data_val = data_train[train_y]
    val_vector = train_vector[train_y]
    val_keys = train_y

    data_train = data_train[train_x]
    train_vector = train_vector[train_x]
    train_keys = train_x
    
    print("data_train   ", data_train.shape)
    print("train_vector ", train_vector.shape)
    print("train_keys   ", train_keys.shape)


    """
    train_set = int(data_train.shape[0] * 0.9)
    print("train_set:", train_set)

    data_val = data_train[train_set:,:]
    val_vector = train_vector[train_set:,:]
    val_keys = train_keys[train_set:]

    data_train = data_train[:train_set,:]
    train_vector = train_vector[:train_set,:]
    train_keys = train_keys[:train_set]
    """


    #with open( 'train_keys.txt', 'w') as f:
    #    for i in range(0, len(train_keys)):
    #            f.write(f"{train_keys[i]}\n")

    print("data_train", data_train.shape)
    print("train_vector", train_vector.shape)

    # unique_rows = np.unique( data_train, axis = 0 ) # (27000, 5000)
    # unique_rows = np.unique( train_vector, axis = 0) # (44043, 10)

    #sentences = tokenizer.sequences_to_texts( train_vector )
    #print("sentences", len(sentences))
    #ts = []
    #for k, v in enumerate(sentences):
    #    ts.append(v)
    
    #for i in range(10):
    #    print(ts[i])

    #c = 0
    #with open( 'train_captions.txt', 'w' ) as f:
    #    for item in ts:
    #        c += 1
    #        f.write( "%s\n" % item )
    #print(c)
    #sys.exit(0)
    
    vocab_size = tokenizer.num_words
    print("vocab_size", vocab_size)

    train_generator = data_loader.data_generator(data_train, train_vector, train_keys, _unit_size = _unit_size, _vocab_size=vocab_size, _batch_size = _batch_size)
    val_generator = data_loader.data_generator(data_val, val_vector, val_keys, _unit_size = _unit_size, _vocab_size=vocab_size, _batch_size = _batch_size)

    ## Define model
    NIC_model = model(vocab_size, _input_size, _max_length, _reg)
    print(NIC_model.summary())

    NIC_model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = _lr, decay = _decay), metrics = [keras.metrics.CategoricalAccuracy(name='accuracy')])

    # Checkpoint name template
    file_path = params_dir['data_dir'] + '/model-ep{epoch:03d}-loss{loss:.4f}-val_loss{val_loss:.4f}.h5'
    file_path_latest = params_dir['data_dir'] + '/latest-model.h5'
    checkpoint = ModelCheckpoint(file_path, 
            monitor='val_loss', 
            verbose=1, 
            save_weights_only = True, 
            save_best_only = True,
            mode = 'min',
            period=1)
    checkpoint2 = ModelCheckpoint(file_path_latest,
            monitor='loss', #'val_loss', 
            verbose=1, 
            save_weights_only = True, 
            save_best_only = False,
            mode = 'min',
            period=1)
    history = History()
    batch_loss = custom_callbacks.BatchLoss(f'batch_training_log.csv', data_dir)
    csv_logger = CSVLogger(f'{data_dir}/training_log.csv')
    early_stop = custom_callbacks.EarlyStoppingByLossVal('loss', value=0.05)

    print("steps per epoch training:", data_train.shape[0]/_batch_size)
    print("steps per epoch validation:", data_val.shape[0]/_batch_size)

    try:
        NIC_model.fit_generator(train_generator, 
                steps_per_epoch = data_train.shape[0]//_batch_size, 
                epochs = _epochs, 
                verbose = 1, 
                validation_data = val_generator,
                validation_steps = data_val.shape[0]//_batch_size,
                callbacks=[checkpoint2, history, csv_logger, early_stop, batch_loss],
                initial_epoch = _initial_epoch)
    except Exception as e:
        raise e
    except KeyboardInterrupt as e:
        print("--Keyboard Interrupt--")



if __name__ == "__main__":

    nsd_loader = NSDAccess("/home/seagie/NSD")
    nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)


    if not os.path.isdir(params_dir['data_dir']):
        os.makedirs(params_dir['data_dir'])
        print("> created data folder:", params_dir['data_dir'])

    train(_epochs = params_dir['epochs'], _input_size = params_dir['input'], _unit_size = params_dir['units'], _batch_size = params_dir['batch_size'], _max_length = params_dir['max_length'], _reg = params_dir['L2_reg'], _lr = params_dir["LR"], _decay = params_dir['lr_decay'], _initial_epoch = 0)













