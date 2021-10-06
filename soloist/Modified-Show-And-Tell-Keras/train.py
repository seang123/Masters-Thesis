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

#from TensorBoardCaption import TensorBoardCaption

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
from DataLoaders import data_loader
from DataLoaders import load_betas
from parameters import params_dir

from DataLoaders import new_load_betas as betas_loader
from DataLoaders import load_avg_betas

# set np seed
np.random.seed(42)


#keras.backend.set_session(keras.backend.tf.Session(config=keras.backend.tf.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)))




def train(_epochs, _input_size, _unit_size, _batch_size = 128, _max_length = 20, in_reg = 1e-4, lstm_reg=1e-4, out_reg=1e-4, _lr = 0.001, _decay=0., _initial_epoch=0):
    """ Main training function

    """
    data_dir = params_dir['data_dir']
    
    #data_train, train_vector, data_val, val_vector, tokenizer, train_keys, val_keys = data_loader.load_data_img(_max_length = _max_length, train_test_split = 0.9)
    #data_train, train_vector, data_val, val_vector, tokenizer, train_keys, val_keys, ext_train_keys, ext_val_keys = data_loader.load_data_betas(_max_length = _max_length)

    #data_train, train_vector, data_val, val_vector, tokenizer, train_keys, val_keys = load_betas.load_data_betas_partial(load_train=True, load_val=True, shuffle_data=False)
    
    #data_train, train_vector, data_val, val_vector, tokenizer, train_keys, val_keys = load_betas.load_data_betas(top_k = 5000, _max_length = _max_length)

    captions_path = "/huge/seagie/data/subj_2/captions/"
    betas_path    = "/huge/seagie/data/subj_2/betas_meaned/"

    nsd_dir = '/home/seagie/NSD2/'

    """
    tokenizer, _ = betas_loader.build_tokenizer(captions_path, params_dir['top_k'])
    nsd_keys_all, nsd_keys_set_train, nsd_keys_set_val = betas_loader.get_nsd_keys(betas_path)
    train_pairs = betas_loader.create_pairs(nsd_keys_set_train, betas_path, captions_path)
    val_pairs = betas_loader.create_pairs( nsd_keys_set_val, betas_path, captions_path )

    train_generator = betas_loader.batch_generator(train_pairs, betas_path, captions_path, tokenizer, params_dir['batch_size'], params_dir['max_length'], params_dir['top_k'], params_dir['units'])
    val_generator = betas_loader.batch_generator(val_pairs, betas_path, captions_path, tokenizer, params_dir['batch_size'], params_dir['max_length'], params_dir['top_k'], params_dir['units'])
    """
    tokenizer, _ = load_avg_betas.build_tokenizer(captions_path, params_dir['top_k'])
    nsd_keys, _ = load_avg_betas.get_nsd_keys(nsd_dir)
    shr_nsd_keys = load_avg_betas.get_shr_nsd_keys(nsd_dir)

    print("nsd_keys", len(nsd_keys))
    print("shr_nsd_keys", len(shr_nsd_keys))

    train_keys = [i for i in nsd_keys if i not in shr_nsd_keys]
    val_keys = shr_nsd_keys

    train_pairs = load_avg_betas.create_pairs(train_keys, captions_path)
    val_pairs = load_avg_betas.create_pairs(val_keys, captions_path)

    print("train_pairs", len(train_pairs))
    print("val_pairs", len(val_pairs))

    train_generator = load_avg_betas.batch_generator(train_pairs, betas_path, captions_path, tokenizer,
                                                     params_dir['batch_size'],
                                                     params_dir['max_length'],
                                                     params_dir['top_k'],
                                                     params_dir['units'])
    val_generator = load_avg_betas.batch_generator(val_pairs, betas_path, captions_path, tokenizer,
                                                     params_dir['batch_size'],
                                                     params_dir['max_length'],
                                                     params_dir['top_k'],
                                                     params_dir['units'])


    #with open( 'train_keys.txt', 'w') as f:
    #    for i in range(0, len(train_keys)):
    #            f.write(f"{train_keys[i]}\n")

    #print("data_train", data_train.shape)
    #print("train_vector", train_vector.shape)


    
    vocab_size = tokenizer.num_words
    print("vocab_size", vocab_size)

    # train_generator = data_loader.data_generator(data_train, train_vector, train_keys, _unit_size = _unit_size, _vocab_size=vocab_size, _batch_size = _batch_size)
    # val_generator = data_loader.data_generator(data_val, val_vector, val_keys, _unit_size = _unit_size, _vocab_size=vocab_size, _batch_size = _batch_size)

    ## Define model
    NIC_model = model(vocab_size, _input_size, _max_length, in_reg, lstm_reg, out_reg)
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
    #epoch_history = custom_callbacks.EpochHistory()

    print("steps per epoch training:", len(train_pairs)/_batch_size)
    print("steps per epoch validation:", len(val_pairs)/_batch_size)

    try:
        hist = NIC_model.fit_generator(train_generator, 
                steps_per_epoch = len(train_pairs)//_batch_size, 
                epochs = _epochs, 
                verbose = 1, 
                validation_data = val_generator,
                validation_steps = len(val_pairs)//_batch_size,
                callbacks=[checkpoint2, history, csv_logger, early_stop, batch_loss],
                initial_epoch = _initial_epoch)
    except Exception as e:
        raise e
    except KeyboardInterrupt as e:
        print("--Keyboard Interrupt--")

    print(hist.params)
    print(hist.history.keys())
    print(NIC_model.losses)


if __name__ == "__main__":

    nsd_loader = NSDAccess("/home/seagie/NSD")
    nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)


    if not os.path.isdir(params_dir['data_dir']):
        os.makedirs(params_dir['data_dir'])
        print("> created data folder:", params_dir['data_dir'])

    train(_epochs = params_dir['epochs'], _input_size = params_dir['input'], _unit_size = params_dir['units'], _batch_size = params_dir['batch_size'], _max_length = params_dir['max_length'], in_reg = params_dir['L2_reg'], lstm_reg = params_dir['LSTM_reg'], out_reg = params_dir['out_reg'], _lr = params_dir["LR"], _decay = params_dir['lr_decay'], _initial_epoch = 0)













