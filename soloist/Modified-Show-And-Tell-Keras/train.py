'''
File to train the NIC model, based on the paper:

https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf
'''

import keras
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History 
from keras.callbacks import CSVLogger
from callbacks import BatchHistory
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils import to_categorical

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

from data_loader import load_data_pca, data_generator, load_data_img
import data_loader
from parameters import params_dir

#keras.backend.set_session(keras.backend.tf.Session(config=keras.backend.tf.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)))




def train(_epochs, _input_size, _unit_size, _batch_size = 128, _max_length = 20, _reg = 1e-4, _lr = 0.001, _decay=0., _initial_epoch=0):
    """ Main training function

    """
    data_dir = params_dir['data_dir']
    
    #data_train, train_vector, data_val, val_vector, tokenizer = load_pca_data(_max_length = _max_length)
    data_train, train_vector, data_val, val_vector, tokenizer, _, _ = load_data_img(_max_length = _max_length, train_test_split = 0.9)
    
    vocab_size = tokenizer.num_words
    print("vocb_size", vocab_size)

    train_generator = data_generator(data_train, train_vector, _unit_size = _unit_size, _vocab_size=vocab_size, _batch_size = _batch_size)
    val_generator = data_generator(data_val, val_vector, _unit_size = _unit_size, _vocab_size=vocab_size, _batch_size = _batch_size)

    [features, text, a0, c0], target = train_generator.__next__()

    ## Define model
    NIC_model = model(vocab_size, _input_size, _max_length, _reg)
    print(NIC_model.summary())

    NIC_model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = _lr, decay = _decay), metrics = ['accuracy'])

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
            monitor='val_loss', 
            verbose=1, 
            save_weights_only = True, 
            save_best_only = False,
            mode = 'min',
            period=1)
    history = History()
    batch_history = BatchHistory()
    csv_logger = CSVLogger(f'{data_dir}/training_log.csv')

    print("steps per epoch", data_train.shape[0]/_batch_size)

    try:
        NIC_model.fit_generator(train_generator, 
                steps_per_epoch = data_train.shape[0]//_batch_size, 
                epochs = _epochs, 
                verbose = 1, 
                validation_data = val_generator,
                validation_steps = 500, 
                callbacks=[checkpoint, checkpoint2, history, batch_history, csv_logger],
                initial_epoch = _initial_epoch)
    except Exception as e:
        raise e
    except KeyboardInterrupt as e:
        print("--Keyboard Interrupt--")

    batch_epoch_loss = batch_history.get_loss()

    # Store per-epoch loss history
    with open(f'{data_dir}/loss_epoch.json', 'w+') as f:
        json.dumps(history.history, f)
    with open(f'{data_dir}/loss_batch_epoch.json', 'w+') as f:
        json.dumps(batch_epoch_loss, f)


if __name__ == "__main__":

    nsd_loader = NSDAccess("/home/seagie/NSD")
    nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)


    if not os.path.isdir(params_dir['data_dir']):
        os.makedirs(params_dir['data_dir'])
        print("> created data folder:", params_dir['data_dir'])

    train(_epochs = params_dir['epochs'], _input_size = params_dir['input'], _unit_size = params_dir['units'], _batch_size = params_dir['batch_size'], _max_length = params_dir['max_length'], _reg = params_dir['L2_reg'], _lr = params_dir["LR"], _decay = 0., _initial_epoch = 0)













