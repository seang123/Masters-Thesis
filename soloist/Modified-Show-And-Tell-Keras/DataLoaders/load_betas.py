
import numpy as np
from nsd_access import NSDAccess
import pandas as pd
import sys
import my_utils as uu

import keras
from keras.utils import to_categorical



# set np seed
np.random.seed(42)

def load_data_betas_partial(load_train = True, load_val = True, shuffle_data = True, top_k = 5000, _max_length = 20):
    """ Partially load betas data

    Parameters
    ----------
        load_train - bool
            if True return train data
        load_val - bool
            if True return val data
        top_k - int
            vocab size
        _max_length - int
            max length of captions (in words)
    Returns
    -------
        data_train
        train_vector
        data_val
        val_vector
        tokenizer
        train_keys
        val_keys

    """
    ## Load the proper NSD keys that correspond to the betas shown
    unq_img_keys = []
    shr_img_keys = []
    #with open("./betas/betas_unq_vc_keys.txt", "r") as f, open("./betas/betas_shr_vc_keys.txt", "r") as g:
    with open("/huge/seagie/betas_nozscore/betas_unq_vc_keys.txt", "r") as f, open("/huge/seagie/betas_nozscore/betas_shr_vc_keys.txt", "r") as g:
        f_lines = f.readlines()
        for line in f_lines:
            unq_img_keys.append(int(line))

        g_lines = g.readlines()
        for line in g_lines:
            shr_img_keys.append(int(line))

    assert( len(unq_img_keys) == 27000 )
    assert( len(shr_img_keys) == 3000  )

    train_keys = unq_img_keys
    val_keys = shr_img_keys

    ## Load annotations
    annt_dict = uu.load_json("../../modified_annotations_dictionary.json")

    ## Split captions into train|test split
    captions_train = []
    for k,i in enumerate(train_keys):
        caps = annt_dict[str(i)]
        for cap in caps:
            # (betas key, NSD key, caption)
            idx_cap = (k, i, cap)
            captions_train.append(idx_cap)

    captions_val = []
    for k,i in enumerate(val_keys):
        caps = annt_dict[str(i)]
        for cap in caps:
            idx_cap = (k, i, cap)
            captions_val.append(idx_cap)


    all_captions = captions_train + captions_val
    all_caps = [i[2] for i in all_captions]

    tokenizer = keras.preprocessing.text.Tokenizer(num_words = top_k, oov_token='<unk>', filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')
    tokenizer.fit_on_texts(all_caps)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    train_vector = None
    val_vector = None
    data_train = None
    data_val = None
    nsd_train_keys = None
    nsd_val_keys = None

    if load_train:
        train_seqs = tokenizer.texts_to_sequences([i[2] for i in captions_train])
        train_vector = keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=_max_length, padding='post')
        print("train_vector:", train_vector.shape)
        #betas_unq = np.load('./betas/betas_unq_vc.npy').astype(np.float32) 
        betas_unq = np.load('/huge/seagie/betas_nozscore/betas_unq_vc.npy').astype(np.float32) 

        data_train = []
        for i in captions_train:
            idx = i[0]
            img = betas_unq[idx]
            data_train.append(img)

        del betas_unq
        data_train = np.array(data_train)

        assert( data_train.shape[0] == train_vector.shape[0] )

        nsd_train_keys = [i[1] for i in captions_train]
        nsd_train_keys = np.array(nsd_train_keys)
        assert( data_train.shape[0] == len(nsd_train_keys) )

        if shuffle_data:
            # Shuffle train data
            shuffle_idx_train = np.arange(data_train.shape[0])
            np.random.shuffle(shuffle_idx_train)

            data_train = data_train[shuffle_idx_train]
            train_vector = train_vector[shuffle_idx_val]
            nsd_train_keys = nsd_train_keys[shuffle_idx_val]

    if load_val:
        val_seqs = tokenizer.texts_to_sequences([i[2] for i in captions_val])
        val_vector = keras.preprocessing.sequence.pad_sequences(val_seqs, maxlen=_max_length, padding='post')
        print("val_vector:", val_vector.shape)

        #betas_shr = np.load('./betas/betas_shr_vc.npy').astype(np.float32) 
        betas_shr = np.load('/huge/seagie/betas_nozscore/betas_shr_vc.npy').astype(np.float32) 

        data_val = []
        for i in captions_val:
            idx = i[0]
            img = betas_shr[idx]
            data_val.append(img)

        del betas_shr
        data_val = np.array(data_val)

        assert( data_val.shape[0] == val_vector.shape[0] )

        nsd_val_keys = [i[1] for i in captions_val]
        nsd_val_keys = np.array(nsd_val_keys)

        assert( data_val.shape[0] == len(nsd_val_keys) )

        if shuffle_data:
            # Shuffle val data
            shuffle_idx_val = np.arange(data_val.shape[0])
            np.random.shuffle(shuffle_idx_val)

            data_val = data_val[shuffle_idx_val]
            val_vector = val_vector[shuffle_idx_val]
            nsd_val_keys = nsd_val_keys[shuffle_idx_val]



    return data_train, train_vector, data_val, val_vector, tokenizer, nsd_train_keys, nsd_val_keys


def load_data_betas(top_k =5000, _max_length = 20):
    """ Load the betas data for the visual cortex map

    Parameters
    ----------
    top_k : int
        vocab size; nr words
    _max_length : int
        Pad, or shorten, all captions to this length

    Returns
    -------

    TODO: bool parameter to allow separate loading of training / testing sets
    """
    np.random.seed(42)

    ## Load the proper NSD keys that correspond to the betas shown
    unq_img_keys = []
    shr_img_keys = []
    with open("./betas/betas_unq_vc_keys.txt", "r") as f, open("./betas/betas_shr_vc_keys.txt", "r") as g:
    #with open("../../ThinkAndTell/keys/unq_img_keys.txt", "r") as f, open("../../ThinkAndTell/keys/shr_img_keys.txt", "r") as g:
    #with open( "/huge/seagie/betas_nozscore/betas_unq_vc_keys.txt", "r" ) as f, open(
    #        "/huge/seagie/betas_nozscore/betas_shr_vc_keys.txt", "r" ) as g:

        f_lines = f.readlines()
        for line in f_lines:
            unq_img_keys.append(int(line))

        g_lines = g.readlines()
        for line in g_lines:
            shr_img_keys.append(int(line))

    assert( len(unq_img_keys) == 27000 )
    assert( len(shr_img_keys) == 3000  )


    train_keys = unq_img_keys
    val_keys = shr_img_keys

    ## Load annotations
    annt_dict = uu.load_json("../../modified_annotations_dictionary.json")

    ## Split captions into train|test split
    temp = 0
    temp2 = 0
    captions_train = []
    for k,i in enumerate(train_keys):
        caps = annt_dict[str(i)]
        temp += len(caps)
        temp2 += 1
        for cap in caps:
            # (betas key, NSD key, caption)
            idx_cap = (k, i, cap)
            captions_train.append(idx_cap)
    print(f"Avg nr. captions - training: {temp/temp2}")

    temp = 0
    temp2 = 0
    captions_val = []
    for k,i in enumerate(val_keys):
        caps = annt_dict[str(i)]
        temp += len( caps )
        temp2 += 1
        for cap in caps:
            idx_cap = (k, i, cap)
            captions_val.append(idx_cap)
    print(f"Avg nr. captions - validation: {temp / temp2}")


    all_captions = captions_train + captions_val
    all_caps = [i[2] for i in all_captions]

    tokenizer = keras.preprocessing.text.Tokenizer(num_words = top_k, oov_token='<unk>', filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')
    tokenizer.fit_on_texts(all_caps)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    
    train_seqs = tokenizer.texts_to_sequences([i[2] for i in captions_train])
    train_vector = keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=_max_length, truncating='post', padding='post')
    print("train_vector:", train_vector.shape)

    val_seqs = tokenizer.texts_to_sequences([i[2] for i in captions_val])
    val_vector = keras.preprocessing.sequence.pad_sequences(val_seqs, maxlen=_max_length, truncating='post', padding='post')
    print("val_vector:", val_vector.shape)


    # TODO: load the full betas vectors
    betas_shr = np.load("../../ThinkAndTell/SVD/data/pca_subj02_betas_shr_vc.npy")#.astype(np.float32)
    betas_unq = np.load("../../ThinkAndTell/SVD/data/pca_subj02_betas_unq_vc.npy")#.astype(np.float32)
    #betas_shr = np.load('./betas/betas_shr_vc.npy').astype(np.float32)
    #betas_unq = np.load('./betas/betas_unq_vc.npy').astype(np.float32)
    #betas_shr = np.load( '/huge/seagie/betas_nozscore/betas_shr_vc.npy' ).astype( np.float32 )
    #betas_unq = np.load( '/huge/seagie/betas_nozscore/betas_unq_vc.npy' ).astype( np.float32 )

    """
    all_betas = np.concatenate((betas_unq, betas_shr))
    # axis 0 takes the mean across voxels | axis 1 takes the mean across a whole brain for a trial
    betas_mean = np.mean(all_betas, axis = 0)
    betas_std = np.std(all_betas, axis = 0)

    print("betas_mean", betas_mean.shape)
    print("betas_std", betas_std.shape)
    print("betas_shr", betas_shr.shape)
    print("betas_unq", betas_unq.shape)

    ## If using axis=0 in mean and std
    betas_shr = (betas_shr - betas_mean) / betas_std
    betas_unq = (betas_unq - betas_mean) / betas_std
    ## if using axis=1 in mean and std
    # betas_mean = betas_mean.reshape((betas_mean.shape[0], 1))
    # betas_std = betas_std.reshape((betas_std.shape[0], 1))

    # all_betas = (all_betas - betas_mean) / betas_std
    # betas_unq = all_betas[:27000]
    # betas_shr = all_betas[27000:]
    """

    # assert( betas_shr.shape[0] == len(val_keys) )
    # assert( betas_unq.shape[0] == len(train_keys))

    ## Create the data arrays
    # For each (betas_idx, NSD_idx, caption) tuple take the betas_idx and save that beta vector to a list
    # this results in duplicated beta vectors which we want since each caption appears ~5 times
    data_train = []
    for i in captions_train:
        idx = i[0]
        img = betas_unq[idx]
        data_train.append(img)

    data_val = []
    for i in captions_val:
        idx = i[0]
        img = betas_shr[idx]
        data_val.append(img)

    data_train = np.array(data_train)
    data_val = np.array(data_val)

    assert( data_train.shape[0] == train_vector.shape[0] )
    assert( data_val.shape[0] == val_vector.shape[0] )

    nsd_train_keys = [i[1] for i in captions_train]
    nsd_val_keys = [i[1] for i in captions_val]
    
    nsd_train_keys = np.array(nsd_train_keys)
    nsd_val_keys = np.array(nsd_val_keys)

    assert( data_train.shape[0] == len(nsd_train_keys) ) 
    assert( data_val.shape[0] == len(nsd_val_keys) )

    ## Shuffle data
    # train data
    """
    shuffle_idx_train = np.arange(data_train.shape[0])
    np.random.shuffle(shuffle_idx_train)
    data_train = data_train[shuffle_idx_train]

    train_vector = train_vector[shuffle_idx_train]
    nsd_train_keys = nsd_train_keys[shuffle_idx_train]

    # val data
    shuffle_idx_val = np.arange(data_val.shape[0])
    np.random.shuffle(shuffle_idx_val)
    data_val = data_val[shuffle_idx_val]

    val_vector = val_vector[shuffle_idx_val]
    nsd_val_keys = nsd_val_keys[shuffle_idx_val]
    """

    ## Different way to shuffle in place
    # train
    rng_state = np.random.get_state()
    np.random.shuffle(data_train)
    np.random.set_state( rng_state )
    np.random.shuffle(train_vector)
    np.random.set_state( rng_state )
    np.random.shuffle(nsd_train_keys)
    # val
    np.random.set_state( rng_state )
    np.random.shuffle(data_val)
    np.random.set_state( rng_state )
    np.random.shuffle(val_vector)
    np.random.set_state( rng_state )
    np.random.shuffle(nsd_val_keys)

    return data_train, train_vector, data_val, val_vector, tokenizer, nsd_train_keys, nsd_val_keys

