import sys
sys.path.append("/home/seagie/NSD/Code/Masters-Thesis/soloist/Modified-Show-And-Tell-Keras/")
import numpy as np
from nsd_access import NSDAccess
import pandas as pd
import my_utils as uu

import tensorflow as tf
from tensorflow.keras.utils import to_categorical


# set np seed
np.random.seed(42)


def unison_shuffled_copies(a, b):
    """Shuffle two np arrays in the same way
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def load_data_pca(_max_length = 20):
    """
    betas unq has 135090 captions
    betas shr has 15012
    """

    with open("../../ThinkAndTell/SVD/data/pca_subj02_betas_shr_vc.npy", "rb") as f, open("../../ThinkAndTell/SVD/data/pca_subj02_betas_unq_vc.npy", "rb") as g:
        betas_shr = np.load(f).astype(np.float32)
        betas_unq = np.load(g).astype(np.float32)

    unq_img_keys = []
    shr_img_keys = []
    with open("../../ThinkAndTell/keys/unq_img_keys.txt", "r") as f, open("../../ThinkAndTell/keys/shr_img_keys.txt", "r") as g:
        f_lines = f.readlines()
        for line in f_lines:
            unq_img_keys.append(int(line))

        g_lines = g.readlines()
        for line in g_lines:
            shr_img_keys.append(int(line))

    img_keys = []
    with open("../../ThinkAndTell/keys/img_indicies.txt") as f:
        lines = f.readlines()
        for line in lines:
            img_keys.append(int(line.rstrip('\n')))

    annt_dict = uu.load_json("../../modified_annotations_dictionary.json")
    captions_train = []
    nr_captions_train = []
    for i in range(0, len(unq_img_keys)):
        caps = annt_dict[str(unq_img_keys[i])]
        captions_train.append(caps)
        nr_captions_train.append(len(caps))

    captions_val = []
    nr_captions_val = []
    for i in range(0, len(shr_img_keys)):
        caps = annt_dict[str(shr_img_keys[i])]
        captions_val.append(caps)
        nr_captions_val.append(len(caps))

    print("betas unq:", betas_unq.shape, "nr captions:", len(captions_train), "total captions:", sum(nr_captions_train))
    print("betas shr:", betas_shr.shape, "nr captions:", len(captions_val), "total captions:", sum(nr_captions_val))

    total_nr_train_captions = sum(nr_captions_train)
    total_nr_val_captions   = sum(nr_captions_val)

    # First init the tokenizer
    top_k = 5000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = top_k, oov_token='<unk>', filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')

    all_caps = captions_train + captions_val
    all_caps = [item for sublist in all_caps for item in sublist]
    print("total nr. captions:", len(all_caps))
    tokenizer.fit_on_texts(all_caps)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    #all_seqs = tokenizer.texts_to_sequences(all_caps)
    #cap_vector = tf.keras.preprocessing.sequence.pad_sequences(all_seqs, maxlen=_max_length, padding='post') # (150102, 20)

    train_seqs = tokenizer.texts_to_sequences([i for sublist in captions_train for i in sublist])
    train_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=_max_length, padding='post')
    print("train_vector:", train_vector.shape)

    val_seqs = tokenizer.texts_to_sequences([i for sublist in captions_val for i in sublist])
    val_vector = tf.keras.preprocessing.sequence.pad_sequences(val_seqs, maxlen=_max_length, padding='post')
    print("val_vector:", val_vector.shape)

    
    ## Generate replicated betas data sets
    # Test set
    idx = 0
    data_train = np.zeros((total_nr_train_captions, betas_unq.shape[1]))
    for i in range(0, betas_unq.shape[0]):
        nr_caps = nr_captions_train[i]
        data_rep = np.tile(betas_unq[i], (nr_caps, 1)) # (nr caps, 5000)
        
        data_train[idx:idx+nr_caps] = data_rep
        idx += nr_caps

    # Validation set
    idx = 0
    data_val = np.zeros((total_nr_val_captions, betas_shr.shape[1]))
    for i in range(0, betas_shr.shape[0]):
        nr_caps = nr_captions_val[i]
        data_rep = np.tile(betas_shr[i], (nr_caps, 1))
        data_val[idx:idx+nr_caps] = data_rep
        idx += nr_caps

    print("data shpe")
    print("train:",data_train.shape)
    print("val:  ",data_val.shape)

    return data_train, train_vector, data_val, val_vector, tokenizer



def load_data_img(top_k = 5000, _max_length = 20, train_test_split = 0.9):
    """ Re-worked load_data_img method

    Currently testing it. Thing it will work better.

    """

    img_features = np.load('/home/seagie/NSD/Code/Masters-Thesis/ShowAndTell/img_features_vgg16').astype(np.float32) # (73_000, 4096)

    ## Train | Val split keys
    # shuffle the keys
    # Note: this only shuffles the images, the captions relating to an image still show up in order
    keys = np.loadtxt('/home/seagie/NSD/Code/Masters-Thesis/soloist/Modified-Show-And-Tell-Keras/shuffled_73k_keys.txt').astype(np.int32)
    #keys = np.arange(0, 73000)
    #np.random.shuffle(keys)
    slice_index = int(len(keys) * train_test_split)
    train_keys, val_keys = keys[:slice_index], keys[slice_index:]

#    assert( np.any(keys != np.arange(0, 73000)) ), "keys not shuffled"

    annt_dict = uu.load_json("/home/seagie/NSD/Code/Masters-Thesis/modified_annotations_dictionary.json")

    ## Extract the captions from the data file
    # Split based on shuffled train/val split key
    # Training captions
    captions_train = []
    for i in train_keys:
        caps = annt_dict[str(i)]
        for cap in caps:
            idx_cap = (i, cap)
            captions_train.append(idx_cap)

    captions_val = []
    for i in val_keys:
        caps = annt_dict[str(i)]
        for cap in caps:
            idx_cap = (i, cap)
            captions_val.append(idx_cap)


    all_captions = captions_train + captions_val
    all_caps = [i[1] for i in all_captions]

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = top_k, oov_token='<unk>', filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')
    tokenizer.fit_on_texts(all_caps)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    
    train_seqs = tokenizer.texts_to_sequences([i[1] for i in captions_train])
    train_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=_max_length, truncating='post', padding='post')
    print("train_vector:", train_vector.shape)

    val_seqs = tokenizer.texts_to_sequences([i[1] for i in captions_val])
    val_vector = tf.keras.preprocessing.sequence.pad_sequences(val_seqs, maxlen=_max_length, truncating='post', padding='post')
    print("val_vector:", val_vector.shape)

    data_train = []
    for i in captions_train:
        idx = i[0]
        img = img_features[idx]
        data_train.append(img)

    data_val = []
    for i in captions_val:
        idx = i[0]
        img = img_features[idx]
        data_val.append(img)
    
    data_train = np.array(data_train)
    data_val   = np.array(data_val)
    print("data_train:",data_train.shape)
    print("data_val:",data_val.shape)


    assert( data_train.shape[0] == train_vector.shape[0])
    assert( data_val.shape[0] == val_vector.shape[0])

    ext_train_keys = [i[0] for i in captions_train]
    ext_val_keys = [i[0] for i in captions_val]

    ext_train_keys = np.array(ext_train_keys)
    ext_val_keys = np.array(ext_val_keys)
    
    assert( data_train.shape[0] == len(ext_train_keys) ) 
    assert( data_val.shape[0] == len(ext_val_keys) )

    """
    ## Shuffle data
    # train data
    shuffle_idx_train = np.arange(data_train.shape[0])
    np.random.shuffle(shuffle_idx_train)
    data_train = data_train[shuffle_idx_train]

    train_vector = train_vector[shuffle_idx_train]
    ext_train_keys = ext_train_keys[shuffle_idx_train]

    # val data
    shuffle_idx_val = np.arange(data_val.shape[0])
    np.random.shuffle(shuffle_idx_val)
    data_val = data_val[shuffle_idx_val]

    val_vector = val_vector[shuffle_idx_val]
    ext_val_keys = ext_val_keys[shuffle_idx_val]
    """

    rng_state = np.random.get_state()
    np.random.shuffle( data_train )
    np.random.set_state( rng_state )
    np.random.shuffle( train_vector )
    np.random.set_state( rng_state )
    np.random.shuffle( ext_train_keys )
    # val
    np.random.set_state( rng_state )
    np.random.shuffle( data_val )
    np.random.set_state( rng_state )
    np.random.shuffle( val_vector )
    np.random.set_state( rng_state )
    np.random.shuffle( ext_val_keys )

    return data_train, train_vector, data_val, val_vector, tokenizer, ext_train_keys, ext_val_keys

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

    """

    ## Load the proper NSD keys that correspond to the betas shown
    unq_img_keys = []
    shr_img_keys = []
    with open("./betas/betas_unq_vc_keys.txt", "r") as f, open("./betas/betas_shr_vc_keys.txt", "r") as g:
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

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = top_k, oov_token='<unk>', filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')
    tokenizer.fit_on_texts(all_caps)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    
    train_seqs = tokenizer.texts_to_sequences([i[2] for i in captions_train])
    train_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=_max_length, padding='post')
    print("train_vector:", train_vector.shape)

    val_seqs = tokenizer.texts_to_sequences([i[2] for i in captions_val])
    val_vector = tf.keras.preprocessing.sequence.pad_sequences(val_seqs, maxlen=_max_length, padding='post')
    print("val_vector:", val_vector.shape)


    # TODO: load the full betas vectors
#    betas_shr = np.load('./betas/betas_shr_vc.npy', mmap_mode = 'r').astype(np.float32) 
#    betas_unq = np.load('./betas/betas_unq_vc.npy', mmap_mode = 'r').astype(np.float32)
    betas_shr = np.memmap('./betas/betas_shr_vc.npy', dtype='float32', mode='r', shape = (3000, 62756))
    betas_unq = np.memmap('./betas/betas_unq_vc.npy', dtype='float32', mode='r', shape = (27000, 62756))

    assert( betas_shr.shape[0] == len(val_keys) )
    assert( betas_unq.shape[0] == len(train_keys))

    ## Create the data arrays | Replicate betas 'n' times
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

    assert( data_train.shape[0] == train_vector.shape[0] ), f"{data_train.shape[0]} == {train_vector.shape[0]}"
    assert( data_val.shape[0] == val_vector.shape[0] ), f"{data_val.shape[0]} == {val_vector.shape[0]}"

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
    

    return data_train, train_vector, data_val, val_vector, tokenizer, None, None, nsd_train_keys, nsd_val_keys







def data_generator(_data, _captions, _keys, _unit_size, _vocab_size, _batch_size, training=True): 
    """Generator to load batched data

    Parameter
    =========
        _data        -  data matrix
        _captions    -  captions matrix
        _keys        -  key corresponding to dataset image
        _unit_size   -  size of hidden units in LSTM/embedding
        _vocab_size  -  size of vocabulary
        _batch_size  -  batch size to load
        training     -  whether the generated dataset is for training or validation
    Return
    ======
        tuple
            list
                - batched data
                - batched captions
                - initial hidden state
                - initial carry state
            - target caption
            - NSD keys

    TODO: add shuffle ability
    """
    rng_state = np.random.get_state()

    N = _data.shape[0]

    assert( len(_keys) == N ), f"{len(_keys)} != {N}"

    #while True:
    for i in range(0, N, _batch_size):

        # The relevant captions for the batch
        text = _captions[i:i+_batch_size]

        # Inital states needed for the lstm
        init_state = np.zeros([text.shape[0], _unit_size], dtype=np.float32)

        # create a target caption - this is just the input shifted left one step/word
        target = np.zeros_like(text)
        target[:,:-1] = text[:, 1:]
        target = to_categorical(target, _vocab_size)
        target = target.astype(np.int32)

        keys = _keys[i:i+_batch_size].astype(np.int32)

        if training:
            # NSD 73k keys not necessary for training
            yield ([_data[i:i+_batch_size].astype(np.float32), text, init_state, init_state], target)
        else:
            # Also return the associated NSD 73k keys to allow image plotting
            yield ([_data[i:i+_batch_size].astype(np.float32), text, init_state, init_state], target, keys)

        # Shuffle idea:
        # shuffle the data here at the end of the for loop
        # Once the for loop is over, one epoch has passed so we can reshuffle the data here
        # np.random.set_state( rng_state )
        # np.random.shuffle(_data)
        # np.random.set_state( rng_state )
        # np.random.shuffle( _captions )
        # np.random.set_state( rng_state )
        # np.random.shuffle( _keys )


if __name__ == '__main__':
    print("hi")
