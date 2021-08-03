
import numpy as np
from nsd_access import NSDAccess
import pandas as pd
import sys
import my_utils as uu

import keras
from keras.utils import to_categorical


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
    tokenizer = keras.preprocessing.text.Tokenizer(num_words = top_k, oov_token='<unk>', filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')

    all_caps = captions_train + captions_val
    all_caps = [item for sublist in all_caps for item in sublist]
    print("total nr. captions:", len(all_caps))
    tokenizer.fit_on_texts(all_caps)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    #all_seqs = tokenizer.texts_to_sequences(all_caps)
    #cap_vector = keras.preprocessing.sequence.pad_sequences(all_seqs, maxlen=_max_length, padding='post') # (150102, 20)

    train_seqs = tokenizer.texts_to_sequences([i for sublist in captions_train for i in sublist])
    train_vector = keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=_max_length, padding='post')
    print("train_vector:", train_vector.shape)

    val_seqs = tokenizer.texts_to_sequences([i for sublist in captions_val for i in sublist])
    val_vector = keras.preprocessing.sequence.pad_sequences(val_seqs, maxlen=_max_length, padding='post')
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


def load_data_full():
    """Load the full data (300k,) vector
    """
    raise NotImplementedError

def load_data_img(top_k = 5000, _max_length = 20, train_test_split = 0.9):
    """Load the VGG 16 image feature data and relevant captions

    Parameters
    ==========
        top_k            - vocab size
        max_length       - clip length of captions
        train_test_split - train and test set split size as percentage
        
    Return
    ======
        data_train    - batch sized training data
        train_vector  - batch sized training captions data
        data_val      -  "
        val_vector    -  "

    TODO: change how data is loaded. For each caption create a tuple with that caption and the relevant feature index.
    """
     
    img_features = np.load('../../ShowAndTell/img_features_vgg16').astype(np.float32) # (73_000, 4096)

    ## Train | Val split keys
    # shuffle the keys
    # Note: this only shuffles the images, the captions relating to an image still show up in order
    keys = np.loadtxt('./shuffled_73k_keys.txt').astype(np.int32)
    #keys = np.arange(0, 73000)
    #np.random.shuffle(keys)
    slice_index = int(len(keys) * train_test_split)
    train_keys, val_keys = keys[:slice_index], keys[slice_index:]

    assert( np.any(keys != np.arange(0, 73000)) ), "keys not shuffled"

    annt_dict = uu.load_json("../../modified_annotations_dictionary.json")

    ## Extract the captions from the data file
    # Split based on shuffled train/val split key
    # Training captions
    captions_train = []
    nr_captions_train = []
    for i in train_keys:
        caps = annt_dict[str(i)]
        captions_train.append(caps)
        nr_captions_train.append(len(caps))

    # Validation captions
    captions_val = []
    nr_captions_val = []
    for i in val_keys:
        caps = annt_dict[str(i)]
        captions_val.append(caps)
        nr_captions_val.append(len(caps))
    

    total_nr_train_captions = sum(nr_captions_train)
    total_nr_val_captions   = sum(nr_captions_val)

    tokenizer = keras.preprocessing.text.Tokenizer(num_words = top_k, oov_token='<unk>', filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')
  
    ## Fit the tokenizer on all the captions (train and val)
    all_caps = captions_train + captions_val
    all_caps = [item for sublist in all_caps for item in sublist]
    assert( isinstance(all_caps[0], str)), "all_caps[0] is not a string"
    print("total nr. captions:", len(all_caps))
    assert(len(all_caps) == (total_nr_train_captions + total_nr_val_captions)), "nr of captions incorrect"
    tokenizer.fit_on_texts(all_caps)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    ## Create the tokenizer sequences for each caption
    train_seqs = tokenizer.texts_to_sequences([i for sublist in captions_train for i in sublist])
    train_vector = keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=_max_length, padding='post')
    print("train_vector:", train_vector.shape)

    val_seqs = tokenizer.texts_to_sequences([i for sublist in captions_val for i in sublist])
    val_vector = keras.preprocessing.sequence.pad_sequences(val_seqs, maxlen=_max_length, padding='post')
    print("val_vector:", val_vector.shape)

    ## Split training and validation feature data
    train_split = img_features[train_keys] # (65700, 4096)
    val_split = img_features[val_keys]

    ## Generate replicated betas data sets
    # Train set
    idx = 0
    data_train = np.zeros((total_nr_train_captions, img_features.shape[1]))
    assert( train_split.shape[0] == len(train_keys) )
    assert( train_split[0].shape == (4096,) )
    for i in range(0, len(train_split)):
        key = train_keys[i]
        nr_caps = nr_captions_train[i]
        #data_rep = np.tile(train_split[i], (nr_caps, 1))
        temp = train_split[i]
        temp = np.expand_dims(temp, 0)
        data_rep = np.repeat(temp, nr_caps, axis = 0)
        data_train[idx:idx+nr_caps] = data_rep
        idx += nr_caps

    assert( data_train.shape[0] == train_vector.shape[0] ), 'train feature and caption vectors are not equal length'

    # Val set
    idx = 0
    data_val = np.zeros((total_nr_val_captions, img_features.shape[1]))
    assert( val_split[0].shape == (4096,) )
    for i in range(0, len(val_keys)):
        key = val_keys[i]
        nr_caps = nr_captions_val[i]
        #data_rep = np.tile(val_split[k], (nr_caps, 1)) # (nr caps, 5000)
        temp = val_split[i]
        temp = np.expand_dims(temp, 0)
        data_rep = np.repeat(temp, nr_caps, axis = 0)
        
        data_val[idx:idx+nr_caps] = data_rep
        idx += nr_caps

    assert( data_val.shape[0] == val_vector.shape[0] ), 'val feature and caption vectors are not equal length'

    print(f"total nr of datapoints: {(data_train.shape[0] + data_val.shape[0]):,}")


    ## shuffle training data
    data_train, train_vector = unison_shuffled_copies(data_train, train_vector)
    data_val, val_vector = unison_shuffled_copies(data_val, val_vector)

    return data_train, train_vector, data_val, val_vector, tokenizer, train_keys, val_keys


def load_captions():
    """
    Return
    ======
        tuple of 365_193 captions in the form of (feature_key, caption)
    """

    annt_dict = uu.load_json("../../modified_annotations_dictionary.json")
    assert( isinstance(annt_dict, dict) )
    total_nr_captions = sum( len(i) for i in annt_dict.values() )
    assert(total_nr_captions > 360000)

    all_captions = [] 
    index_captions = []
    for k,v in annt_dict.items():
        for cap in v:
            index_captions.append( (k, cap) )
            all_captions.append(cap)

    assert(len(all_captions) == total_nr_captions)

    print(f"{total_nr_captions} captions loaded")
    return index_captions, all_captions

def load_data_img2(train_test_split_percentage = 0.9):
    """ Load VGG16 image data
    
    """

    ## Load the pre-shuffled keys (for consistent train/val split)
    shuffled_keys = np.loadtxt("./shuffled_73k_keys.txt")

    slice_index = int(len(keys) * train_test_split_percentage)
    train_keys, val_keys = shuffled_keys[:slice_index], shuffled_keys[slice_index:]


    ## Load the captions
    index_caption, all_captions = load_captions()

    print(index_captions[0])
    print(all_captions[0])

    tokenizer = keras.preprocessing.text.Tokenizer(num_words = top_k, oov_token='<unk>', filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')

    tokenizer.fit_on_texts(all_captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    """## Create the tokenizer sequences for each caption
    train_seqs = tokenizer.texts_to_sequences([i for sublist in captions_train for i in sublist])
    train_vector = keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=_max_length, padding='post')
    print("train_vector:", train_vector.shape)

    val_seqs = tokenizer.texts_to_sequences([i for sublist in captions_val for i in sublist])
    val_vector = keras.preprocessing.sequence.pad_sequences(val_seqs, maxlen=_max_length, padding='post')
    print("val_vector:", val_vector.shape)
    """


    return index_captions, tokenizer




def data_generator(_data, _captions, _unit_size, _vocab_size, _batch_size, training=True): 
    """Generator to load batched data

    Parameter
    =========
        _data        -  data matrix
        _captions    -  captions matrix
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

    TODO: add shuffle ability
    """


    N = _data.shape[0]
    while True:
        for i in range(0, N, _batch_size):

            text = _captions[i:i+_batch_size]
            init_state = np.zeros([text.shape[0], _unit_size])

            # create a target caption - this is just the input shifted left one step/word
            target = np.zeros_like(text)
            target[:,:-1] = text[:, 1:]
            target = to_categorical(target, _vocab_size)

            yield ([_data[i:i+_batch_size], text, init_state, init_state], target)


    # ---- Shuffle -----
    # Not complete, what happens when we get to the last few samples and the _buffer is not able to fill to buffer_size
    # need a way to check how much data is left (ie. rolling data index)
    # Also need a incrementing index to know which data to add to the buffer next
    """
    while True:
        _buffer = []
        # fill buffer
        while len(_buffer) < buffer_size:

            ## Create one datapoint instance
            text = _captions[i]
            init_state = np.zeros([text.shape[0], _unit_size])

            # create a target caption - this is just the input shifted left one step/word
            target = np.zeros_like(text)
            target[:,:-1] = text[:, 1:]
            target = to_categorical(target, _vocab_size)

            _buffer.append(([_data[i], text, init_state, init_state], target))

        for i in range(0, N, _batch_size):
            indicies = np.random.randint(0, len(_buffer), _batch_size) 
            sample = _buffer[indicies]
            del _buffer[indicies]
            yield sample
    """
