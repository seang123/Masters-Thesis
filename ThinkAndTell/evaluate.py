import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nsd_access import NSDAccess
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import utils
import pandas as pd
from model import Encoder, Decoder, CaptionGenerator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0 everything, 3 nothing
import tensorflow as tf
import json
from parameters import parameters 
from load_dataset import load_dataset



max_length = parameters['max_length']
embedding_dim = parameters['embedding_dim']
units = parameters['units']
top_k = parameters['top_k']
vocab_size = top_k + 1
n_samples = 1 # nr. of images to test

print("> preparing captions")
## get img_indicies for subj02
img_keys = []
with open("img_indicies.txt") as f:
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


if os.path.exists("./tokenizer_config.txt"):
    with open('./tokenizer_config.txt') as json_file:
        json_string = json.load(json_file)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
    print("Tokenizer loaded from config file")
else:
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token='<unk>', filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')
    tokenizer.fit_on_texts(captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

seqs = tokenizer.texts_to_sequences(captions)

cap_vector = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=max_length, padding='post')

#### DATASETS ####
print(f"Creating dataset and sampling {n_samples} values")

assert os.path.exists("data/visual_mask_rh.npy")
with open('data/visual_mask_lh.npy', 'rb') as f, open('data/visual_mask_rh.npy', 'rb') as g:
    visual_mask_lh = np.load(f)
    visual_mask_rh = np.load(g)

visual_mask = np.vstack([visual_mask_lh, visual_mask_rh]).squeeze()
DIM = int(np.sum(visual_mask))

def apply_mask(x, mask):
    return x[mask==1]

dataset_cap = tf.data.Dataset.from_tensor_slices(cap_vector)

# returns: [betas, dim, subj(1,2,..), sess(1-40), idx, id73k]
dataset_unq = load_dataset("subj02", "unique", nparallel=tf.data.experimental.AUTOTUNE)
dataset_shr = load_dataset("subj02", "shared", nparallel=tf.data.experimental.AUTOTUNE)

## Apply the mask to unique data
dataset_unq = dataset_unq.map(lambda a,b,c: (apply_mask(a, visual_mask), b,c))
dataset_unq = dataset_unq.map(lambda a,b,c: (tf.ensure_shape(a, shape=(DIM,)),b,c))
# Apply mask to shared data
dataset_shr = dataset_shr.map(lambda a,b,c: (apply_mask(a, visual_mask), b,c))
dataset_shr = dataset_shr.map(lambda a,b,c: (tf.ensure_shape(a, shape=(DIM,)),b,c))


## Connect the unique and shared datasets into one ##
dataset_cmp = dataset_unq.concatenate(dataset_shr)

def extend_func(a, b, c):
    l = len(annt_dict[str(c)])
    caps = annt_dict[str(c)]
    seqs = tokenizer.texts_to_sequences(caps)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=max_length, padding='post') # (150102, 75)

    # Properly reshape things to fit with the proper captions
    a1 = np.tile(a, l).reshape((l, a.shape[0]))
    b1 = np.tile(b, l).reshape((l, 1))
    c1 = np.tile(c, l).reshape((l, 1))
    return (a1, b1, c1, cap_vector)

dataset_cmp = dataset_cmp.map(lambda a, b, c: tf.numpy_function(extend_func, [a, b, c], [tf.float32, tf.int64, tf.int64, tf.int32]))
dataset_cmp = dataset_cmp.flat_map(lambda a,b,c,d: tf.data.Dataset.from_tensor_slices((a,b,c,d)))

dataset_sample = dataset_cmp.shuffle(1000).batch(1).take(n_samples)


encoder = Encoder(embedding_dim)
#decoder = Decoder(embedding_dim, units, vocab_size, use_stateful=True)
decoder = Decoder(embedding_dim, units, vocab_size)
model = CaptionGenerator(encoder, decoder, tokenizer, max_length)
optimizer = tf.keras.optimizers.Adam()

checkpoint_path = f"./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                   decoder=decoder,
                   optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    # expect_partial() hides the warning that the optimizer state wasn't loaded. This is fine since we arn't interested in training here.

    print(f"Checkpoint loaded! From epoch {start_epoch}")

def print_pred(pred, target):
    print("Generated:")
    print(pred)
    print("Target:")
    print("".join(" ".join(list(map(lambda i: tokenizer.index_word[i], target[0,:].numpy()))).split("<pad>")[0])) # one-liner to print target caption


def forward(betas, dec_input):
    features = encoder(betas)
    x = decoder.embedding(dec_input)
    
    out, h, c = decoder.lstm(x)

    y = decoder.fc1(out)
    y = decoder.fc2(y) # y=(1, 5001)

    return y, h, c

def evaluate(betas, target, k=5):
    """Beam search using stateful LSTM
    eg. k = 2

                    / <fire>
              <the>
            /       \ <car>
           /
    <start>                    ...
           \
            \     / <fire>
              <a>
                  \ <person> 
    """

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * k, 1)
    for i in range(0, dec_input.shape[0]):

        y, h, c = forward(betas, dec_input[i,:])
        # Get K predictions
        pred_id = tf.random.categorical(y, k).numpy()[0]#[0].numpy()

        # get loss for each prediction 


        break 

def simple_eval(betas, target):
    """Simple evaluation using LSTM stateful=False
    """
    print("---simple_eval----")

    features = encoder(betas)
    x, h, c = decoder((target, features)) # x=(1, 75, 5001)

    result = []
    for i in range(max_length):
        samp = tf.expand_dims(x[0,i,:], 0)
        pred_id = tf.random.categorical(samp, 1)[0][0].numpy()
        word = tokenizer.index_word[pred_id]
        result.append(word)
        if word == '<end>':
            break

    print_pred(result, target)
    return


for (batch, (betas, idx, img, cap)) in dataset_sample.enumerate():
    
    print("img", img.numpy())
    with tf.device('/cpu'):
        predictions = simple_eval(betas, cap)
    #evaluate(betas, cap)




