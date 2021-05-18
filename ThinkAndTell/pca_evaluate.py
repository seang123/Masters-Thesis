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
import argparse
from nv_monitor import monitor
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

gpu_to_use = monitor(10000)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')

parser = argparse.ArgumentParser(description="Evaluating LSTM caption network")
parser.add_argument("--save_img", default=True)
parser.add_argument("--n", type=int, default=1)
parser.add_argument("--show", default=True)


max_length = parameters['max_length']
embedding_dim = parameters['embedding_dim']
units = parameters['units']
top_k = parameters['top_k']
vocab_size = top_k + 1
n_samples = 1 # nr. of images to test
        
nsd_loader = NSDAccess("/home/seagie/NSD")
nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)

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

assert os.path.exists("masks/visual_mask_rh.npy")
with open('masks/visual_mask_lh.npy', 'rb') as f, open('masks/visual_mask_rh.npy', 'rb') as g:
    visual_mask_lh = np.load(f)
    visual_mask_rh = np.load(g)

visual_mask = np.vstack([visual_mask_lh, visual_mask_rh]).squeeze()
DIM = int(np.sum(visual_mask))

def apply_mask(x, mask):
    return x[mask==1]


def extend_func(a, c):
    l = len(annt_dict[str(c)])
    caps = annt_dict[str(c)]
    seqs = tokenizer.texts_to_sequences(caps)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=max_length, padding='post')

    # Properly reshape things to fit with the proper captions
    a1 = np.tile(a, l).reshape((l, a.shape[0]))
    c1 = np.tile(c, l).reshape((l, 1))
    return (a1, c1, cap_vector)


with open("./SVD/data/pca_subj02_betas_shr_vc.npy", "rb") as f:
    betas_shr = np.load(f).astype(np.float32)


shr_img_keys = []
with open("./keys/shr_img_keys.txt", "r") as g:
    g_lines = g.readlines()
    for line in g_lines:
        shr_img_keys.append(int(line))


dataset_shr = tf.data.Dataset.from_tensor_slices((betas_shr, shr_img_keys))

dataset_shr = dataset_shr.map(lambda a,b: tf.numpy_function(extend_func, [a,b], [tf.float32, tf.int32, tf.int32]))
dataset_shr = dataset_shr.flat_map(lambda a,b,c: tf.data.Dataset.from_tensor_slices((a,b,c)))

dataset_test = dataset_shr.shuffle(5000)
    

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

def print_pred(pred, target, bscore):
    print("Generated:")
    #print(pred)
    print("  ", " ".join(pred))
    print("Target:")
    print("  ", "".join(" ".join(list(map(lambda i: tokenizer.index_word[i], target[0,:].numpy()))).split("<pad>")[0])) # one-liner to print target caption
    print(f"bleu score: {bscore:.4f}")


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

def bleu_score(candidate, img_key):
    """ Calc. BLEU score for a generated caption

    Parameters
    ----------
        candidate : [words] 
        img_key   : int - used to get the list of references captions
    """

    captions = annt_dict[str(img_key)] # returns list of sentence strings
    # convert to list of words and remove <start> token
    references = []
    for i in captions:
        temp = i.split(" ")
        references.append(temp[1:])

    #score = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    chencherry = SmoothingFunction()
    try:
        score1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0), smoothing_function=chencherry.method4)
        score2 = sentence_bleu(references, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method4)
        score3 = sentence_bleu(references, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method4)
        score4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method4)

        score1i = sentence_bleu(references, candidate, weights=(1, 0, 0, 0))
        score2i = sentence_bleu(references, candidate, weights=(0, 1, 0, 0))
        score3i = sentence_bleu(references, candidate, weights=(0, 0, 1, 0))
        score4i = sentence_bleu(references, candidate, weights=(0, 0, 0, 1))


    except ZeroDivisionError as e:
        print("Bad candidate for bleu score calculation", img_key, candidate, captions[0])
        return [[0, 0, 0, 0], [0,0,0,0]]


    return [[score1, score2, score3, score4],[score1i, score2i, score3i, score4i]]


def simple_eval(betas, target, img_key):
    """Simple evaluation using LSTM stateful=False
    """
    #print("---simple_eval----")

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

    bscores = bleu_score(result, img_key)

    return result, bscores

def save_fig(img_id, pred):
    """ Save .png of image specified by key
    """
    out = " ".join(pred)

    img = nsd_loader.read_images(img_id)
    fig = plt.figure()
    plt.imshow(img)
    plt.title(f"img: {img_id}\n" + out)
    plt.savefig(f"./test_output/test_img_{img_id}.png")
    plt.close(fig)

def main(args):

    bleu_scores = []

    for (batch, (betas, img, cap)) in dataset_test.enumerate():
        print(batch.numpy(), end='\r')

        a = tf.expand_dims(betas, 0)
        b = tf.expand_dims(cap, 0)
        c = img.numpy()[0]

        img_id = img.numpy()[0]

        with tf.device('/cpu'):
            predictions, bscore = simple_eval(a, b, c)
            if args.show == True:
                print(f"\n--img key: {img_id}--")
                print_pred(predictions, b, bscore[0][-1])
            bleu_scores.append(bscore)

        if args.save_img == True:
            save_fig(img_id, predictions)

        if batch >= args.n - 1 and args.n != -1:
            break

    bleu_scores = np.array(bleu_scores) # (n, 2, 4)
    print(bleu_scores.shape)

    print(f"\n\nCumulative BLEU scores:\n") 
    print(f"      \t    min  |   max  |  avg")
    print(f"----------------------------------")
    print(f"BLEU-1:   {(min(bleu_scores[:,0,0])):.4f} | {(max(bleu_scores[:,0,0])):.4f} | {(sum(bleu_scores[:,0,0])/len(bleu_scores)):.4f}")
    print(f"BLEU-2:   {(min(bleu_scores[:,0,1])):.4f} | {(max(bleu_scores[:,0,1])):.4f} | {(sum(bleu_scores[:,0,1])/len(bleu_scores)):.4f}")
    print(f"BLEU-3:   {(min(bleu_scores[:,0,2])):.4f} | {(max(bleu_scores[:,0,2])):.4f} | {(sum(bleu_scores[:,0,2])/len(bleu_scores)):.4f}")
    print(f"BLEU-4:   {(min(bleu_scores[:,0,3])):.4f} | {(max(bleu_scores[:,0,3])):.4f} | {(sum(bleu_scores[:,0,3])/len(bleu_scores)):.4f}")

    print(f"\n\nIndividual BLEU scores:\n") 
    print(f"      \t    min  |   max  |  avg")
    print(f"----------------------------------")
    print(f"BLEU-1:   {(min(bleu_scores[:,1,0])):.4f} | {(max(bleu_scores[:,1,0])):.4f} | {(sum(bleu_scores[:,1,0])/len(bleu_scores)):.4f}")
    print(f"BLEU-2:   {(min(bleu_scores[:,1,1])):.4f} | {(max(bleu_scores[:,1,1])):.4f} | {(sum(bleu_scores[:,1,1])/len(bleu_scores)):.4f}")
    print(f"BLEU-3:   {(min(bleu_scores[:,1,2])):.4f} | {(max(bleu_scores[:,1,2])):.4f} | {(sum(bleu_scores[:,1,2])/len(bleu_scores)):.4f}")
    print(f"BLEU-4:   {(min(bleu_scores[:,1,3])):.4f} | {(max(bleu_scores[:,1,3])):.4f} | {(sum(bleu_scores[:,1,3])/len(bleu_scores)):.4f}")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



