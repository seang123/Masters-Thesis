
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import yaml
import pandas as pd
import time
import os, sys
import tensorflow as tf
import numpy as np
import tqdm
from Model import NIC, lc_NIC
from DataLoaders import load_avg_betas as loader
from DataLoaders import data_generator_guse as generator
from nsd_access import NSDAccess
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu
from tabulate import tabulate
import argparse

gpu_to_use = 0

# Allow memory growth on GPU device
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')

parser = argparse.ArgumentParser(description='Evaluate NIC model')
parser.add_argument('--dir', type=str, required=True)
args = parser.parse_args()

#with open("./Log/attention_test_loader_2/config.yaml", "r") as f:
with open(f"./Log/{args.dir}/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    print(f"Config file loaded:\n\t {f.name}")

run_name = config['run']
out_path = os.path.join(config['log'], run_name, 'eval_out')

np.random.seed(config['seed'])
tf.random.set_seed(config['seed'])

## Parameters
vocab_size = config['top_k'] + 1
batch_size = 32

if not os.path.exists(out_path):
    os.makedirs(out_path)
    print(f"Creating evaluation output dir:\n\t{out_path}")
else:
    print(f"Evaluation output to dir:\n\t{out_path}")


## Load data
tokenizer, _ = loader.build_tokenizer([2], config['top_k'])

train_keys, val_keys = loader.get_nsd_keys('2')
print("train_keys:", train_keys.shape)
print("val_keys:", val_keys.shape)

train_pairs = loader.create_pairs(train_keys)
val_pairs   = loader.create_pairs(val_keys)

def remove_dup_pairs(pairs):
    """ Remove duplicates from the pairs list, based on NSD key """
    print("Removing duplicates from pairs list ... ")
    return list({v[0]:v for v in pairs}.values())

train_pairs = remove_dup_pairs(train_pairs)
val_pairs   = remove_dup_pairs(val_pairs)

np.random.shuffle(val_pairs)

print(f"train_pairs: {len(train_pairs)}")
print(f"val_pairs  : {len(val_pairs)}")

#val_generator = create_generator(train_pairs, training=False)
val_generator = generator.DataGenerator(
        val_pairs,
        batch_size, 
        tokenizer, 
        config['units'], 
        config['max_length'], 
        vocab_size, 
        pre_load_betas=False,
        shuffle=False, 
        training=False)

#x = val_generator[0]
#x2 = val_generator[1]

print("data loaded successfully")

## Set-up model
model = lc_NIC.NIC(
        #loader.get_groups(config['embedding_features'])[0],
        #loader.get_groups(config['embedding_features'])[1],
        loader.get_groups(32)[0],
        loader.get_groups(32)[1],
        config['units'],
        config['embedding_features'], 
        config['embedding_text'],
        config['attn_units'],
        vocab_size,
        config['max_length'],
        config['dropout_input'],
        config['dropout_features'],
        config['dropout_text'],
        config['dropout_attn'],
        config['dropout_lstm'],
        config['input_reg'],
        config['attn_reg'],
        config['lstm_reg'],
        config['output_reg']
        )

x = np.random.uniform(0, 1, size=(config['batch_size'], config['input']['full'])).astype(dtype=np.float32)
y = np.random.randint(0, 2, size=(config['batch_size'], config['max_length']), dtype=np.int32)
z1 = np.random.uniform(0, 1, size=(config['batch_size'], config['units'])).astype(dtype=np.float32)
z2 = np.random.uniform(0, 1, size=(config['batch_size'], config['units'])).astype(dtype=np.float32)
model((x,y,z1,z2, None), training=False)
print("model built")


## Restore model from Checkpoint
model_dir = f"{os.path.join(config['log'], config['run'])}/model/model-latest.h5"
model_dir = f"{os.path.join(config['log'], config['run'])}/model/model-ep010.h5"

model.load_weights(model_dir,by_name=True,skip_mismatch=True)
#model.load_weights(model_dir)
print(f"Model weights loaded")
print(f" - from {model_dir}")

nsd_loader = NSDAccess("/home/seagie/NSD3/")
nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)
print("NSDAccess loader initialized ... ")

def targets_to_sentences(targets: np.array) -> list:
    """ Returns a list of target sentences

    Parameters
    ----------
        target : ndarray
            the target caption as one-hot encoded words 
    Returns:
        sentence : [string]
    """
    return tokenizer.sequences_to_texts(np.argmax(targets, axis=2))


def compute_bleu(candidate):
    """ Compute the blue score for a given candidate and its reference nsd key """

    caption, key = candidate

    weights = [
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        (1./1., 0, 0, 0),
        (1./2., 1./2., 0, 0),
        (1./3., 1./3., 1./3., 0),
        (1./4., 1./4., 1./4., 1./4.)     
    ]

    chencherry = SmoothingFunction()

    # get reference set
    #references = [i['caption'] for i in nsd_loader.read_image_coco_info([int(keys[i])-1])]
    references = []
    with open(f"{config['dataset']['captions_path']}/SUB2_KID{key}.txt", "r") as f:
        content = f.read()
        for line in content.splitlines():
            cap = line.replace(".", " ")
            cap = cap.replace(",", " ")
            cap = cap.strip()
            cap = cap.split(" ")
            cap = [i.lower() for i in cap]
            cap = cap #+ ['<end>']
            references.append(cap)

    caption_split = caption.split(" ")
    try:
        caption_split.remove('<pad>')
    except ValueError:
        print("no '<pad>' token in caption")

    bleus = []
    for w in weights:
        bleus.append( sentence_bleu(references, caption_split, weights=w, smoothing_function=chencherry.method0) )

    #print("-------")
    #print("caption:\n\t", caption_split)
    #print("references:\n")
    #for ref in references:
    #    print("\t", ref)
    #print(bleus)
    return bleus
    

def eval_full_set():
    """ Evaluate the model the entire validation set """
    nsd_loader = NSDAccess("/home/seagie/NSD3")
    nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)

    bleu_scores = []
    candidates = []
    for i in tqdm.tqdm(range(0, len(val_generator)+1)):
        sample = val_generator[i]
        features, _, a0, c0, _ = sample[0]
        target = sample[1]
        keys = sample[2]

        ## Create start word
        start_seq = np.repeat([tokenizer.word_index['<start>']], features.shape[0])

        ## Call model
        outputs = model.greedy_predict(features, tf.convert_to_tensor(a0), tf.convert_to_tensor(c0), start_seq, config['max_length'], config['units'], tokenizer) 
        # outputs: (max_len, bs, 1, 5001)
        # outputs: (max_len, bs, 1)
        outputs  = np.squeeze(outputs)
        captions = np.transpose(outputs, axes=[1,0])

        #outputs = np.squeeze(outputs, axis = 2) # (max_len, bs, 1)
        #captions = np.argmax(outputs, axis = 2) # (max_len, bs)
        #captions = np.transpose(captions, axes=[1,0]) # (bs, max_len)
        captions = tokenizer.sequences_to_texts(captions)

        # Convert one-hot targets to captions
        target_sentences = targets_to_sentences(target) 
        
        for i, v in enumerate(captions):
            candidates.append( (v, keys[i]) )

    print("Candidate captions generated:", len(candidates))


    ## Compute BLEU scores
    for i, cand in enumerate(candidates):
        bleu_scores.append( compute_bleu(cand) )

    bleu_scores = np.array(bleu_scores)

    table = [['BlEU-1', np.mean(bleu_scores[:,0]), np.mean(bleu_scores[:,4])],
            ['BLEU-2', np.mean(bleu_scores[:,1]), np.mean(bleu_scores[:,5])],
            ['BLEU-3', np.mean(bleu_scores[:,2]), np.mean(bleu_scores[:,6])],
            ['BLEU-4', np.mean(bleu_scores[:,3]), np.mean(bleu_scores[:,7])]]

    print(tabulate(table, headers=['Individual', 'Cumulative']))
    

def model_eval(nr_of_batches = 1):
    """ Evaluate the model on some data

    Outputs
    -------
        - prints the produced candidate caption + its target caption for a given input
        - Saves the relevant NSD images together with the candidate caption 
    """


    for i in range(nr_of_batches):
#        sample = val_generator.__next__()
        sample = val_generator[0]
        features, _, a0, c0, _ = sample[0]
        target = sample[1]
        keys = sample[2]

        start_seq = np.repeat([tokenizer.word_index['<start>']], features.shape[0])

        outputs = model.greedy_predict(features, tf.convert_to_tensor(a0), tf.convert_to_tensor(c0), start_seq, config['max_length'], config['units'], tokenizer) 
        # outputs: (max_len, bs, 1, 5001)

        captions = np.squeeze(outputs)
        #outputs = np.squeeze(outputs, axis = 2) # (max_len, bs, 5001)
        #captions = np.argmax(outputs, axis = 2) # (max_len, bs)

        captions = np.transpose(captions, axes=[1,0]) # (bs, max_len)
        captions = tokenizer.sequences_to_texts(captions)

        # TODO: Remove everything after <end> token
        
        bleu = []
        for i, cap in enumerate(captions):
            bleu.append(compute_bleu((cap, keys[i])))

        # Convert one-hot targets to captions
        target_sentences = targets_to_sentences(target) 

        def remove_pad(caption):
            x = caption.split(" ")
            try:
                while True:
                    x.remove('<pad>')
            except ValueError:
                # No pad token present so we can continue
                pass
            return " ".join(x)

        for k, v in enumerate(captions):
            print()
            print("Candidate:", remove_pad(v))
            print("Target:   ", remove_pad(target_sentences[k]))
            print("BLEU:     ", f"{(bleu[k][0]):.2f} {(bleu[k][1]):.2f} {(bleu[k][2]):.2f} {(bleu[k][3]):.2f}")
            print("NSD:", keys[k])
            
            img = nsd_loader.read_images(int(keys[k])-1)
            fig = plt.figure()
            plt.imshow(img)
            plt.title(f"{remove_pad(v)}\n{remove_pad(target_sentences[k])}")
            plt.savefig(f"{out_path}/img_{keys[k]}.png")
            plt.close(fig)
        
    return



if __name__ == '__main__':
    nr_batchs = 1
    model_eval(nr_batchs)
    #eval_full_set()













