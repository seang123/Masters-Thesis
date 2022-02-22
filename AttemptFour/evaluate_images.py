
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import yaml
import pandas as pd
import time
import os, sys
import tensorflow as tf
import tqdm
import numpy as np
from Model import NIC
from Model import img_NIC
#from DataLoaders import load_avg_betas as loader
from DataLoaders import load_avg_betas as loader
from DataLoaders import data_generator_image as generator
from nsd_access import NSDAccess
import argparse

gpu_to_use = 0

# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')

# Arg parser
parser = argparse.ArgumentParser(description='Evaluate NIC model')
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--e', type=int, required=True)
args = parser.parse_args()

with open(f"./Log/{args.dir}/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    print(f"Config file loaded: \n\t{f.name}")

run_name = config['run']
out_path = os.path.join(config['log'], run_name, 'eval_out')

# Seed
np.random.seed(config['seed'])
tf.random.set_seed(config['seed'])

## Parameters
vocab_size = config['top_k'] + 1
batch_size = 10 # config['batch_size'] # 1 # TODO: .predict() doesnt seem to work with batch_size > 1


if not os.path.exists(out_path):
    os.makedirs(out_path)
    print(f"Creating evaluation output dir:\n\t{out_path}")
else:
    print(f"Evaluation output to dir:\n\t{out_path}")


## Load data 
train_keys, val_keys = loader.get_nsd_keys('2')
tokenizer, _ = loader.build_tokenizer(np.arange(1, 73001), config['top_k'])
# Pairs
train_pairs = np.array(loader.create_pairs(train_keys))
val_pairs = np.array(loader.create_pairs(val_keys))
print(f"train_pairs: {len(train_pairs)}")
print(f"val_pairs  : {len(val_pairs)}")

def remove_dup_pairs(pairs):
    """ Remove duplicates from the pairs list, based on NSD key """
    print("Removing duplicates from pairs list ... ")
    return list({v[0]:v for v in pairs}.values())

train_pairs = remove_dup_pairs(train_pairs)
val_pairs   = remove_dup_pairs(val_pairs)
print(f"train_pairs: {len(train_pairs)}")
print(f"val_pairs  : {len(val_pairs)}")

print("data loaded successfully")
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
print("Generator initialised - nr. batches:", len(val_generator))

## Set-up model
model = img_NIC.NIC(
        config['group_size'],
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

init_generator = generator.DataGenerator(
        val_pairs,
        batch_size, 
        tokenizer, 
        config['units'], 
        config['max_length'], 
        vocab_size, 
        pre_load_betas=False,
        shuffle=False, 
        training=False)
data = init_generator.__getitem__(0)
model(data[0], False)
print("- Model built -")


## Restore model from Checkpoint
#model_dir = f"{os.path.join(config['log'], config['run'])}/model/model-latest.h5"
model_dir = f"{os.path.join(config['log'], config['run'])}/model/model-ep{args.e:03}.h5"
model.load_weights(model_dir,by_name=True,skip_mismatch=True)
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
    

def eval_model():
    """ Run inference on the image NIC model 
   
    Stores the output (int matrix) to .npy file 
    Stores the attention maps (float matrix | if exists) to .npy file
    """

    all_outputs = []
    all_attention_scores = []
    print("Generator length:", len(val_generator))
    print("Max length:", config['max_length'])
    for i in tqdm.tqdm(range(0, len(val_generator)+1)):
        sample = val_generator[i]
        features, _, a0, c0 = sample[0]
        target = sample[1]
        keys = sample[2]

        start_seq = np.repeat([tokenizer.word_index['<start>']], features.shape[0])

        outputs, attention_scores = model.greedy_predict(
                features, 
                tf.convert_to_tensor(a0), 
                tf.convert_to_tensor(c0), 
                start_seq, 
                config['max_length'], 
                config['units'], 
                tokenizer
        )
        all_outputs.append(outputs)
        all_attention_scores.append(attention_scores)


    outputs = np.concatenate((all_outputs), axis=0)
    print("outputs:", outputs.shape)
    if all_attention_scores[0] != None:
        attention_scores = np.swapaxes(np.concatenate((all_attention_scores), axis=1), 0, 1)
        print("attn_scores:", attention_scores.shape)
        with open(f"{out_path}/attention_scores.npy", "wb") as f:
            np.save(f, attention_scores)

    with open(f"{out_path}/output_captions.npy", "wb") as f:
        np.save(f, outputs)
    with open(f"{out_path}/tokenizer.json", "w") as f:
        f.write(tokenizer.to_json())

    return outputs, attention_scores

if __name__ == '__main__':
    #batch_eval(1)
    eval_model()













