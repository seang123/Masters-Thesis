import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import yaml
import pandas as pd
import time
import json
import os, sys
import tensorflow as tf
import numpy as np
import tqdm
from Model import img_NIC as lc_NIC
from DataLoaders import load_avg_betas as loader
from DataLoaders import data_generator_image as generator
from tabulate import tabulate
import argparse
from itertools import groupby

gpu_to_use = 1
# Allow memory growth on GPU device
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')

## ======= Arg parse =========
parser = argparse.ArgumentParser(description='Evaluate NIC model')
parser.add_argument('--dir', type=str, required=False)
parser.add_argument('--e', type=int, required=False)
args = parser.parse_args()

## ======= Parameters =========
model_name = "no_attn_loss_const_lr2"
epoch = 41
subject = '2'

if args.dir != None: model_name = args.dir
if args.e != None: epoch = args.e

model_dir = f"./Log/{model_name}/model/model-ep{epoch:03}.h5"
print("Model dir:   ", model_dir)

time.sleep(5)

## ======= Config =========

with open(f"./Log/{model_name}/config_img.yaml", "r") as f:
    config = yaml.safe_load(f)
    print(f"Config file loaded:\n\t {f.name}")

run_name = config['run']
#out_path = os.path.join(config['log'], run_name, 'eval_out')
out_path = './Eval/one_shot/'
out_path = f'./Log/{run_name}/eval_out'
if not os.path.exists(out_path):
    os.makedirs(out_path)
    print(f"Creating evaluation output dir:\n\t{out_path}")
else:
    print(f"Evaluation output to dir:\n\t{out_path}")

np.random.seed(config['seed'])
tf.random.set_seed(config['seed'])

vocab_size = config['top_k'] + 1
batch_size = 64

## ======= Load data =========

train_keys, val_keys, test_keys = loader.get_nsd_keys(subject)
print("train_keys:", train_keys.shape)
print("val_keys:", val_keys.shape)
print("test_keys:", test_keys.shape)

tokenizer = loader.load_tokenizer()
#tokenizer, _ = loader.build_tokenizer(np.arange(1, 73001), config['top_k'])
#tokenizer, _ = loader.build_tokenizer(np.concatenate((train_keys, val_keys)), config['top_k'])

train_pairs = loader.create_pairs(train_keys, single=True)
val_pairs   = loader.create_pairs(val_keys, single=True)
test_pairs   = loader.create_pairs(test_keys, single=True)
print(f"train_pairs: {len(train_pairs)}")
print(f"val_pairs  : {len(val_pairs)}")
print(f"test_pairs : {len(test_pairs)}")

def remove_dup_pairs(pairs):
    """ Remove duplicates from the pairs list, based on NSD key """
    return list({v[0]:v for v in pairs}.values())
#print("Removing duplicates from pairs list ... ")
#train_pairs = remove_dup_pairs(train_pairs)
#val_pairs   = remove_dup_pairs(val_pairs)
#test_pairs  = remove_dup_pairs(test_pairs)
#print(f"train_pairs: {len(train_pairs)}")
#print(f"val_pairs  : {len(val_pairs)}")

## ======= Data Generator =========
data_generator = generator.DataGenerator(
        test_pairs, 
        batch_size, 
        tokenizer, 
        config['units'], 
        config['max_length'], 
        vocab_size, 
        pre_load_betas=False,
        shuffle=False, 
        training=False)
print("len generator:", len(data_generator))

## ======= Model =========
model = lc_NIC.NIC(
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
        config['dropout_out'],
        config['input_reg'],
        config['attn_reg'],
        config['lstm_reg'],
        config['output_reg']
        )
# Build model
model(data_generator.__getitem__(0)[0], training=False)
print("--= Model built =--")

# Load weights
model.load_weights(model_dir,by_name=True,skip_mismatch=True)
print(f"Model weights loaded")
print(f" - from {model_dir}")


## ======= Evaluate Model =========

def eval_model():
    """ Runs the generators input through the model and returns the output and attention scores """

    all_outputs = []
    all_outputs_raw = []
    all_attention_scores = []

    for i in tqdm.tqdm(range(0, len(data_generator)+1)):
        sample = data_generator[i]
        features, _, a0, c0 = sample[0]
        target = sample[1]
        keys = sample[2]

        start_seq = np.repeat([tokenizer.word_index['<start>']], features.shape[0])

        outputs, outputs_raw, attention_scores = model.greedy_predict(features, tf.convert_to_tensor(a0), tf.convert_to_tensor(c0), start_seq, config['max_length'], config['units'], tokenizer) 
        all_outputs.append(outputs)
        all_outputs_raw.append(outputs_raw)
        all_attention_scores.append(attention_scores)

    # Concat the batches into one matrix
    outputs = np.concatenate((all_outputs), axis=0)
    outputs_raw = np.concatenate((all_outputs_raw), axis=0)
    attention_scores = np.swapaxes(np.concatenate((all_attention_scores), axis=1), 0, 1)

    print("outputs:", outputs.shape)
    print("outputs_raw:", outputs_raw.shape)
    print("attention scores:", attention_scores.shape)

    with open(f"{out_path}/output_captions_{epoch}.npy", "wb") as f:
        np.save(f, outputs)
    with open(f"{out_path}/output_captions_raw_{epoch}.npy", "wb") as f:
        np.save(f, outputs_raw)
    with open(f"{out_path}/attention_scores_{epoch}.npy", "wb") as f:
        np.save(f, attention_scores)
    with open(f"{out_path}/tokenizer.json", "w") as f:
        f.write(tokenizer.to_json())


    return outputs, attention_scores


if __name__ == '__main__':
    eval_model()
