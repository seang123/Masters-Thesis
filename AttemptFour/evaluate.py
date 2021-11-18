
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import yaml
import pandas as pd
import time
import os, sys
import tensorflow as tf
import numpy as np
from Model import NIC, lc_NIC
from DataLoaders import load_avg_betas as loader
from DataLoaders import data_generator_guse as generator
from nsd_access import NSDAccess

gpu_to_use = 2

# Allow memory growth on GPU device
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')

with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)
    print(f"Config file loaded.")

run_name = config['run']
out_path = os.path.join(config['log'], run_name, 'eval_out')

np.random.seed(config['seed'])
tf.random.set_seed(config['seed'])

## Parameters
vocab_size = config['top_k'] + 1
batch_size = 5 # config['batch_size'] # 1 # TODO: .predict() doesnt seem to work with batch_size > 1


if not os.path.exists(out_path):
    os.makedirs(out_path)
    print(f"Creating evaluation output dir: {out_path}")
else:
    print(f"Evaluation output to dir: {out_path}")


## Load data
tokenizer, _ = loader.build_tokenizer(config['dataset']['captions_path'], config['top_k'])

nsd_keys, shr_nsd_keys = loader.get_nsd_keys(config['dataset']['nsd_dir'])

train_keys = nsd_keys
val_keys = shr_nsd_keys
print("mixing train and val sets")
all_keys = np.concatenate((train_keys, val_keys))
np.random.shuffle(all_keys)
train_keys = all_keys[:9000]
val_keys = all_keys[9000:]
print("train_keys",train_keys.shape)
print("val_keys", val_keys.shape)

train_pairs = loader.create_pairs(train_keys, config['dataset']['captions_path'])
val_pairs   = loader.create_pairs(val_keys, config['dataset']['captions_path'])

print(f"train_pairs: {len(train_pairs)}")
print(f"val_pairs  : {len(val_pairs)}")

create_generator = lambda pairs, training: loader.lc_batch_generator(pairs,
            config['dataset']['betas_path'],
            config['dataset']['captions_path'],
            tokenizer,
            batch_size,
            config['max_length'],
            vocab_size,
            config['units'],
            training = training,
        )

#val_generator = create_generator(train_pairs, training=False)
val_generator = generator.DataGenerator(
        train_pairs, 
        batch_size, 
        tokenizer, 
        config['units'], 
        config['max_length'], 
        vocab_size, 
        nsd_keys = train_keys, 
        pre_load_betas=False,
        shuffle=True, 
        training=False)

#x = val_generator[0]
#x2 = val_generator[1]

print("data loaded successfully")

## Set-up model
model = lc_NIC.NIC(
        loader.get_groups(config['embedding_features'])[0],
        loader.get_groups(config['embedding_features'])[1],
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
        config['input_reg'],
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
#model_dir = f"{os.path.join(config['log'], config['run'])}/model/model-ep011.h5"

model.load_weights(model_dir,by_name=True,skip_mismatch=True)
#model.load_weights(model_dir)
print(f"Model weights loaded")
print(f" - from {model_dir}")


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
    

def model_eval(nr_of_batches = 1):
    """ Evaluate the model on some data

    Outputs
    -------
        - prints the produced candidate caption + its target caption for a given input
        - Saves the relevant NSD images together with the candidate caption 
    """
    nsd_loader = NSDAccess("/home/seagie/NSD")
    nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)


    for i in range(nr_of_batches):
#        sample = val_generator.__next__()
        sample = val_generator[0]
        features, _, a0, c0, _ = sample[0]
        target = sample[1]
        keys = sample[2]

        start_seq = np.repeat([tokenizer.word_index['<start>']], features.shape[0])
        print("start_seq:   ", start_seq.shape)
        print(start_seq)

        outputs = model.greedy_predict(features, tf.convert_to_tensor(a0), tf.convert_to_tensor(c0), start_seq, config['max_length'], config['units'], tokenizer) 
        # outputs: (max_len, bs, 1, 5001)

        outputs = np.squeeze(outputs, axis = 2) # (max_len, bs, 1)

        captions = np.argmax(outputs, axis = 2) # (max_len, bs)
        captions = np.transpose(captions, axes=[1,0]) # (bs, max_len)
        captions = tokenizer.sequences_to_texts(captions)

        # Convert one-hot targets to captions
        target_sentences = targets_to_sentences(target) 

        for k, v in enumerate(captions):
            print()
            print("Candidate:", v)
            print("Target:   ", target_sentences[k])
            print("NSD:", keys[k])
            
            img = nsd_loader.read_images(int(keys[k]))
            fig = plt.figure()
            plt.imshow(img)
            plt.title(v)
            plt.savefig(f"{out_path}/img_{keys[k]}.png")
            plt.close(fig)
        
    return



if __name__ == '__main__':
    nr_batchs = 1
    model_eval(nr_batchs)













