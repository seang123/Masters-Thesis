
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import yaml
import pandas as pd
import time
import os, sys
import tensorflow as tf
import numpy as np
from Model import NIC
#from DataLoaders import load_avg_betas as loader
from DataLoaders import load_images as loader
from nsd_access import NSDAccess

gpu_to_use = 2

# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')

with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)
    print(f"Config file loaded.")

run_name = config['run']
out_path = os.path.join(config['log'], run_name, 'eval_out')

## Parameters
vocab_size = config['top_k'] + 1
batch_size = 10 # config['batch_size'] # 1 # TODO: .predict() doesnt seem to work with batch_size > 1


if not os.path.exists(out_path):
    os.makedirs(out_path)
    print(f"Creating evaluation output dir: {out_path}")
else:
    print(f"Evaluation output to dir: {out_path}")


## Load data 
data_train, train_vector, data_val, val_vector, tokenizer, train_keys, val_keys = loader.load_data_img(top_k = config['top_k'], _max_length = config['max_length'], train_test_split = 0.9)

def create_generator(data, cap, keys):
    return loader.data_generator(data, cap, keys, config['units'], config['top_k'], batch_size, training=False)

val_generator = create_generator(data_val, val_vector, val_keys)

print("data loaded successfully")

## Set-up model
model = NIC.NIC(
        config['input']['vc'],
        config['units'], 
        config['embedding_dim'], 
        vocab_size,
        config['max_length'],
        config['dropout'],
        config['input_reg'],
        config['lstm_reg'],
        config['output_reg']
        )

x = np.random.uniform(0, 1, size=(config['batch_size'], config['input']['mscoco'])).astype(dtype=np.float32)
y = np.random.randint(0, 2, size=(config['batch_size'], config['max_length']), dtype=np.int32)
z1 = np.random.uniform(0, 1, size=(config['batch_size'], config['units'])).astype(dtype=np.float32)
z2 = np.random.uniform(0, 1, size=(config['batch_size'], config['units'])).astype(dtype=np.float32)
model(x,y,z1,z2)
print("model built")


## Restore model from Checkpoint
model_dir = f"{os.path.join(config['log'], config['run'])}/model/model-latest.h5"
model.load_weights(model_dir,by_name=True,skip_mismatch=True)
print(f"Model weights loaded")

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
        sample = val_generator.__next__()
        features, _, a0, c0 = sample[0]
        target = sample[1]
        keys = sample[2]

        start_seq = np.repeat([tokenizer.word_index['<start>']], features.shape[0])
        print("start_seq:   ", start_seq.shape)

        outputs = model.greedy_predict(features, tf.convert_to_tensor(a0), tf.convert_to_tensor(c0), start_seq, config['max_length'], config['units']) # (10, 128, 1, 5001)

        outputs = np.squeeze(outputs, axis = 2) # (10, 128, 5001)

        captions = np.argmax(outputs, axis = 2) # (10, 128)
        captions = np.transpose(captions, axes=[1,0]) # (128, 10)
        captions = tokenizer.sequences_to_texts(captions)

        # Convert one-hot targets to captions
        target_sentences = targets_to_sentences(target) 

        for k, v in enumerate(captions):
            print()
            print("Candidate:", v)
            print("Target:   ", target_sentences[k])
            print("NSD:", keys[k])
            
            # TODO: save nsd image with title =v
            img = nsd_loader.read_images(keys[k])
            fig = plt.figure()
            plt.imshow(img)
            plt.title(v)
            plt.savefig(f"{out_path}/img_{keys[k]}.png")
            plt.close(fig)
        


    return



if __name__ == '__main__':
    model_eval(1)













