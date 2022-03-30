import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
np.random.seed(6)
import os, sys
import tqdm
from scipy import stats

colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Compare the correlation of the 5 caption guse with the inference guse
# a. Compute correlation between 5 captions -> average
# b. Compute correlation between 5 captions and inference -> average

# 1. Load captions (pre-process as during training)
# 2. Get the inference model output (tokenizer)
# 3. guse embed inference output 
# 4. Compare

gpu_to_use = 0
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')

guse_sim = '/fast/seagie/data/subj_2/subj_2_guse_pre_processed.npy'
log_dir = './Log/proper_split_sub2/eval_out'
output_file = log_dir + '/output_captions_train.npy'
tokenizer_loc = f"{log_dir}/tokenizer.json"

with open(tokenizer_loc, "r") as f:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
print("Tokenizer loaded ...")

def output_to_caption(output):
    def clean(caption):
        x = caption.split(" ")
        x = [i for i in x if i != '<pad>' and i != '<end>']
        return " ".join(x)
    captions = tokenizer.sequences_to_texts(output)
    captions = [[clean(i)] for i in captions]
    return captions

def compute_inference_guse(captions):
    def get_google_encoder(sem_dir):
        """ Load/download the GUSE embedding model """
        export_module_dir = os.path.join(sem_dir, "google_encoding")

        if not os.path.exists(export_module_dir):
            module_url = \
                "https://tfhub.dev/google/universal-sentence-encoder/4"
            model = hub.load(module_url)
            tf.saved_model.save(model, export_module_dir)
            print(f"module {module_url} loaded")
        else:
            model = hub.load(export_module_dir)
            print(f"module {export_module_dir} loaded")

        return model
    def get_GUSE_embeddings(sem_dir, sentences, model=None):
        """ Return the GUSE embedding vector """
        if model is None:
            model = get_google_encoder(sem_dir)
        embeddings = model(sentences)
        return embeddings

    #if os.path.exists(f"{log_dir}/inference_guse.npy"):
    #    print("loading inference guse from disk")
    #    return np.load(f"{log_dir}/inference_guse.npy")
    #else:
    GUSE_model_path = './GUSE_model'
    GUSE_model = get_google_encoder(GUSE_model_path)
    guse = np.array([np.array(get_GUSE_embeddings(GUSE_model_path, x, GUSE_model)) for x in captions])
    print("guse:", guse.shape)
    return guse


if __name__ == '__main__':
    output = np.squeeze(np.load(output_file), axis=-1)
    print("output:", output.shape)
    captions = output_to_caption(output)

    indices = np.random.randint(0, 9000, 1000)

    #guse_inf = compute_inference_guse(captions) # (1000, 1, 512)
    guse_inf = np.load(f"{log_dir}/inference_guse_train.npy")#[indices]
    print("guse_inf:", guse_inf.shape)

    #with open(f"{log_dir}/inference_guse_train.npy", "wb") as f:
    #    np.save(f, guse_inf)

    guse_train = np.load(guse_sim)[:9000]#[indices] # disk_data = (10000, 5, 512)
    print("guse_train:", guse_train.shape)

    correlations = []
    for trial in tqdm.tqdm(range(guse_train.shape[0])):
        corr = np.corrcoef(guse_train[trial, :, :])
        mean_corr = np.mean(corr, axis=(0,1))
        correlations.append(mean_corr)

    correlations = np.array(correlations)
    print("corrcoef:", correlations.shape)
    print("----")

    # Consistency 
    correlations_inf = []
    for trial in tqdm.tqdm(range(guse_train.shape[0])):
        t = np.concatenate((guse_train[trial,:,:], guse_inf[trial, :, :]), axis=0)
        t = np.corrcoef(t)
        t = np.mean(t[-1][:-1])
        correlations_inf.append(t)


    correlations_inf = np.array(correlations_inf) # (1000, 1000)
    print(correlations_inf.shape)

    m, b = np.polyfit(correlations, correlations_inf, 1)

    r, p = stats.pearsonr(correlations, correlations_inf)
            
    fig = plt.figure(figsize=(16,9))
    plt.scatter(correlations, correlations_inf)
    plt.plot(correlations, m*correlations + b, c=colours[1])
    plt.title(f"Correlation between inference captions and the 5 target captions for that sample\nPearson’s corr. coef.: {r:.3f}")
    plt.ylabel("Brain-caption correlation")
    plt.xlabel("Caption consistency")
    plt.savefig("./corr.png")
    plt.close(fig)
            
