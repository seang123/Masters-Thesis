import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
#import DataLoaders.load_avg_betas as loader
from scipy import stats
import sys
import nibabel as nb
#import cortex
import tbparse
from DataLoaders import load_avg_betas as loader
from nsd_access import NSDAccess
import subprocess
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu

def test_groups():
    groups = loader.groups
    print(len(groups))
    print(len(groups[0]))

    g, _ = loader.remove_groups(2, [0, 180, 359])
    print(len(g))
    print(len(g[0]))
    print(len(g[1]))
    print(len(g[2]))

test_groups()
raise


def categorical_sample():
    import tensorflow as tf
    x = [[0.1, 0.8, 0.1], [0.3, 0.4, 0.3]]
    samples = tf.random.categorical(tf.math.log(x), 1).numpy()
    print(samples.shape)
    print(samples)


def view_behav():
    df = pd.read_csv(f"~/NSD3/nsddata/ppdata/subj02/behav/responses.tsv", sep='\t')
    print(df.columns)
    print(df.shape)

    df_two = df.loc[df['SUBJECT'] == 2]
    print(df_two.shape)
    print(np.nansum(df_two['ISCORRECT']))



def gest_test_set_515():
    from ian_code import nsd_get_data as ngd
    nsd_dir = "/home/seagie/NSD3/"

    x = ngd.get_1000(nsd_dir)
    print("val: ", len(x))

    #x = loader.get_test_set()
    x = ngd.get_conditions_515(nsd_dir, 40)
    print("test:", len(x))

    #x = pd.DataFrame(x, columns=['nsd_key'])
    #print(x.head())

    #with open("./TrainData/test_nsd.csv", "w") as f:
    #    x.to_csv(f, index=False)





### Test cross entropy loss
def test_ce_loss():

    x1 = [0.1, 0.05, 0.05, 0.2, 0.6]
    x2 = [0.01, 0.02, 0.03, 0.01, 0.93]

    y = [0, 0, 0, 0, 1]

    import tensorflow as tf
    cce = tf.keras.losses.CategoricalCrossentropy()
    print("x1: ", cce(y, x1).numpy())
    print("x2: ", cce(y, x2).numpy())

    def ce(y, x):
        return -sum([y[i] * np.log(x[i]) for i in range(len(x))])

    print("x1: ", ce(y, x1))
    print("x2: ", ce(y, x2))


### Test BLEU scores
def test_bleu():
    ref = [['hello', 'this', 'is', 'a', 'test'], ['bleu', 'ocean', 'seagul', 'green', 'red']]
    hyp = ['hello', 'there', 'you', 'person', 'this']

    print(1/5)

    weights = [
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        #(1./1., 0., 0., 0.),
        #(1./2., 1./2., 0., 0.),
        #(1./3., 1./3., 1./3., 0.),
        #(1./4., 1./4., 1./4., 1./4.)
    ]

    for w in weights:
        b = sentence_bleu(ref, hyp, weights = w)
        print(b)



def test_tokenizer(): 
    tokenizer2 = loader.load_tokenizer()
    #tokenizer, _ = loader.build_tokenizer(np.arange(1, 73001), 5000)

    #js = tokenizer.to_json()
    #import json
    #with open('tokenizer_73k.json', 'w') as f:
    #    json.dump(js, f)


    #d = tokenizer.word_counts
    #print(type(d))

    for i in range(50):
        #print(tokenizer.index_word[i], tokenizer2.index_word[i])
        print(tokenizer2.index_word[i])

test_tokenizer()
raise

### Generate Val/Test split
def get_val_split():
    x = np.arange(0, 1000)
    np.random.shuffle(x)

    val_split = x[:250]

    with open(f"./TrainData/val_split.txt", "w") as f:
        for i in val_split:
            f.write(f"{i}\n")







#
# Check coco info related to mscoco image_id
#
def coco_info():
    nsd_loader = NSDAccess("/home/seagie/NSD3/")
    nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)
    print("NSDAccess loader initialized ... ")

    # Captions
    target = nsd_loader.read_image_coco_info([key-1])
    print(target)
    print(target[0]['caption'])

    # Image
    img = nsd_loader.read_images(key-1)
    fig = plt.figure()
    plt.imshow(img)
    plt.title(f"key: {key}")
    plt.savefig(f"./temp_img_key{key}.png")
    plt.close(fig)
