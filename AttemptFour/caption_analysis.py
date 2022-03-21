import pandas as pd
import numpy as np
import json
from DataLoaders import load_avg_betas as loader

captions_path = "/fast/seagie/data/captions/"
keys = range(1,73001)

top_k = 5000


def unique_words():
    import tensorflow as tf
    with open("./TrainData/tokenizer_73k.json", 'r') as f:
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.load(f))

    words_tuple = list(tokenizer.word_counts.items())
    words_tuple = sorted(words_tuple, key=lambda x: x[1], reverse=True)

    vocab = words_tuple[:top_k]

    # Load subj 2
    train_keys, val_keys = loader.get_nsd_keys('2')
    keys_sub2 = np.concatenate((train_keys, val_keys))
    print(keys_sub2.shape)
    print("\n Sub2 keys ")
    tokenizer_sub2, _ = loader.build_tokenizer(keys_sub2, top_k)
    words_tuple_sub2 = list(tokenizer_sub2.word_counts.items())
    words_tuple_sub2 = sorted(words_tuple_sub2, key=lambda x: x[1], reverse = True)
    vocab_2 = words_tuple_sub2[:top_k]

    overlap = 0
    total = 0
    for i, v in enumerate(vocab):
        total += 1
        for k, e in enumerate(vocab_2):
            if v[0] == e[0]:
                overlap += 1
                break

    print("overlap:", overlap)
    print("total:  ", total)
    print("overlap/total", overlap/total)


def statistics():
    caption_length = []
    caption_count = 0
    all_captions = []

    d = {}

    keys = np.arange(1, 73001)

    c = 0
    for idx, key in enumerate(keys):
        with open(f"{captions_path}/KID{key}.txt") as f:
            content = f.read()
            for cid, i in enumerate(content.splitlines()):
                cap = i.replace(".", " ").replace(",", " ").strip().split(" ")
                cap = [i.lower() for i in cap if i != '']
                cap2 = ['<start>'] + cap + ['<end>']
                caption_length.append( len(cap) )
                caption_count += 1
                all_captions.append(cap)
                d[c] = {'raw':len(cap), 'token':len(cap2)}
                cap = " ".join(cap)
                c += 1

    df = pd.DataFrame.from_dict(d, "index")

    print("avg. caption length:", sum(caption_length)/caption_count)

    print("Min:", min(caption_length))
    print("Max:", max(caption_length))
    print("Mean:", np.mean(caption_length))
    print("----")

    print(df.describe(percentiles = [.25, .5, .75, .9, .99]))

if __name__ == '__main__':
    statistics()

    #unique_words()
