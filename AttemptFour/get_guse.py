import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import time

GUSE_model_path = './GUSE_model'
nsd_keys = './TrainData/subj02_conditions.csv'
captions_path = '/huge/seagie/data/subj_2/captions/'

def get_captions(keys: list):
    """ Given a list of NSD keys, return the relevent captions 

    Parameters
    ----------
        keys : list
            NSD keys
    Returns
    -------
        all_captions : nested list
            all captions for the specified keys
    """

    all_captions = []
    for i, key in enumerate(keys):
        captions = []
        with open(f"{captions_path}/SUB2_KID{key}.txt", "r") as f:
            content = f.read()
            for line in content.splitlines():
                captions.append(line)

        assert len(captions) == 5, f"{len(captions)}\n{key}"
        all_captions.append( captions )
    
    return all_captions

def get_google_encoder(sem_dir):
    """ Load/download the GUSE embedding model
    """
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
    """ Return the GUSE embedding vector
    """
    if model is None:
        model = get_google_encoder(sem_dir)
    embeddings = model(sentences)
    return embeddings


def create_single(keys):

    g = np.zeros((50000, 512))
    for i, key in enumerate(keys):
        for c in range(5):
            with open(f"/huge/seagie/data/subj_2/guse/guse_embedding_KID{key}_CID{c}.npy", "rb") as f:
                g[i * c, :] = np.load(f)
                
    with open(f"/huge/seagie/data/subj_2/guse/guse_embeddings_flat.npy", "wb") as f:
        np.save(f, g)


if __name__ == '__main__':
    """
    Every caption gets a GUSE embedding (512,) since we have 5 captions per NSD key
    the guse for a single key is (5,512)
    """

    df = pd.read_csv(f"{nsd_keys}")
    nsd_keys = list( df['nsd_key'].values )

    sample_captions = get_captions(nsd_keys) ### a list of captions that you chose ie. [6,37,45]
    print("sample_captions", len(sample_captions))

    create_single(nsd_keys)
    raise

    #GUSE_model = get_google_encoder(GUSE_model_path)
    #guse = np.array([np.array(get_GUSE_embeddings(GUSE_model_path, x, GUSE_model)) for x in sample_captions])

    with open(f"/huge/seagie/data/subj_2/guse/guse_embeddings.npy", "rb") as f:
        guse = np.load(f)
    print(guse.shape)

    for i, key in enumerate(nsd_keys):
        for cap_idx in range(5):
            with open(f"/huge/seagie/data/subj_2/guse/guse_embedding_KID{key}_CID{cap_idx}.npy", "wb") as f:
                np.save(f, guse[i,cap_idx,:])

    # Also save them all together
    #with open(f"/huge/seagie/data/subj_2/guse/guse_embeddings.npy", "wb") as f:
    #    np.save(f, guse)
    #    print(f"embeddings saved to {f.name}")

