import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd
from nsd_access import NSDAccess
import sys
sys.path.append("/home/seagie/NSD/Code/Masters-Thesis/AttemptFour/")
import tqdm
from DataLoaders import load_avg_betas as loader
import numpy as np
import nibabel as nb
import cortex

captions_path = "/fast/seagie/data/subj_2/captions/"
betas_path    = "/fast/seagie/data/subj_2/betas_averaged/"
out_dir = "./Visualization"

# Init NSDLoader
nsd_loader = NSDAccess("/home/seagie/NSD3/")
nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)
print("NSDAccess loader initialized ... ")

# Glasser setup
GLASSER_LH = '/home/danant/misc/lh.HCP_MMP1.mgz'
GLASSER_RH = '/home/danant/misc/rh.HCP_MMP1.mgz'
VISUAL_MASK = '/home/danant/misc/visual_parcels_glasser.csv'
## Load glasser regions
glasser_lh = nb.load(GLASSER_LH).get_data()
glasser_rh = nb.load(GLASSER_RH).get_data()
glasser = np.vstack((glasser_lh, glasser_rh)).flatten() # (327684,)

train_keys, val_keys = loader.get_nsd_keys('2')
print("train_keys:", train_keys.shape)
print("val_keys:", val_keys.shape)

def get_beta_by_key(key):
    with open(f"{betas_path}/subj02_KID{key}.npy", "rb") as f:
        return np.load(f)

def get_picture(key):
    return nsd_loader.read_images(int(key)-1)
def get_target(key):
    target = nsd_loader.read_image_coco_info([int(key)-1])
    return target[-1]['caption']

def vert(betas):
    vert = cortex.Vertex(betas, 'fsaverage')
    im, extents = cortex.quickflat.make_flatmap_image(vert)
    return im

def get_avg_betas(keys):
    betas = np.zeros((len(keys), 327684), dtype=np.float32)
    for k, v in enumerate(tqdm.tqdm(keys, total=len(keys))):
        betas[k,:] = get_beta_by_key(v)
    return betas

def avg_betas(name = 'training'):
    """ Plot the betas averaged across all training trials """
    
    if name == 'training':
        betas = get_avg_betas(train_keys)
    elif name == 'val':
        betas = get_avg_betas(val_keys)
    else:
        raise Exception("wrong name")

    #betas_mean = np.mean(betas, axis=0)# / betas.shape[0]
    #print("min", np.min(betas_mean))
    #print("max", np.max(betas_mean))

    betas_mean = np.sqrt(np.sum(np.power(betas, 2), axis=0))

    fig = plt.figure(figsize=(20,20), dpi = 100)
    plt.imshow(vert(betas_mean), cmap=plt.get_cmap('viridis'))
    plt.grid(False)
    plt.axis('off')
    plt.savefig(f"./avg_betas_{name}_trials.png", bbox_inches='tight')
    plt.close(fig)

def temp():

    ls = []
    for i in range(200):
        ls.append(get_target(train_keys[i]))

    for i, v in enumerate(ls):
        print(i, "-", v)
    

def single_trial():
    """ Plot betas and stimuli for a single trial """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(40, 40) ) # figsize=(w,h)
    fig.subplots_adjust(wspace=0, top=0.85)#, hspace=0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Idx 2 from val keys has very nice betas
    # 171 from train - plane
    # 12 from train - baseball catcher
    key = train_keys[12]
    print("Key:", key)

    axes[0].imshow(vert(get_beta_by_key(key)), cmap = plt.get_cmap('RdBu_r')) # 'RdBu'
    axes[0].grid(False)
    axes[0].axis('off')

    axes[1].imshow(get_picture(key))
    axes[1].grid(False)
    axes[1].axis('off')

    txt=get_target(key)
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=72)

    plt.savefig("example_beta_and_image.png")
    plt.close(fig)

    return

if __name__ == '__main__':
    #single_trial()
    #temp()
    avg_betas('training')
    #avg_betas('val')
