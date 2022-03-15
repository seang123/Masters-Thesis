# Run the pycocoevalcap.py metric suit on the output of the model.
# Candidate outputs need to be stored as [{'image_id': id, 'caption': ""} ...] in .json file

# 1. Take the outputs.npy and tokenizer.json and convert the tokenized candidates to text captions
# 2. Store the captions to disk
# 3. Run analysis

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import argparse
import os
import pandas as pd
import numpy as np
import json
import tensorflow.keras.preprocessing.text as keras_text
from nsd_access import NSDAccess



# ArgParser
parser = argparse.ArgumentParser(description="Metric Suit")
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--e', type=str, required=False)
args = parser.parse_args()

# Data dir
data_loc = os.path.join('Log', args.dir, 'eval_out')
output_loc = f"{data_loc}/output_captions.npy"
if args.e != None: output_loc = f"{data_loc}/output_captions_{args.e}.npy"
annotation_file = f"/home/seagie/NSD3/nsddata_stimuli/stimuli/nsd/annotations/captions_train2017.json" # mscoco train-set contains all validation images
#annotation_file = f"./captions_val2014.json"
results_file = f"{data_loc}/captions_results.json"
tokenizer_loc = f"{data_loc}/tokenizer.json"

# Validation data NSD keys
conditions = pd.read_csv(f"./TrainData/subj02_conditions.csv")
val_keys = conditions.loc[conditions['is_shared'] == 1]
val_keys = val_keys.reset_index(drop=True)

# NSD_access (not needed, just for testing)
nsd_loader = NSDAccess("/home/seagie/NSD3/")
nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)
print("NSDAccess loader initialized ... ")

# Data loader
def load_data():
    with open(output_loc, "rb") as f:
        outputs = np.load(f)
    return outputs

def remove_end_pad(caption):
    x = caption.split(" ")
    x = [i for i in x if i != '<pad>' and i != '<end>']
    return " ".join(x)


def create_data_json(captions):
    """ Store the captions as json 
    Parameters:
    -----------
        captions: ndarray
            tokenized candidate captions (1000, 13, 1)
    """
    captions = np.squeeze(captions, axis=-1)
    with open(tokenizer_loc, "r") as f:
        tokenizer = keras_text.tokenizer_from_json(f.read())
    
    # Tokenized captions -> Text captions
    captions = tokenizer.sequences_to_texts(captions)

    # We need to get the overall COCO image_id (different from the NSD keys) 
    targets = nsd_loader.read_image_coco_info(list(val_keys['nsd_key']-1)) # len == 1000 len(0) == 5

    results = []
    for i, key in enumerate(val_keys['nsd_key']):
        mod_cap = remove_end_pad(captions[i])
        results.append( {"image_id": targets[i][-1]['image_id'], "caption": mod_cap} )

    with open(f"{results_file}", "w") as f:
        json.dump(results, f)

    return


def run_metrics():
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
            print(f'{metric}: {score:.3f}')

    return

if __name__ == '__main__':
    captions = load_data()
    create_data_json(captions)
    run_metrics()
