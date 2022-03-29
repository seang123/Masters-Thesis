import nibabel as nb
import cortex
import os, sys
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
from nsd_access import NSDAccess
import tensorflow as tf
from collections import defaultdict
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu

import sample_captions as sc

data_file = './Eval/one_shot/output_captions_41.npy'

nsd_loader = sc.init_nsd()


def load_data():
    with open(data_file, "rb") as f:
        data = np.squeeze(np.load(f), axis=-1)
    return data



if __name__ == '__main__':
    outputs = load_data()

    targets = sc.load_captions()
    bleu_score = sc.all_bleu_scores(outputs, targets)
