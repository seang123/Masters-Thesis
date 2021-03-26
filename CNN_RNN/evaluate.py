# Evaluation of the CNN-RNN network

# imports
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from nsd_access import NSDAccess
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import utils
import time
import json
import pandas as pd
import logging
import tensorflow as tf
import string
import time
import tqdm
from itertools import zip_longest
import h5py
import random
import collections
from model import CNN_Encoder, RNN_Decoder
import datetime

def evaluate(img_name_val):
    """
    Evaluate an image
    """
    rid = np.random.randint(0, len(img_name_val)):

