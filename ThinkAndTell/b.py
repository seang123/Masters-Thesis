import sys, os
import json
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import utils

annt_dict = utils.load_json("../modified_annotations_dictionary.json")

nr_cap = 0
for k, v in annt_dict.items():
    nr_cap += len(v)

print(nr_cap)
