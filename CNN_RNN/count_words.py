from nsd_access import NSDAccess
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import utils
import re
import time

#
# There are 4_535_976 total words (including <start> <end)) 
# They are made up of 29_437 unique words (including <start> <end>)
#
annt_dict = utils.load_json("../modified_annotations_dictionary.json")


ls = []

# Put all captions into a single list
for i in range(0, 73000):
    ls.extend(annt_dict[str(i)])

tic = time.time()

# Split on space for all captions - gives a list of all words 
lss = []
for i in range(0, len(ls)):
    lss.extend(re.split(r'\s+', ls[i]))


print(f"split: {time.time() - tic}")

# get unique words
unique = set(lss) 

print(f"there are {len(ls):,} captions")
print(f"there are {len(lss):,} words")
print(f"there are {len(unique):,} unique words")
