from nsd_access import NSDAccess
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import utils
import re
import time
import nltk
import msgpack as mp

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

for i in range(0, len(lss)):
    lss[i] = lss[i].lower()

print(f"split: {(time.time() - tic):.2f} sec")

# get unique words
unique = set(lss) 

print(f"there are {len(ls):,} captions")
print(f"there are {len(lss):,} words")
print(f"there are {len(unique):,} unique words")

n_common = 20
print(f"\n{n_common} most common words")
freq_dist = nltk.FreqDist(lss)
sorted_freq_dist = {k: v for k, v in sorted(freq_dist.items(), key=lambda item: item[1], reverse=True)}

# print n most common words
c = 0
for i, j in sorted_freq_dist.items():
    print(i, "--", j)
    if c > n_common:
        break
    c += 1


