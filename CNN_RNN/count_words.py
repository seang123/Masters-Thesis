from nsd_access import NSDAccess
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import utils
import re
import time
import nltk
import msgpack as mp
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#
# There are 4_535_976 total words (including <start> <end)) 
# They are made up of 29_437 unique words (including <start> <end>)
#
annt_dict = utils.load_json("../modified_annotations_dictionary.json")


captions = []

# Put all captions into a single list
for i in range(0, 73000):
    captions.extend(annt_dict[str(i)])

## Compute the count of lengths
lengths = {}
for i in range(0, len(captions)):
    l = len(captions[i])
    if str(l) not in lengths:
        lengths[str(l)] = 1
    else:
        lengths[str(l)] += 1

lengths = dict(sorted(lengths.items()))


fig = plt.figure(figsize=(20,7)
plt.bar(list(lengths.keys()), lengths.values())
plt.title("Distribution of caption lengths")
plt.xlabel("Length")
plt.ylabel("Count")
plt.savefig("./distribution_of_captions.png")
plt.close(fig)



tic = time.time()

# Split on space for all captions - gives a list of all words 
words = []
for i in range(0, len(captions)):
    words.extend(re.split(r'\s+', captions[i]))

for i in range(0, len(words)):
    words[i] = words[i].lower()

print(f"split: {(time.time() - tic):.2f} sec")

# get unique words
unique = set(words) 

print(f"there are {len(captions):,} captions")
print(f"there are {len(words):,} words")
print(f"there are {len(unique):,} unique words")

n_common = 20
print(f"\n{n_common} most common words")
freq_dist = nltk.FreqDist(words)
sorted_freq_dist = {k: v for k, v in sorted(freq_dist.items(), key=lambda item: item[1], reverse=True)}

# print n most common words
c = 0
for i, j in sorted_freq_dist.items():
    print(i, "--", j)
    if c > n_common:
        break
    c += 1


