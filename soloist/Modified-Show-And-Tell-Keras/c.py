from nsd_access import NSDAccess
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import my_utils as uu


annt_dict = uu.load_json("../../modified_annotations_dictionary.json")

unq_img_keys = []
shr_img_keys = []
with open("../../ThinkAndTell/keys/unq_img_keys.txt", "r") as f, open("../../ThinkAndTell/keys/shr_img_keys.txt", "r") as g:
    f_lines = f.readlines()
    for line in f_lines:
        unq_img_keys.append(int(line))

    g_lines = g.readlines()
    for line in g_lines:
        shr_img_keys.append(int(line))

train_keys = unq_img_keys
val_keys = shr_img_keys


captions_train = []
nr_captions_train = []
for i in train_keys:
    caps = annt_dict[str(i)]
    captions_train.append(caps)
    nr_captions_train.append(len(caps))

freq_table = {}
for i in range(0, len(captions_train)):
    for j in captions_train[i]:
        sentence = j.lower()
        words = sentence.split(" ")
        for w in words:
            freq_table[w] = freq_table.get(w, 0) + 1

print(len(freq_table.keys()))
print('<start>', freq_table['<start>'])
print('a', freq_table['a'])
print('man', freq_table['man'])
print('skateboard', freq_table['skateboard'])


freq_table = {k: v for k, v in sorted(freq_table.items(), key=lambda item: item[1])}

c = 0
for k, v in freq_table.items():
    print(k, v)
    c += 1
    if c >= 10:
        break













